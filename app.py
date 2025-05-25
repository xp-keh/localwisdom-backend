from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
from groq import Groq
import json
from config.utils import get_env_value

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Load embedding model
model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db_poc_v2")
collection = chroma_client.get_collection(name="paper_data")

# Initialize GROQ API
groq_api_key = get_env_value("GROQ_API_KEY")

groq_client = Groq(api_key=groq_api_key)

def get_embedding(query):
    """Generate embedding for the given query."""
    embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    return embedding

def retrieve_documents(query: str, top_k: int = 5):
    query_embedding = get_embedding([query])[0].tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    retrieved_docs = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        id = results["ids"][0][i]
        retrieved_docs.append({
            "id": id,
            "title": metadata["title"],
            "disaster_type": metadata["disaster_type"],
            "page": metadata["page"],
            "reference": metadata["reference"],
            "text": doc,
        })

    return retrieved_docs

def expand_queries(original_query: str, num_queries: int = 3):
    """Generate multiple variations of the query using GROQ API."""
    prompt = f"""
    Buat {num_queries} variasi dari query yang ada dibawah, pastikan hasil pencarian query variasi tetap sama dengan query asli,
    Pastikan jawab langsung tanpa angka dan tanpa awalan apapun dan hanya dipisah berdasarkan "newline"
    
    Query Asli: "{original_query}"
    
    Query Variasi:
    """
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Kamu asisten pencari kueri variasi"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=200
    )

    queries = response.choices[0].message.content.split("\n")
    return [q.strip() for q in queries if q.strip()]

def multi_query_retrieval(original_query: str, num_queries: int = 3, top_k: int = 5):
    """Retrieve documents using multiple query variations."""
    queries = expand_queries(original_query, num_queries)
    
    retrieved_docs = []
    for query in queries:
        retrieved_docs.extend(retrieve_documents(query, top_k))

    # Remove duplicates based on text
    seen_texts = set()
    unique_docs = []
    for doc in retrieved_docs:
        if doc["text"] not in seen_texts:
            seen_texts.add(doc["text"])
            unique_docs.append(doc)

    return unique_docs[:top_k]

def generate_answer(query: str, context_docs: list):
    """Use GROQ API to generate an answer based on retrieved documents."""
    context_text = "\n\n".join([f"Judul:{doc['title']}\nHalaman:{doc['page']}\nReferensi:{doc['reference']}\nBencana:{doc['disaster_type']}\nKonten:\n{doc['text']}" for doc in context_docs])

    prompt = f"""
    Dekomposisi pertanyaan menjadi 3 bagian yaitu bencana, objek, dan peristiwa yang diamati. 

    Jika pertanyaan kurang lengkap atau tidak ada minimal 3 poin yaitu bencana, objek, dan peristiwa yang diamati, beri respons json error beserta alasannya.
    Contohnya jika tidak ada bencana dalam pertanyaan maka respons dengan:
    {{
        "error": "Tidak terdapat bencana, gagal menganalisis!"
    }}

    Contoh Pertanyaan: "Perubahan air warna air di sungai berarti tanda akan banjir"
    Contoh Proses Dekomposisi:
    - Bencana: banjir
    - Objek: sungai
    - Peristiwa: perubahan warna air

    Selanjutnya cocokkan dengan konteks yang diberikan dan berikan keluaran berupa pilihan berikut:
    - Tidak ada kesamaan dengan local knowledge di Indonesia (Jika sama sekali tidak ada kesamaan)
    - Ada kesamaan bencana (Jika ada kesamaan bencana yang diperkirakan datang seperti banjir)
    - Ada kesamaan objek (Jika ada kesamaan objek yang diamati misalkan sungai, angin, dll)
    - Ada kesamaan objek dan bencana (Jika ada kesamaan objek dan bencana misalkan mengamati angin untuk memperkirakan banjir)
    - Ada kesamaan local knowledge (Jika ada kesamaan objek, bencana dan peristiwa misalkan mengamati angin ke arah daratan untuk memperkirakan banjir)

    Berikan referensi (ref) **JIKA DAN HANYA JIKA** ada kesamaan bencana, objek, atau peristiwa antara konteks dengan pertanyaan!
    
    Berikan jawaban hanya dalam bentuk raw json seperti berikut (pastikan keluaran merupakan json valid, dan escape \" untuk semua reference yang menjelaskan judul), tanpa adanya code snippet:
    {{
        "decomposition": {{
            "bencana": "banjir",
            "objek": "sungai",
            "peristiwa": "perubahan warna air"
        }},
        "similarity": "Ada kesamaan local knowledge",
        "notes": "Tidak ada kesamaan objek dan peristiwa yang diamati, tetapi ada kesamaan bencana (banjir) dalam beberapa konteks yang diberikan.",
        "ref": [
            {{"title": "Budaya Di Indonesia", "disaster_type":"banjir", "page": "50-51", "reference":"I. G. A. Paramita, \"Bencana, Agama dan Kearifan Lokal\" Dharmasmrti, vol. 18, no. 1, pp. 36–44, Mei 2018."}},
            {{"title": "Budaya Di Yogyakarta", "disaster_type":"banjir", "page": "501", "reference":"I. G. A. Paramita, \"Bencana, Agama dan Kearifan Lokal\" Dharmasmrti, vol. 18, no. 1, pp. 36–44, Mei 2018."}},
            {{"title": "Budaya Di Parangtritis", "disaster_type":"banjir", "page": "50", "reference":"I. G. A. Paramita, \"Bencana, Agama dan Kearifan Lokal\" Dharmasmrti, vol. 18, no. 1, pp. 36–44, Mei 2018."}}
        ]
    }}
    
    Konteks:
    {context_text}

    Pertanyaan: {query}

    Jawaban:
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  
        messages=[
            {
                "role": "system",
                "content": 'Kamu asisten yang hanya menjawab berdasarkan konteks yang diberikan dan keluaran yang diminta'
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content

@app.route('/query', methods=['POST'])
def query_endpoint():
    """API endpoint to process user queries."""
    data = request.get_json()
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    retrieved_docs = multi_query_retrieval(query, num_queries=5)
    answer = generate_answer(query, retrieved_docs)
    try:
        data_dict = json.loads(answer)
    except:
        return jsonify({"error": "Internal Server Error"}), 500

    if data_dict.get("error"):
        return data_dict, 500

    return jsonify(data_dict)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)