from openai import OpenAI
import os
import numpy
import json
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE=os.getenv("KNOWLEDGE_BASE")
VECTOR_STORE_FILE=os.getenv("VECTOR_STORE_FILE")

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embiggings(text):
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )

    return response.data[0].embedding

def add_embeddings(chunks):
    for chunk in chunks:
        chunk["vector"] = get_embiggings(chunk["text"])

    return chunks


def cosine_similarity(A,B):
    dot_product = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

def load_documents(folder=KNOWLEDGE_BASE):
    documents=[]

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path,"r",encoding="utf-8") as f:
            documents.append({
                "source":filename,
                "text": f.read()
            })

    return documents

def create_chunk(doc):
    chunks = []

    chunk_id = 0
    for part in doc["text"].split("\n\n"):
        part = part.strip()
        if part:
            chunks.append(
                {
                    "source":doc["source"],
                    "chunk_id":chunk_id,
                    "text":part,
                }
            )

            chunk_id+=1

    return chunks

def create_chunks():
    documents = load_documents()

    chunks = []
    for doc in documents:
        chunks.extend(create_chunk(doc))

    return chunks

def save_vector_store(chunks):
    with open(VECTOR_STORE_FILE, "w", encoding="utf-8") as f:
         json.dump(chunks,f,indent=4)

def load_vector_store():
    with open(VECTOR_STORE_FILE,"r",encoding="utf-8") as f:
        return json.load(f)

def search(query, top_k=3):
    query_vector = get_embiggings(query)

    chunks = []
    if os.path.exists(VECTOR_STORE_FILE):
        chunks = load_vector_store()
        print(f"Loaded {len(chunks)} from vector store {VECTOR_STORE_FILE}")
    else:
        chunks = create_chunks()
        chunks = add_embeddings(chunks)
        save_vector_store(chunks)
        print(f"Created & embedded chunks of len {len(chunks)} into vector store {VECTOR_STORE_FILE}")


    result = []

    for chunk in chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        result.append(
            {
                "score":float(score),
                "source":chunk["source"],
                "chunk_id":chunk["chunk_id"],
                "text":chunk["text"],
                "size":len(chunk["text"])
            }
        )

    result.sort(key= lambda chunk:chunk['score'], reverse=True)
    return result[:top_k]


print("how many leaves i have?")
print(search("how many leaves i have?"))

print()
print("how many medical leaves with travel bonus do i have?")
print(search("how many medical leaves with travel bonus do i have?"))