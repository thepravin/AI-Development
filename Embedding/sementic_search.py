from openai import OpenAI
import os
import numpy
from dotenv import load_dotenv

load_dotenv()

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


def cosine_similarity(A,B):
    dot_product = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

text1 = "Python is easy programing language"
text2 = "Python language used for AI development"
text3 = "I am pravin I love Python programing language"

embedding1 = get_embiggings(text1)
embedding2 = get_embiggings(text2)
embedding3 = get_embiggings(text3)

similarity_12 = cosine_similarity(embedding1, embedding2)
similarity_13 = cosine_similarity(embedding1, embedding3)
similarity_23 = cosine_similarity(embedding2, embedding3)

print("Similarity between text1 and text2 : ", f"{similarity_12:.2f}")
print("Similarity between text1 and text3 : ", f"{similarity_13:.2f}")
print("Similarity between text2 and text3 : ", f"{similarity_23:.2f}")