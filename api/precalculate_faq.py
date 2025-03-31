import openai
import numpy as np
import json
import os
from dotenv import load_dotenv

# Cargar API key desde .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Asegúrate de tener esto en tu .env

# Cargar datos del FAQ desde el archivo JSON
with open("./api/faq_data.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

# Función para obtener embedding
def obtener_embedding(texto):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texto
    )
    return response["data"][0]["embedding"]

# Precalcular embeddings solo para las preguntas
faq_embeddings = {
    pregunta: obtener_embedding(pregunta)
    for pregunta in faq.keys()
}

# Guardar embeddings en JSON
with open("./api/faq_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(faq_embeddings, f)

print("Embeddings generados y guardados correctamente.")
