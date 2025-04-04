import json
import numpy as np
import openai
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== GOOGLE SHEETS (opcional, si mantienes tu registro) ==========
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
google_creds_json = os.getenv("GOOGLE_CREDENTIALS")

if google_creds_json:
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(google_creds_json), scope)
else:
    creds = ServiceAccountCredentials.from_json_keyfile_name("./api/guias-digitales-9c87ddbffba6.json", scope)

client = gspread.authorize(creds)
SHEET_NAME = "Chat Interacciones"
sheet = client.open(SHEET_NAME).sheet1

def guardar_interaccion(user_id, pregunta, respuesta):
    # Ajusta o amplía campos si lo requieres
    timestamp = datetime.datetime.now().isoformat()
    row = [timestamp, user_id, pregunta, respuesta]
    sheet.append_row(row)

# ========== FUNCIONES DE PROCESAMIENTO DE TEXTO Y EMBEDDINGS ==========

def obtener_embedding(texto: str) -> np.ndarray:
    """Genera un embedding con el modelo text-embedding-ada-002."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texto
    )
    return np.array(response["data"][0]["embedding"])

def similaridad_coseno(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcula la similitud coseno normalizada."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Carga de CHUNKS y EMBEDDINGS (provenientes de tus PDFs)
# pdf_chunks.json -> {"chunk_0": "...texto del chunk...", "chunk_1": "...", ...}
# pdf_embeddings.json -> {"chunk_0": [0.0123, ...], "chunk_1": [...], ...}

with open("./api/pdf_chunks.json", "r", encoding="utf-8") as f:
    pdf_chunks = json.load(f)  # dict chunk_id -> texto

with open("./api/pdf_embeddings.json", "r", encoding="utf-8") as f:
    raw_embeddings = json.load(f)  # dict chunk_id -> [float, float, ...]

pdf_embeddings = {
    k: np.array(v) for k, v in raw_embeddings.items()
}

def encontrar_mejor_chunk(pregunta: str) -> str:
    """Devuelve el chunk de texto más relevante para la pregunta."""
    embedding_pregunta = obtener_embedding(pregunta)
    best_chunk_id = None
    best_score = -1

    for chunk_id, chunk_vec in pdf_embeddings.items():
        score = similaridad_coseno(embedding_pregunta, chunk_vec)
        if score > best_score:
            best_score = score
            best_chunk_id = chunk_id
    
    # Podrías aplicar un umbral, p. ej. si best_score < 0.75, devuelves None
    return pdf_chunks[best_chunk_id] if best_chunk_id else ""

# ========== LÓGICA DEL CHAT ==========

# Historial de conversación por sesión
user_sessions = {}

def enriquece_html(texto: str) -> str:
    """Convierte saltos dobles en párrafos HTML, etc. (opcional)."""
    partes = texto.split("\n\n")
    return "".join([f"<p>{parte.strip()}</p><br>" for parte in partes])

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    pregunta_usuario = data.get("message", "")
    user_id = request.client.host

    # 1. Recuperación del chunk relevante
    contexto_relevante = encontrar_mejor_chunk(pregunta_usuario)

    # 2. Construimos el prompt del sistema y usuario
    # En "content" del system prompt, pones instrucciones fijas, etc.
    system_prompt = {
        "role": "system",
        "content": """
    Eres un asistente que debe responder siempre en un formato HTML amigable, siguiendo estas reglas:

    1. Usa <strong> ... </strong> para destacar palabras importantes (negritas).
    2. Usa <em> ... </em> para poner texto en cursivas.
    3. Usa <p> ... </p> para cada párrafo y <br> para saltos de línea.
    4. Si tienes listados, usa <ul> o <ol> con <li> para cada ítem.
    5. NO respondas todo en un solo bloque gigante, divídelo en múltiples párrafos cortos (máximo 3 o 4 líneas cada uno).
    6. Entre cada párrafo, agrega una línea en blanco adicional para espaciar.
    7. Si necesitas más de 4 párrafos, hazlos en secciones o bloques cortos.

    Ejemplo:
    <p>Primer párrafo con introducción.</p><br>

    <p>Segundo párrafo con <strong>negritas</strong> y tal vez <em>cursivas</em>.</p><br>

    <ul>
    <li>Elemento 1</li>
    <li>Elemento 2</li>
    </ul>

    <p>Último párrafo de cierre.</p><br><br>

    Si tu respuesta es demasiado larga, crea varios párrafos cumpliendo estas reglas.

     <!-- MEMBRESÍAS -->
  <Membresias>
    <membresia>Plan Básico (Gratis)</strong>: Ideal para comenzar. Incluye una landing básica con galería de 5 fotos y visibilidad inicial en la plataforma. Sin costo y sin alcance garantizado.</membresia>
    <membresia>Membresía SMART (1 propiedad - $28,000 MXN + IVA / anual)</strong>: Ofrece una landing optimizada, campañas de display nativo, estrategia de linkbuilding en el ecosistema de México Desconocido®, posiciones exclusivas en el destino y generación de contenido social. Garantiza un alcance de <strong>1.5 millones</strong> de impactos.</membresia>
    <membresia>Membresía SMART 3 (hasta 3 propiedades - $70,500 MXN + IVA / anual)</strong>: Los mismos beneficios aplicados hasta 3 propiedades. Alcance garantizado de <strong>4.8 millones</strong>.</membresia>
    <membresia>Membresía SMART 5 (hasta 6 propiedades - $118,800 MXN + IVA / anual)</strong>: Máxima visibilidad y presencia para negocios con varias sedes o servicios. Incluye todos los beneficios anteriores con un alcance garantizado de <strong>5.4 millones</strong>.</membresia>
  </Membresias>
    """
    }

    # Añadimos el contexto PDF en un rol "user" (o "system", según prefieras).
    # Esto es opcional, pero a menudo se hace un "system" con un meta-prompt, y un "user" con la info.
    context_chunk_message = {
        "role": "system",
        "content": f"Contexto PDF:\n\n{contexto_relevante}"
    }

    # Mensaje del usuario real
    user_message = {
        "role": "user",
        "content": pregunta_usuario
    }

    # 3. Usamos un historial de conversación por user_id
    if user_id not in user_sessions:
        user_sessions[user_id] = []

    # Agregamos los nuevos mensajes
    conversation = user_sessions[user_id]
    # Añade (o reemplaza) tu system prompt
    conversation.append(system_prompt)
    conversation.append(context_chunk_message)
    conversation.append(user_message)

    # Llamada a ChatCompletion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.3
    )

    # Extraemos la respuesta
    respuesta_gpt = enriquece_html(response.choices[0].message["content"])
    
    # Añadimos la respuesta al historial
    conversation.append({"role": "assistant", "content": respuesta_gpt})
    
    # Opcional: guardar la interacción en Google Sheets
    guardar_interaccion(user_id, pregunta_usuario, respuesta_gpt)

    return {
        "response": respuesta_gpt
    }
