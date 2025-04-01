from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
import datetime

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io




# Autenticación con Google Sheets
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive" 
]
# Intenta cargar desde variable de entorno (Railway)
google_creds_json = os.getenv("GOOGLE_CREDENTIALS")

if google_creds_json:
    # Si existe la variable en Railway, úsala
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(google_creds_json), scope)
else:
    # Si estás en local, usa archivo local
    creds = ServiceAccountCredentials.from_json_keyfile_name("./api/guias-digitales-9c87ddbffba6.json", scope)

client = gspread.authorize(creds)

# Abre la hoja
SHEET_NAME = "Chat Interacciones"
sheet = client.open(SHEET_NAME).sheet1

def guardar_interaccion(user_id, pregunta, respuesta, origen="gpt",tipo_negocio="desconocido",intencion="desconocido",nivel_conocimiento="desconocido"):
    timestamp = datetime.datetime.now().isoformat()
    row = [
        timestamp,
        user_id,
        pregunta,
        respuesta,
        origen,
        tipo_negocio,
        intencion,
        nivel_conocimiento
    ]
    sheet.append_row(row)


def analizar_usuario(mensaje):
    prompt = f"""
Eres un analizador de perfil de usuario. Dado el siguiente mensaje, devuelve una estructura JSON con:

- tipo_negocio: (hotel, restaurante, guía, otro)
- intencion: (registrarse, aumentar visibilidad, solo informarse, otro)
- nivel_conocimiento: (nuevo, ya conoce Escapadas.mx, registrado)

Mensaje del usuario:
"{mensaje}"

Responde solo el JSON, sin explicación.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        perfil = json.loads(response.choices[0].message["content"])
    except Exception:
        perfil = {
            "tipo_negocio": "desconocido",
            "intencion": "desconocido",
            "nivel_conocimiento": "desconocido"
        }
    return perfil


def parafrasear_respuesta(texto, estilo="más empático y conversacional"):
    prompt = (
        f"Reformula este contenido en un tono {estilo}, manteniendo la información y formato en HTML amigable, "
        f"con párrafos <p>, saltos de línea <br> y palabras clave en <strong>:\n\n{texto}"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message["content"]


app = FastAPI()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def enriquece_html(texto):
    partes = texto.split("\n\n")  # Suponiendo que hay saltos dobles
    return "".join([f"<p>{parte.strip()}</p><br>" for parte in partes])

def guardar_interaccio_old(user_id, pregunta, respuesta, origen="gpt"):
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "pregunta": pregunta,
        "respuesta": respuesta,
        "origen": origen  # puede ser "faq" o "gpt"
    }

    ruta_log = "conversaciones.json"
    
    # Si ya existe el archivo, carga el contenido
    if os.path.exists(ruta_log):
        with open(ruta_log, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(log_entry)

    with open(ruta_log, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Cargar embeddings y datos del FAQ
with open("./api/faq_embeddings.json", "r", encoding="utf-8") as f:
    raw_embeddings = json.load(f)

faq_embeddings = {
    pregunta: np.array(embedding)
    for pregunta, embedding in raw_embeddings.items()
}

with open("./api/faq_data.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

# Historial de conversación por sesión (temporal)
user_sessions = {}

# Embedding de la pregunta
def obtener_embedding(texto):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texto
    )
    return np.array(response["data"][0]["embedding"])

# Buscar pregunta similar
def encontrar_pregunta_mas_similar(pregunta_usuario):
    embedding_usuario = obtener_embedding(pregunta_usuario)
    similitudes = {
        pregunta: np.dot(embedding_usuario, embedding) / (
            np.linalg.norm(embedding_usuario) * np.linalg.norm(embedding)
        )
        for pregunta, embedding in faq_embeddings.items()
    }
    pregunta_mas_similar = max(similitudes, key=similitudes.get)
    mayor_similitud = similitudes[pregunta_mas_similar]
    if mayor_similitud > 0.85:
        return pregunta_mas_similar
    return None

# Endpoint principal
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    pregunta_usuario = data.get("message", "")
    user_id = request.client.host

    # 1. Buscar coincidencia en el FAQ
    pregunta_similar = encontrar_pregunta_mas_similar(pregunta_usuario)
    if pregunta_similar:
        respuesta_original = faq[pregunta_similar]["respuesta"]
        respuesta_parafraseada = parafrasear_respuesta(respuesta_original)

        perfil_usuario = analizar_usuario(pregunta_usuario)

        guardar_interaccion(user_id, pregunta_usuario, respuesta_parafraseada, origen="faq")
        return {"response": respuesta_parafraseada, "sticker": faq[pregunta_similar]["sticker"]}
        #guardar_interaccion(user_id, pregunta_usuario, respuesta["respuesta"], origen="faq",tipo_negocio=perfil_usuario["tipo_negocio"],intencion=perfil_usuario["intencion"],nivel_conocimiento=perfil_usuario["nivel_conocimiento"])
        #return {"response": respuesta["respuesta"], "sticker": respuesta["sticker"]}

    # 2. Si no hay coincidencia, usar memoria y GPT
    if user_id not in user_sessions:
        user_sessions[user_id] = [
            {"role": "system", "content": '''
            <AgentInstructions>
  <Role>
    <name>Eres un asistente</name>
    <description>Estoy aquí para ayudarte a explorar oportunidades que impulsen tu negocio turístico y te permitan conectar con viajeros que buscan vivir momentos inolvidables.
    Solo puedes proporcionar información de tu base de conocimientos. No eres un chatgpt gratuito o similar. No respondas cosas que no van de acuerdo a tu contexto.
    </description>
    
  </Role>

  <Goal>
    <Primary>Informar a los prestadores de servicios turísticos sobre las soluciones que ofrece Escapadas.mx para aumentar la visibilidad y rentabilidad de sus negocios, motivándolos a unirse a la plataforma. Solo respondes cosas que tengan que ver con tu base de conocimiento, no eres una herramienta que puedan usar los usuarios para inteligencia artificial. En formato html.</Primary>
    <Secondary>    Solo proporcionas información sobre prestadores de servicios como: hoteles, restaurantes, tours, entretenimientos. Cualquier otro servicio invita a los usuarios a el medio de contacto de escapadas.mx</Secondary>
  </Goal>

  <Instructions>
    <Instruction>Paso 0: Solo debes proporcionar información de tu base de conocimiento y no permitir que los usuarios te usen como un modelo de lenguaje gratuito.</Instruction>
    <Instruction>Paso 1: Saluda cordialmente y preséntate mencionando que tu objetivo es resolver dudas, mostrando empatía y entusiasmo por ayudar.</Instruction>
      <Instruction>Paso 2: Explica cómo Escapadas.mx puede beneficiar a su negocio, destacando los siguientes puntos:
      - Visibilidad en el ecosistema digital de México Desconocido®: Conexión con una comunidad activa de viajeros, ampliando el alcance y posicionamiento del negocio en plataformas clave dentro del turismo mexicano.
      - Credibilidad y confianza: Transmisión de la esencia del negocio resaltando lo que lo diferencia, asegurando que la oferta resuene de manera efectiva con viajeros que valoran la calidad y lo genuino.
      - Estrategias para mitigar la dependencia estacional: Creación de campañas estratégicas que atraen viajeros durante todas las temporadas, asegurando un flujo constante incluso fuera de las fechas más concurridas.
      - Audiencia segmentada: Llegar a quienes realmente valoran lo que se ofrece, con segmentación precisa y estrategias avanzadas que maximizan las oportunidades de conversión.
      - Mayor rentabilidad sin costos de intermediación por reservas: Dirección de los viajeros directamente al canal de reservas del negocio, sin cobrar comisiones ni utilizar intermediarios, garantizando el control total de los ingresos y optimizando la rentabilidad.
    </Instruction>
    <Instruction>Paso 3: Proporciona información sobre cómo registrarse en la plataforma y los planes disponibles, resaltando que hay opciones sin costo inicial.</Instruction>
    <Instruction>Paso 4: Responde cualquier pregunta adicional que el usuario pueda tener y ofrece asistencia para el proceso de registro. El formato de tus respuestas debe contener  parrafos <p> <br> y palabras importantes en formato "strong" para facilitar la lectura.</Instruction>
    <Instruction>Medios de contacto: Visita esta página web y proporciona los datos solicitados: https://negocios.escapadas.mx/login?tab=signup Si necesitas asistencia personalizada comunicate al correo:alex.contacto@escapadas.mx o al teléfono:+52 56 4085 8541</Instruction>
  </Instructions>

  <Membresias>
   Explica que existen distintos planes de registro y membresía para sumarse a Escapadas.mx, dependiendo del nivel de visibilidad que el negocio desee alcanzar:

  - <p><strong>Plan Básico (Gratis)</strong>: Ideal para comenzar. Incluye una landing básica con galería de 5 fotos y visibilidad inicial en la plataforma. Sin costo y sin alcance garantizado.</p>

- <p><strong>Membresía SMART (1 propiedad - $28,000 MXN + IVA / anual)</strong>: Ofrece una landing optimizada, campañas de display nativo, estrategia de linkbuilding en el ecosistema de México Desconocido®, posiciones exclusivas en el destino y generación de contenido social. Garantiza un alcance de <strong>1.5 millones</strong> de impactos.</p>

- <p><strong>Membresía SMART 3 (hasta 3 propiedades - $70,500 MXN + IVA / anual)</strong>: Los mismos beneficios aplicados hasta 3 propiedades. Alcance garantizado de <strong>4.8 millones</strong>.</p>

- <p><strong>Membresía SMART 5 (hasta 6 propiedades - $118,800 MXN + IVA / anual)</strong>: Máxima visibilidad y presencia para negocios con varias sedes o servicios. Incluye todos los beneficios anteriores con un alcance garantizado de <strong>5.4 millones</strong>.</p>

Asegura al usuario que todos los planes fueron diseñados para adaptarse a distintos niveles de crecimiento y que incluso pueden iniciar gratis, y escalar conforme vean resultados.
</Membresias>

<LandingSmart>
  Si el usuario pregunta sobre la Landing SMART, explícale que es una herramienta completa que ayuda a que los viajeros no solo vean su negocio, sino que lo elijan. Detalla que combina:

- <p><strong>SEO</b>: Para aparecer en los resultados cuando las personas buscan lo que el negocio ofrece.</p>
- <p><strong>Marketing</b>: Contenidos optimizados y segmentados que atraen a los viajeros adecuados.</p>
- <p><strong>Amplificación</b>: Campañas en redes sociales con alcance garantizado.</p>
- <p><strong>Relevancia</b>: Contar la esencia auténtica del negocio, generando conexión con los viajeros.</p>
- <p><strong>Tracking</strong>: Medición continua de resultados para mejorar cada acción.</p>

Asegúrate de transmitir que esta estrategia integral maximiza la visibilidad y relevancia del negocio, conectándolo con una audiencia segmentada que realmente está interesada en lo que ofrece.
</LandingSmart>

</FormatoRespuesta>
  El formato de tus respuestas debe contener parrafos <p> <br> y palabras importantes en formato "strong" para facilitar la lectura.
  <p>Parte de la respuesta </p>
  <p>Segunda parte de la respuesta</p>
  <p>Tercer parte de la respuesta</p>
  <p>.....</p> 
</FormatoRespuesta>


  <ResponseTemplate>
      "¡Hola!. Quisiera contarte cómo <strong>Escapadas.mx</strong> puede ayudar a que más viajeros descubran y se enamoren de tu negocio. ¿Podrías contarme un poco sobre tu empresa para ofrecerte información más detallada?",
  </ResponseTemplate>

  <ResponseTemplate>
    Si no entiendes bien la pregunta, ofrece estas opciones:
    <ul>
    <li>¿Te gustaría conocer los planes disponibles?</li>
    <li>¿Quieres saber cómo mejorar la visibilidad de tu negocio?</li>
    <li>¿Quieres que te ayude con el proceso de registro?</li>
    </ul>
  </ResponseTemplate>

  <Examples>
    <Example>
      <UserInput>
        Tengo un pequeño hotel en un pueblo mágico y quiero atraer más huéspedes.
      </UserInput>
      <AgentOutput>
        ¡Qué maravilla tener un hotel en un pueblo mágico! <strong>Escapadas.mx</strong> puede ayudarte a aumentar la visibilidad de tu hotel conectándote con una comunidad activa de viajeros que buscan experiencias auténticas. <br>Además, al destacar lo que hace único a tu hotel, podemos transmitir esa esencia que atraerá a más huéspedes. ¿Te gustaría saber más sobre nuestros planes y cómo registrarte?
      </AgentOutput>
    </Example>

    <Example>
      <UserInput>
        ¿Cuáles son los costos de anunciarme en su plataforma?
      </UserInput>
      <AgentOutput>
        ¡Excelente pregunta! En Escapadas.mx ofrecemos diferentes planes adaptados a las necesidades de cada negocio, incluyendo opciones sin costo inicial. Nuestro objetivo es que puedas aumentar tu rentabilidad sin preocuparte por costos de intermediación. ¿Te gustaría que te detalle los planes disponibles y sus beneficios?
      </AgentOutput>
    </Example>
  </Examples>
</AgentInstructions>
            '''}
        ]

    user_sessions[user_id].append({"role": "user", "content": pregunta_usuario})

    if len(user_sessions[user_id]) > 10:
        user_sessions[user_id] = user_sessions[user_id][-10:]

    # Refuerza el formato justo antes de enviar el prompt
    user_sessions[user_id].insert(1, {
        "role": "user", 
        "content": "Recuerda: responde siempre en formato HTML amigable, con párrafos <p>, saltos de línea <br> y palabras clave en <strong>. Divide la respuesta en bloques cortos para que sea fácil de leer."
    })
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=user_sessions[user_id]
    )

    respuesta_gpt = enriquece_html(response.choices[0].message["content"])
    user_sessions[user_id].append({"role": "assistant", "content": respuesta_gpt})

    perfil_usuario = analizar_usuario(pregunta_usuario)
    guardar_interaccion(user_id, pregunta_usuario, respuesta_gpt, origen="gpt",tipo_negocio=perfil_usuario["tipo_negocio"],intencion=perfil_usuario["intencion"],nivel_conocimiento=perfil_usuario["nivel_conocimiento"])
    return {
        "response": respuesta_gpt,
        "sticker": ""
    }
