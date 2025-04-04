import openai
import PyPDF2
import json
import tiktoken
import os
from dotenv import load_dotenv


load_dotenv()

# -----------------------
# CONFIGURACIONES
# -----------------------
openai.api_key = os.getenv("OPENAI_API_KEY")  # O asigna tu API key aquí directamente
PDF_FILE = "./api/negocios_data.pdf"                 # Ruta a tu archivo PDF
CHUNK_SIZE = 800                              # Tamaño aproximado en tokens para cada fragmento
CHUNK_OVERLAP = 150                            # Superposición de tokens entre chunks para mayor coherencia
OUTPUT_CHUNKS = "./api/pdf_chunks.json"
OUTPUT_EMBEDDINGS = "./api/pdf_embeddings.json"
EMBEDDING_MODEL = "text-embedding-ada-002"

# -----------------------
# FUNCIONES
# -----------------------
def extract_pdf_text(pdf_file: str) -> str:
    """
    Lee el contenido de cada página de un PDF y retorna todo como una sola cadena de texto.
    """
    all_text = []
    with open(pdf_file, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n".join(all_text)

def tokenize_text(text: str, tokenizer) -> list:
    """
    Convierte texto en IDs de tokens usando tiktoken. Retorna la lista de IDs.
    """
    return tokenizer.encode(text)

def detokenize_tokens(tokens: list, tokenizer) -> str:
    """
    Convierte una lista de tokens en cadena de texto.
    """
    return tokenizer.decode(tokens)

def chunk_text(text: str, chunk_size: int, chunk_overlap: int, tokenizer) -> list:
    """
    Divide el texto en fragmentos de aproximadamente chunk_size tokens,
    con una superposición de chunk_overlap para mantener coherencia.
    """
    tokens = tokenize_text(text, tokenizer)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_slice = tokens[start:end]
        chunk_text = detokenize_tokens(chunk_slice, tokenizer)
        chunks.append(chunk_text)
        
        # Avanzamos chunk_size - chunk_overlap para mantener superposición
        start += (chunk_size - chunk_overlap)
    
    return chunks

def generate_embedding(text: str) -> list:
    """
    Llama a la API de OpenAI para obtener el embedding del texto,
    retorna la lista de floats.
    """
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    vector = response["data"][0]["embedding"]
    return vector

# -----------------------
# SCRIPT PRINCIPAL
# -----------------------
def main():
    # 1. Extraer todo el texto del PDF
    print(f"Extrayendo texto de {PDF_FILE} ...")
    full_text = extract_pdf_text(PDF_FILE)

    # 2. Configurar tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Ajustar si usas GPT-3.5/4, etc.

    # 3. Dividir el texto en chunks
    print("Dividiendo el texto en fragmentos (chunks)...")
    chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP, tokenizer)

    # 4. Generar embeddings para cada chunk y guardar en JSON
    pdf_chunks_dict = {}
    pdf_embeddings_dict = {}

    print("Generando embeddings de cada chunk...")
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        pdf_chunks_dict[chunk_id] = chunk
        emb = generate_embedding(chunk)
        pdf_embeddings_dict[chunk_id] = emb

    # 5. Guardar los resultados
    print(f"Guardando {OUTPUT_CHUNKS} y {OUTPUT_EMBEDDINGS} ...")
    with open(OUTPUT_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(pdf_chunks_dict, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_EMBEDDINGS, "w", encoding="utf-8") as f:
        json.dump(pdf_embeddings_dict, f, indent=2, ensure_ascii=False)

    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    main()
