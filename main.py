import os
import time
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from openai import AzureOpenAI, RateLimitError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot API")

# ---- Azure Blob Storage ----
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "pdf-documents")
blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# ---- Document Intelligence ----
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_KEY = os.getenv("DOC_INTEL_KEY")
doc_intel_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_KEY)
)

# ---- Embedding model (ada-002) ----
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")
EMBED_DEPLOYMENT = "text-embedding-ada-002"
embed_client = AzureOpenAI(
    api_key=EMBEDDING_KEY,
    api_version="2024-02-01",
    azure_endpoint=EMBEDDING_ENDPOINT
)

# ---- Chat model (gpt-oss-120b) ----
PHI4_ENDPOINT = os.getenv("AZURE_PHI4_ENDPOINT")  # reused var name, now points to gpt-oss-120b resource
PHI4_KEY = os.getenv("AZURE_PHI4_KEY")
CHAT_DEPLOYMENT = "gpt-oss-120b"
chat_client = AzureOpenAI(
    api_key=PHI4_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=PHI4_ENDPOINT
)

# ---- Azure AI Search ----
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "rag-docs-index"
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)


# =====================================================
# Helper functions
# =====================================================

def upload_to_blob(file_path, filename):
    blob_client = container_client.get_blob_client(filename)
    with open(file_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    return blob_client.url


def extract_text_with_doc_intel(file_path):
    with open(file_path, "rb") as f:
        poller = doc_intel_client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=f.read())
        )
    result = poller.result()
    return result.content


def chunk_text(text, chunk_size=300, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def get_embedding(text):
    response = embed_client.embeddings.create(input=text, model=EMBED_DEPLOYMENT)
    return response.data[0].embedding


def get_next_doc_id():
    """Get a starting id offset based on current doc count in the index."""
    try:
        results = search_client.search(search_text="*", select=["id"], top=1, order_by=["id desc"])
        ids = [int(r["id"]) for r in results]
        return max(ids) + 1 if ids else 0
    except Exception:
        return int(time.time())  # fallback unique-ish id


def search_chunks(question, top_k=1):
    vector = get_embedding(question)
    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["content", "source"]
    )
    return [(r["content"], r["source"]) for r in results]


def ask_question(question):
    chunks = search_chunks(question)
    if not chunks:
        return "No relevant information found in the documents."

    context = "\n\n".join([f"[Source: {src}]\n{content}" for content, src in chunks])
    prompt = f"""Answer the question based only on the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    max_retries = 5
    wait_seconds = 15
    for attempt in range(1, max_retries + 1):
        try:
            completion = chat_client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            return completion.choices[0].message.content
        except RateLimitError:
            if attempt == max_retries:
                return "Sorry, the model is rate-limited and didn't respond after several retries."
            time.sleep(wait_seconds)


# =====================================================
# API Models
# =====================================================

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


# =====================================================
# Endpoints
# =====================================================

@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Upload to Blob Storage
        blob_url = upload_to_blob(tmp_path, file.filename)

        # 2. Extract text via Document Intelligence
        text = extract_text_with_doc_intel(tmp_path)

        # 3. Chunk
        chunks = chunk_text(text)

        # 4. Embed + upload to index
        doc_id = get_next_doc_id()
        documents_to_upload = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            vector = get_embedding(chunk)
            documents_to_upload.append({
                "id": str(doc_id),
                "content": chunk,
                "content_vector": vector,
                "source": file.filename
            })
            doc_id += 1

        if documents_to_upload:
            search_client.upload_documents(documents=documents_to_upload)

        return {
            "filename": file.filename,
            "blob_url": blob_url,
            "chunks_indexed": len(documents_to_upload)
        }
    finally:
        os.remove(tmp_path)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    answer = ask_question(request.question)
    return AskResponse(answer=answer)