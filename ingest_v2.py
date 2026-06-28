import os
import glob
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

load_dotenv()

# ---- CONFIG: change this one line ----
LOCAL_DOCS_FOLDER = r"C:\Users\admin\OneDrive\Desktop\azure_rag"  # folder with PDFs to upload

# ---- Azure Blob Storage ----
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER", "pdf-documents")

blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# ---- Azure Document Intelligence ----
DOC_INTEL_ENDPOINT = os.getenv("DOC_INTEL_ENDPOINT")
DOC_INTEL_KEY = os.getenv("DOC_INTEL_KEY")

doc_intel_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_KEY)
)

# ---- Azure OpenAI (embeddings) ----
AOAI_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT")
AOAI_KEY = os.getenv("EMBEDDING_KEY")
EMBED_DEPLOYMENT = "text-embedding-ada-002"

embed_client = AzureOpenAI(
    api_key=AOAI_KEY,
    api_version="2024-02-01",
    azure_endpoint=AOAI_ENDPOINT
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

# ---- Step 1: Upload PDF to Blob Storage ----
def upload_to_blob(pdf_path):
    filename = os.path.basename(pdf_path)
    blob_client = container_client.get_blob_client(filename)
    with open(pdf_path, "rb") as f:
        blob_client.upload_blob(f, overwrite=True)
    print(f"  Uploaded to Blob Storage: {filename}")
    return blob_client.url

# ---- Step 2: Extract text using Document Intelligence ----
def extract_text_with_doc_intel(pdf_path):
    with open(pdf_path, "rb") as f:
        poller = doc_intel_client.begin_analyze_document(
            "prebuilt-read",  # general text extraction model
            AnalyzeDocumentRequest(bytes_source=f.read())
        )
    result = poller.result()
    return result.content  # full extracted text

# ---- Step 3: Chunk text ----
def chunk_text(text, chunk_size=300, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ---- Step 4: Embed a chunk ----
def get_embedding(text):
    response = embed_client.embeddings.create(
        input=text,
        model=EMBED_DEPLOYMENT
    )
    return response.data[0].embedding

# ---- Step 5: Process all PDFs ----
def main():
    pdf_files = glob.glob(os.path.join(LOCAL_DOCS_FOLDER, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) in {LOCAL_DOCS_FOLDER}")

    documents_to_upload = []
    doc_id = 0

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}")

        # Upload original PDF to Blob Storage
        blob_url = upload_to_blob(pdf_path)

        # Extract text via Document Intelligence
        print("  Extracting text via Document Intelligence...")
        text = extract_text_with_doc_intel(pdf_path)

        # Chunk
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")

        for chunk in chunks:
            if not chunk.strip():
                continue
            vector = get_embedding(chunk)
            documents_to_upload.append({
                "id": str(doc_id),
                "content": chunk,
                "content_vector": vector,
                "source": filename
            })
            doc_id += 1

    print(f"Uploading {len(documents_to_upload)} chunks to index...")
    if documents_to_upload:
        result = search_client.upload_documents(documents=documents_to_upload)
        print(f"Upload complete. {len(result)} documents indexed.")
    else:
        print("No documents to upload.")

if __name__ == "__main__":
    main()