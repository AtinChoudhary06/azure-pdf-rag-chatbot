import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI, RateLimitError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

# ---- Chat model (Phi-4-mini-instruct, plain API key auth) ----
PHI4_ENDPOINT = os.getenv("AZURE_PHI4_ENDPOINT")
PHI4_KEY = os.getenv("AZURE_PHI4_KEY")
CHAT_DEPLOYMENT = "gpt-oss-120b"

chat_client = AzureOpenAI(
    api_key=PHI4_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=PHI4_ENDPOINT
)

# ---- Embedding model (ada-002) ----
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

# ---- Step 1: Embed the question ----
def get_embedding(text):
    response = embed_client.embeddings.create(
        input=text,
        model=EMBED_DEPLOYMENT
    )
    return response.data[0].embedding

# ---- Step 2: Vector search top-k chunks ----
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

# ---- Step 3: Ask the chat model with retrieved context ----
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
                return "Sorry, the model is rate-limited and didn't respond after several retries. Try again in a minute."
            print(f"Rate limited. Retrying in {wait_seconds}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait_seconds)

# ---- Main loop ----
if __name__ == "__main__":
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        answer = ask_question(question)
        print(f"\nAnswer: {answer}\n")