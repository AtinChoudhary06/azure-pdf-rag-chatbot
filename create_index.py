import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)

load_dotenv()

# ---- Config ----
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")   # e.g. https://your-search-name.search.windows.net
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "rag-docs-index"
VECTOR_DIM = 1536  # ada-002

# ---- Define fields ----
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=VECTOR_DIM,
        vector_search_profile_name="my-vector-profile",
    ),
    SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
]

# ---- Vector search config (HNSW algorithm) ----
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="my-hnsw-config")],
    profiles=[
        VectorSearchProfile(
            name="my-vector-profile",
            algorithm_configuration_name="my-hnsw-config",
        )
    ],
)

# ---- Create index ----
index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))
result = client.create_or_update_index(index)

print(f"Index '{result.name}' created successfully with {len(result.fields)} fields.")