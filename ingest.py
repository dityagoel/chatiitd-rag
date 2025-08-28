# ingest.py
import os
from pathlib import Path
from typing import List
import json

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings


load_dotenv()  # take environment variables from .env file

DATA_DIR = Path("sources/test")  # folder with .pdf and .json files
COLLECTION_NAME = "documents"

def load_json_as_documents(path: Path) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    # Generic: flatten strings inside JSON values. Adjust if you have typed JSON schema.
    text = []
    def extract(obj):
        if isinstance(obj, str):
            text.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)
    extract(payload)
    content = "\n".join(text)
    metadata = {"source": str(path)}
    return [Document(page_content=content, metadata=metadata)]

def load_pdf_as_documents(path: Path) -> List[Document]:
    # Use PyPDF loader via simple approach: use pypdf to extract pages
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        metadata = {"source": str(path), "page": i+1}
        docs.append(Document(page_content=txt, metadata=metadata))
    return docs

def main():
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not google_key:
        raise RuntimeError("Set GOOGLE_API_KEY environment variable")

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", None)

    # embeddings = GoogleGenerativeAIEmbeddings(
    #     google_api_key=google_key,
    #     model="gemini-embedding-001"
    # )

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

    # collect all documents
    documents = []
    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() == ".json":
            documents += load_json_as_documents(p)
        elif p.suffix.lower() in (".pdf",):
            documents += load_pdf_as_documents(p)

    # split large documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    split_docs = []
    for d in documents:
        parts = splitter.split_documents([d])
        for part in parts:
            split_docs.append(part)

    # create Qdrant collection and upload
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)  # will fail if no Qdrant running
    qdrant = Qdrant.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=qdrant_url,
        force_recreate=True,
    )

    print(f"Indexed {len(split_docs)} chunks into Qdrant collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    main()
