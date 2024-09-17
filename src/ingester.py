from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

BASE_CHROMA_DB = "./chroma_db"
MODEL = os.getenv("MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
ollama_embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=MODEL)

def get_chunks_for_country(country):
    country_data_dir = f"./data/{country}"
    loader = PyPDFDirectoryLoader(country_data_dir)
    docs = loader.load()
    print(f"Docs for {country}:", docs)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = splitter.split_documents(docs)
    print(f"Num of Chunks for {country}: ", len(chunks))
    return chunks

def vectorize_for_country(country):
    chunks = get_chunks_for_country(country)
    
    print(f"Storing chunks for {country}")

    country_chroma_db = f"{BASE_CHROMA_DB}_{country.lower()}"
    store = Chroma.from_documents(
        documents=chunks, 
        embedding=ollama_embeddings,
        persist_directory=country_chroma_db)

    store.persist()
    print(f"Data for {country} vectorized in Chroma")

def get_store(country):
    country_chroma_db = f"{BASE_CHROMA_DB}_{country.lower()}"
    store = Chroma(
        persist_directory=country_chroma_db,
        embedding_function=ollama_embeddings)
    
    return store

def main():
    parser = argparse.ArgumentParser(description="Vectorize country-specific data")
    parser.add_argument("--country", help="Specify a single country to vectorize")
    parser.add_argument("--all", action="store_true", help="Vectorize all countries")
    args = parser.parse_args()

    print("Using ", OLLAMA_BASE_URL)

    if args.all:
        countries = ["Italy", "France", "India"]  # Add more countries as needed
        for country in countries:
            print(f'Vectorizing data for {country}')
            vectorize_for_country(country)
    elif args.country:
        print(f'Vectorizing data for {args.country}')
        vectorize_for_country(args.country)
    else:
        print("Please specify either --country [COUNTRY_NAME] or --all")

if __name__ == "__main__":
    main()