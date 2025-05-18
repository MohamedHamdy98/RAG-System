from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

data_base = "./db/"
os.makedirs(data_base, exist_ok=True)
chroma_db = os.path.join( data_base, "./chroma_db" )

# Pathes
path = r"./files"
data = os.listdir(path=path)

# Loading The Data
def loading_data():
    docs = []
    for doc in data:
        if doc.endswith(".pdf"):
            llm_loader =PyPDFLoader( os.path.join( path, doc ) )
            docs.extend(llm_loader.load())
    return docs


# Splitting The Data
def splitting_data(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    docs = text_splitter.split_documents(documents=document)
    return docs

# Embedding The Data
def embedding_data(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedding_llm = HuggingFaceEmbeddings(model_name=model_name)
    return embedding_llm

# Store in Chroma vector database
def store_data(documents, embedding_model):
    db = Chroma.from_documents( documents, embedding_model, persist_directory = chroma_db )
    return db



def pipeline_rag():
    print("Loading the Documents...")
    docs = loading_data()
    print(f"✅ Loaded {len(docs)} documents.")
    print(f"Sample doc length: {len(docs[0].page_content)}")

    print("Splitting the Data...")
    chunks  = splitting_data(docs)
    print(f"✅ Split into {len(chunks)} chunks.")

    print("Embedding the Data...")
    embedding_model = embedding_data()

    print("Storing the Data...")
    store_data(chunks , embedding_model)

    print("Pipeline is finishing!")


if __name__ == "__main__":
    pipeline_rag()
