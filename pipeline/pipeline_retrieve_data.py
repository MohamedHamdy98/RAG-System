from langchain_chroma import Chroma
import os
from .pipeline_rag import embedding_data

data_base = "./db/"
os.makedirs(data_base, exist_ok=True)
chroma_db = os.path.join( data_base, "./chroma_db" )

# Pathes
path = "./files"
data = os.listdir(path=path)

embedding_llm = embedding_data()

def loading_db(chroma_path, embedding_llm):
    loaded_vector_db = Chroma(persist_directory=chroma_path, embedding_function=embedding_llm)
    return loaded_vector_db

def retrieve_relevant_knowledge(query, chroma_path=chroma_db, embedding_llm=embedding_llm, k=4):
    db = loading_db(chroma_path, embedding_llm)
    results = db.similarity_search(query, k=k)
    retrieved_texts = [doc.page_content for doc in results]
    return "\n\n".join(retrieved_texts)


if __name__ == "__main__":
    print("Retrieve the Data...")
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        knowledge = retrieve_relevant_knowledge(query=question)
        print("🔍 Retrieved Knowledge:")
        print(knowledge)