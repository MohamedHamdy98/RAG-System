from pipeline.pipeline_rag import pipeline_rag
from pipeline.pipeline_retrieve_data import retrieve_relevant_knowledge

if __name__== "__main__" :
    pipeline_rag()
    print(retrieve_relevant_knowledge(query="what can the vminds do?"))