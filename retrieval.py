from typing import List, Dict, Any
import langchain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Base classes for modularity
class Retriever:
    def retrieve(self) -> List[str]:
        """
        Retrieve relevant documents for a given query.
        Args:
            query (str): User query.
        Returns:
            List[str]: Retrieved documents.
        """
        raise NotImplementedError("Subclasses must implement this method")


class BM25_Retriever(Retriever):
    def __init__(self, documents: List[str]):
        self.documents = documents  
    
    def retrieve(self):
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        return bm25_retriever 

class ChromaDB_Retriever(Retriever):
    def __init__(self, documents: List[str]):
        self.documents = documents  

    def retrieve(self):
        #model= "sentence-transformers/all-mpnet-base-v2"
        model="sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model)
        vectorstore = Chroma.from_documents(self.documents, embeddings) 
        return vectorstore.as_retriever() 

class Ensemble_Retriever(Retriever):
    def __init__(self, documents: List[str]):
        self.documents = documents  # Corpus of documents
    
    def retrieve(self, query: str) -> List[str]:
        bm25_retriever = BM25_Retriever(self.documents).retrieve()
        chromadb_retriever = ChromaDB_Retriever(self.documents).retrieve()

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chromadb_retriever], weights=[0, 1]
        )

        config = {"configurable": {"search_kwargs_chroma": {"k": 7}}}
        
        relevant_contexts = ensemble_retriever.invoke(query, config=config)
        return relevant_contexts




