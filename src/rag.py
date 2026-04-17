# ABOUTME: Implements the RAG system to retrieve relevant credit policy rules.
# ABOUTME: Uses MarkdownHeaderTextSplitter and FAISS with Sentence Transformers.

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Path configuration
POLICY_PATH = os.path.join("data", "credit_policy.md")
DB_PATH = os.path.join("data", "faiss_index")

class PolicyRetriever:
    def __init__(self, policy_path: str = POLICY_PATH):
        self.policy_path = policy_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = self._build_or_load_index()

    def _build_or_load_index(self):
        """Build the FAISS index from markdown or load if exists."""
        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Policy file not found at {self.policy_path}")

        # Split by markdown headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        with open(self.policy_path, "r") as f:
            content = f.read()
        
        splits = markdown_splitter.split_text(content)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        return vectorstore

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """Retrieve the top k relevant policy snippets."""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def get_policy_context(self, client_data: dict) -> str:
        """
        Custom retrieval logic based on client attributes.
        Returns a formatted context string.
        """
        # Formulate a detailed query based on client profile
        query = f"Rules for loan with {client_data.get('credit_amount')} DM, " \
                f"duration {client_data.get('duration')} months, " \
                f"age {client_data.get('age')}, " \
                f"purpose {client_data.get('purpose')}, " \
                f"savings {client_data.get('saving_accounts')}."
        
        relevant_rules = self.retrieve(query)
        context = "\n---\n".join(relevant_rules)
        return context

# Global instance for easy access
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever
