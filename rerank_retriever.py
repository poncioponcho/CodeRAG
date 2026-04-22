from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class RerankRetriever:
    def __init__(self, base_retriever, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", k=3):
        self.base = base_retriever
        self.reranker = CrossEncoder(model_name)
        self.k = k
    
    def invoke(self, query: str):
        # 1. 粗排：检索更多候选（10个）
        candidates = self.base.invoke(query)
        
        # 2. 精排：Cross-Encoder 打分
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # 3. 取 top-k
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.k]]