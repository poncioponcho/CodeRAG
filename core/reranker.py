import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class ONNXReranker:
    def __init__(self, model_path: str = "models/crossencoder-fp32/model.onnx"):
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = None
    
    def _tokenize_pairs(self, query: str, documents: List[str]):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("models/crossencoder-fp32")
        
        features = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        
        return {
            "input_ids": features["input_ids"].astype(np.int64),
            "attention_mask": features["attention_mask"].astype(np.int64),
            "token_type_ids": features.get("token_type_ids", np.zeros_like(features["input_ids"])).astype(np.int64)
        }
    
    def predict(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []
        
        query = pairs[0][0]
        documents = [doc for _, doc in pairs]
        
        inputs = self._tokenize_pairs(query, documents)
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        
        if logits.shape[1] == 1:
            scores = self._sigmoid(logits[:, 0])
        else:
            scores = self._sigmoid(logits[:, 1])
        return scores.tolist()
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        if not documents:
            return []
        
        pairs = [(query, doc) for doc in documents]
        scores = self.predict(pairs)
        
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
