import numpy as np
import onnxruntime as ort

class ONNXEmbedder:
    def __init__(self, model_path: str = "models/bge-small-zh-onnx/model.onnx"):
        providers = [
            ('CoreMLExecutionProvider', {}),
            'CPUExecutionProvider'
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.tokenizer = None
        
    def _tokenize(self, text: str):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("models/bge-small-zh-onnx")
        
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64)
        }
    
    def embed_query(self, text: str) -> np.ndarray:
        inputs = self._tokenize(text)
        outputs = self.session.run(None, inputs)
        embeddings = outputs[0]
        
        attention_mask = inputs["attention_mask"]
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.sum(mask_expanded, axis=1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask
    
    def embed_batch(self, texts: list) -> np.ndarray:
        return np.array([self.embed_query(text)[0] for text in texts])
