"""问题类型分类模块：区分抽象问题与具体技术问题。

抽象问题：需要综合理解多个概念，开放式问题，如"如何理解..."、"谈谈你对...的看法"
具体技术问题：涉及具体技术细节、代码实现、公式定义等，如"Attention公式是什么"、"如何实现DDP"

分类策略：
1. 规则匹配：基于关键词模式识别
2. 机器学习：使用预训练分类模型（可选）
"""

import re
import requests
import json


class QuestionClassifier:
    """问题类型分类器"""
    
    # 具体技术问题关键词模式
    CONCRETE_PATTERNS = [
        # 技术术语关键词
        r'\b(公式|定义|原理|算法|实现|代码|函数|方法|步骤|流程)\b',
        r'\b(如何|怎样|怎么)\s+(实现|编写|使用|配置|启动|部署)\b',
        r'\b(是什么|什么是|有哪些|包括)\b',
        r'\b(区别|差异|对比|比较)\b',
        r'\b(时间复杂度|空间复杂度|复杂度)\b',
        r'\b(PyTorch|TensorFlow|CUDA|GPU|CPU)\b',
        r'\b(参数|超参数|阈值|设置)\b',
        r'\b(训练|微调|推理|预测)\b',
        r'\b(优化|加速|并行|分布式)\b',
        r'\b(Transformer|LSTM|CNN|Attention|GPT|LLaMA)\b',
        r'\b(向量|嵌入|Embedding|检索|召回)\b',
        r'\b(RAG|HyDE|CrossEncoder|BM25|FAISS)\b',
    ]
    
    # 抽象问题关键词模式
    ABSTRACT_PATTERNS = [
        r'\b(如何理解|理解|认识)\b',
        r'\b(谈谈你对|说说你对|阐述)\b',
        r'\b(为什么|为何)\s+(重要|关键|必要)\b',
        r'\b(优势|劣势|挑战|机遇)\b',
        r'\b(未来发展|发展趋势|前景)\b',
        r'\b(权衡|取舍|选择|trade-off)\b',
        r'\b(设计思想|核心思想|本质)\b',
        r'\b(哲学|理念|范式)\b',
    ]
    
    def __init__(self, use_llm_classifier: bool = True):
        self.use_llm_classifier = use_llm_classifier
    
    def _rule_based_classify(self, query: str) -> tuple:
        """基于规则的分类方法"""
        query_lower = query.lower()
        
        # 检测具体技术问题模式
        concrete_score = 0
        for pattern in self.CONCRETE_PATTERNS:
            if re.search(pattern, query_lower):
                concrete_score += 1
        
        # 检测抽象问题模式
        abstract_score = 0
        for pattern in self.ABSTRACT_PATTERNS:
            if re.search(pattern, query_lower):
                abstract_score += 1
        
        # 判断类型
        if abstract_score > concrete_score:
            return "abstract", abstract_score, concrete_score
        elif concrete_score > abstract_score:
            return "concrete", abstract_score, concrete_score
        else:
            # 分数相等时，更倾向于具体问题（保守策略）
            return "concrete", abstract_score, concrete_score
    
    def _llm_based_classify(self, query: str) -> str:
        """基于LLM的分类方法"""
        prompt = f"""请判断以下问题属于"抽象问题"还是"具体技术问题"：

抽象问题：需要综合理解多个概念的开放式问题，如"如何理解注意力机制在Transformer中的作用"、"谈谈你对RAG的看法"

具体技术问题：涉及具体技术细节、代码实现、公式定义等，如"Attention公式是什么"、"如何实现DDP"

问题：{query}

请直接输出"抽象问题"或"具体技术问题"，不要解释。"""
        
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_ctx": 2048}
                }
            )
            resp.raise_for_status()
            result = resp.json()["response"].strip()
            
            if "抽象问题" in result:
                return "abstract"
            elif "具体技术问题" in result:
                return "concrete"
            else:
                # 默认返回具体问题
                return "concrete"
        except Exception as e:
            print(f"[QuestionClassifier] LLM分类失败: {e}，回退到规则分类")
            return None
    
    def classify(self, query: str) -> dict:
        """
        分类问题类型
        
        Returns:
            dict: {
                "type": "abstract" | "concrete" | "mixed",
                "confidence": 0.0 ~ 1.0,
                "should_use_hyde": bool,
                "abstract_score": int,
                "concrete_score": int,
                "method": "rule_based" | "llm_based"
            }
        """
        if not query or len(query.strip()) < 3:
            return {
                "type": "concrete",
                "confidence": 0.5,
                "should_use_hyde": False,
                "abstract_score": 0,
                "concrete_score": 0,
                "method": "rule_based"
            }
        
        # 获取规则分类结果
        rule_type, abs_score, conc_score = self._rule_based_classify(query)
        
        if self.use_llm_classifier:
            llm_type = self._llm_based_classify(query)
            if llm_type is not None:
                # 结合规则和LLM结果
                if llm_type == rule_type:
                    confidence = min(0.95, 0.6 + (abs_score + conc_score) * 0.05)
                    final_type = llm_type
                else:
                    # LLM结果权重更高
                    confidence = 0.7
                    final_type = llm_type
                method = "llm_based"
            else:
                final_type = rule_type
                confidence = min(0.8, 0.5 + (abs_score + conc_score) * 0.05)
                method = "rule_based"
        else:
            final_type = rule_type
            confidence = min(0.8, 0.5 + (abs_score + conc_score) * 0.05)
            method = "rule_based"
        
        return {
            "type": final_type,
            "confidence": round(confidence, 2),
            "should_use_hyde": (final_type == "abstract"),
            "abstract_score": abs_score,
            "concrete_score": conc_score,
            "method": method
        }


# 单例模式
_classifier_instance = None

def get_classifier() -> QuestionClassifier:
    """获取分类器单例"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QuestionClassifier(use_llm_classifier=True)
    return _classifier_instance


# 测试
if __name__ == "__main__":
    classifier = QuestionClassifier(use_llm_classifier=True)
    
    test_cases = [
        "Attention公式是什么？",
        "如何理解注意力机制在Transformer中的作用？",
        "PyTorch如何实现混合精度训练？",
        "谈谈你对RAG的理解",
        "为什么现在大模型都是decoder-only架构？",
        "什么是KV Cache？",
        "如何设计一个高性能的检索系统？",
        "LLM的训练流程是怎样的？",
    ]
    
    for query in test_cases:
        result = classifier.classify(query)
        print(f"问题: {query}")
        print(f"  类型: {result['type']}, 置信度: {result['confidence']}, HyDE: {result['should_use_hyde']}")
        print()
