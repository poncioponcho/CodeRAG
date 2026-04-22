"""语义相似度匹配模块

使用 SentenceTransformer 模型计算文本之间的语义相似度，用于改进测试集筛选逻辑。
"""

import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticMatcher:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        初始化语义匹配器
        
        Args:
            model_name: SentenceTransformer 模型名称
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def compute_similarity(self, text1, text2):
        """
        计算两个文本之间的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 相似度分数 (0-1)
        """
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
        cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return cosine_score
    
    def match_point_in_text(self, point, text, threshold=0.5):
        """
        检查要点是否在文本中有语义匹配
        
        Args:
            point: 要点文本
            text: 文档文本
            threshold: 相似度阈值
            
        Returns:
            bool: 是否匹配
        """
        # 处理短文本情况
        if len(point) < 5:
            # 检查精确匹配
            if point.lower() in text.lower():
                return True
            
            # 处理常见中英文对应关系
            short_term_map = {
                '注意力': ['attention', '自注意力'],
                'attention': ['注意力', '自注意力'],
                'CNN': ['卷积神经网络'],
                '卷积神经网络': ['CNN'],
                'RAG': ['检索增强生成'],
                '检索增强生成': ['RAG'],
                'NLP': ['自然语言处理'],
                '自然语言处理': ['NLP'],
                'GPT': ['生成式预训练Transformer'],
                '生成式预训练Transformer': ['GPT'],
            }
            
            # 检查短文本映射
            if point in short_term_map:
                for term in short_term_map[point]:
                    if term in text:
                        return True
            
            # 检查文本中的短文本映射
            for key, values in short_term_map.items():
                if point in values:
                    if key in text:
                        return True
            
            return False
        
        # 检查精确匹配
        if point.lower() in text.lower():
            return True
        
        # 处理缩写和首字母缩写
        # 提取大写字母组成的缩写
        acronyms = re.findall(r'[A-Z]{2,}', point)
        for acronym in acronyms:
            if acronym in text:
                return True
        
        # 处理首字母缩写与全称的匹配
        # 常见技术缩写映射（包含中英文全称）
        acronym_map = {
            'CNN': ['convolutional neural network', '卷积神经网络'],
            'NLP': ['natural language processing', '自然语言处理'],
            'RAG': ['retrieval augmented generation', '检索增强生成'],
            'GPT': ['generative pre-trained transformer', '生成式预训练Transformer'],
            'LSTM': ['long short-term memory', '长短期记忆网络'],
            'RNN': ['recurrent neural network', '循环神经网络'],
            'SVM': ['support vector machine', '支持向量机'],
            'KNN': ['k-nearest neighbors', 'k最近邻'],
            'PCA': ['principal component analysis', '主成分分析'],
            'MLM': ['masked language modeling', '掩码语言建模'],
        }
        
        # 检查点中的缩写是否对应文本中的全称
        for acronym, full_forms in acronym_map.items():
            if acronym in point:
                for full_form in full_forms:
                    if full_form.lower() in text.lower():
                        return True
            for full_form in full_forms:
                if full_form.lower() in point.lower():
                    if acronym in text:
                        return True
        
        # 处理常见技术术语变体
        term_variants = {
            'batch normalization': ['batchnorm', 'batch norm'],
            'self-attention': ['self attention', '自注意力'],
            'convolutional neural network': ['CNN', '卷积神经网络'],
            'natural language processing': ['NLP', '自然语言处理'],
            'retrieval augmented generation': ['RAG', '检索增强生成'],
            'generative pre-trained transformer': ['GPT', '生成式预训练Transformer'],
            'masked language modeling': ['MLM'],
            'long short-term memory': ['LSTM'],
            'recurrent neural network': ['RNN'],
            'support vector machine': ['SVM'],
            'decision tree': ['决策树'],
            'random forest': ['随机森林'],
            'gradient boosting': ['梯度提升'],
            'k-nearest neighbors': ['KNN'],
            'principal component analysis': ['PCA'],
            'linear regression': ['线性回归'],
            'logistic regression': ['逻辑回归'],
            'neural network': ['神经网络'],
            'deep learning': ['深度学习'],
            'machine learning': ['机器学习'],
            'word embedding': ['词嵌入', '词向量'],
            '词嵌入': ['word embedding', '词向量'],
            '注意力': ['attention', '自注意力'],
            'attention': ['注意力', '自注意力'],
        }
        
        # 检查术语变体
        point_lower = point.lower()
        for term, variants in term_variants.items():
            if term in point_lower:
                for variant in variants:
                    if variant.lower() in text.lower():
                        return True
            for variant in variants:
                if variant.lower() in point_lower:
                    if term.lower() in text.lower():
                        return True
        
        # 计算语义相似度
        similarity = self.compute_similarity(point, text)
        return similarity >= threshold
    
    def batch_match_points(self, points, text, threshold=0.5):
        """
        批量检查多个要点是否在文本中有语义匹配
        
        Args:
            points: 要点列表
            text: 文档文本
            threshold: 相似度阈值
            
        Returns:
            list: 匹配结果列表
        """
        results = []
        for point in points:
            results.append(self.match_point_in_text(point, text, threshold))
        return results

# 全局语义匹配器实例
matcher = SemanticMatcher()

# 便捷函数
def semantic_match(point, text, threshold=0.5):
    """
    便捷函数：检查要点是否在文本中有语义匹配
    """
    return matcher.match_point_in_text(point, text, threshold)

def get_similarity_score(text1, text2):
    """
    便捷函数：计算两个文本之间的语义相似度
    """
    return matcher.compute_similarity(text1, text2)