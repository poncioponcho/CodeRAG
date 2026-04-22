"""兼容层：RerankRetriever 已迁移至 retrieval_core.py。
保留此文件以维持旧代码的向后兼容导入。
"""
from retrieval_core import RerankRetriever

__all__ = ["RerankRetriever"]
