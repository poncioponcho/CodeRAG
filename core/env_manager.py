#!/usr/bin/env python3
"""
环境变量管理模块

负责加载和管理环境变量，确保敏感信息不硬编码在代码中
"""

import os
import dotenv


class EnvironmentManager:
    """
    环境变量管理器
    """
    
    def __init__(self, env_file: str = '.env'):
        """
        初始化环境变量管理器
        
        Args:
            env_file: 环境变量文件路径
        """
        # 加载环境变量文件
        dotenv.load_dotenv(env_file)
        
        # 配置默认值
        self.defaults = {
            'OLLAMA_URL': 'http://localhost:11434',
            'OLLAMA_MODEL': 'qwen2.5:7b',
            'CACHE_DIR': './cache',
            'EMBEDDING_MODEL_PATH': 'models/bge-small-zh-onnx/model.onnx',
            'RERANKER_MODEL_PATH': 'models/crossencoder-fp32/model.onnx',
            'EVALUATION_THRESHOLD': '0.55',
            'OLLAMA_KEEP_ALIVE': '1h'
        }
    
    def get(self, key: str, default=None) -> str:
        """
        获取环境变量
        
        Args:
            key: 环境变量键
            default: 默认值
            
        Returns:
            环境变量值
        """
        if default is None:
            default = self.defaults.get(key, None)
        
        return os.getenv(key, default)
    
    def get_bool(self, key: str, default=False) -> bool:
        """
        获取布尔类型环境变量
        
        Args:
            key: 环境变量键
            default: 默认值
            
        Returns:
            布尔值
        """
        value = self.get(key)
        if value is None:
            return default
        
        return str(value).lower() in ('true', '1', 'yes', 'y', 'on')
    
    def get_int(self, key: str, default=0) -> int:
        """
        获取整数类型环境变量
        
        Args:
            key: 环境变量键
            default: 默认值
            
        Returns:
            整数值
        """
        value = self.get(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_float(self, key: str, default=0.0) -> float:
        """
        获取浮点数类型环境变量
        
        Args:
            key: 环境变量键
            default: 默认值
            
        Returns:
            浮点数值
        """
        value = self.get(key)
        if value is None:
            return default
        
        try:
            return float(value)
        except ValueError:
            return default
    
    def get_ollama_url(self) -> str:
        """
        获取 Ollama API URL
        
        Returns:
            Ollama API URL
        """
        return self.get('OLLAMA_URL')
    
    def get_ollama_model(self) -> str:
        """
        获取 Ollama 模型名称
        
        Returns:
            模型名称
        """
        return self.get('OLLAMA_MODEL')
    
    def get_cache_dir(self) -> str:
        """
        获取缓存目录
        
        Returns:
            缓存目录路径
        """
        return self.get('CACHE_DIR')
    
    def get_embedding_model_path(self) -> str:
        """
        获取 Embedding 模型路径
        
        Returns:
            模型路径
        """
        return self.get('EMBEDDING_MODEL_PATH')
    
    def get_reranker_model_path(self) -> str:
        """
        获取 Reranker 模型路径
        
        Returns:
            模型路径
        """
        return self.get('RERANKER_MODEL_PATH')
    
    def get_evaluation_threshold(self) -> float:
        """
        获取评估阈值
        
        Returns:
            评估阈值
        """
        return self.get_float('EVALUATION_THRESHOLD')


# 全局实例
env_manager = EnvironmentManager()
