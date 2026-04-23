"""缓存管理模块：实现智能缓存机制，支持多级缓存和LRU策略。"""

import time
import hashlib
from collections import OrderedDict


class CacheManager:
    """智能缓存管理器"""
    
    def __init__(self, cache_type="memory", capacity=10000, expiration=86400):
        """
        初始化缓存管理器
        
        Args:
            cache_type: 缓存类型 ("memory" 或 "redis")
            capacity: 缓存容量
            expiration: 缓存过期时间（秒）
        """
        self.cache_type = cache_type
        self.capacity = capacity
        self.expiration = expiration
        self.redis_client = None
        
        if cache_type == "memory":
            # 内存缓存：使用OrderedDict实现LRU
            self.cache = OrderedDict()
            self.metadata = {}  # 存储过期时间
        elif cache_type == "redis":
            # Redis缓存：需要安装redis包
            try:
                import redis
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                print("[CacheManager] Redis缓存初始化成功")
            except Exception as e:
                print(f"[CacheManager] Redis初始化失败，回退到内存缓存: {e}")
                self.cache_type = "memory"
                self.cache = OrderedDict()
                self.metadata = {}
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}")
    
    def _generate_key(self, query, question_type=None, model_version="v1"):
        """生成缓存键"""
        key_str = f"{query}_{question_type or 'unknown'}_{model_version}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key):
        """获取缓存"""
        if self.cache_type == "redis":
            try:
                value = self.redis_client.get(key)
                if value:
                    import json
                    return json.loads(value)
                return None
            except Exception as e:
                print(f"[CacheManager] Redis获取失败: {e}")
                return None
        else:
            # 内存缓存
            if key in self.cache:
                # 检查过期时间
                if time.time() < self.metadata.get(key, 0):
                    # 更新访问顺序（LRU）
                    self.cache.move_to_end(key)
                    return self.cache[key]
                else:
                    # 缓存过期
                    self._remove(key)
                    return None
            return None
    
    def set(self, key, value, expire=None):
        """设置缓存"""
        expire_time = expire or self.expiration
        
        if self.cache_type == "redis":
            try:
                import json
                self.redis_client.setex(key, expire_time, json.dumps(value))
                return True
            except Exception as e:
                print(f"[CacheManager] Redis设置失败: {e}")
                return False
        else:
            # 内存缓存
            # 检查容量
            if len(self.cache) >= self.capacity:
                # 移除最久未使用的
                self._remove(next(iter(self.cache)))
            
            self.cache[key] = value
            self.metadata[key] = time.time() + expire_time
            # 更新访问顺序
            self.cache.move_to_end(key)
            return True
    
    def _remove(self, key):
        """移除缓存"""
        if self.cache_type == "redis":
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"[CacheManager] Redis删除失败: {e}")
        else:
            if key in self.cache:
                del self.cache[key]
            if key in self.metadata:
                del self.metadata[key]
    
    def invalidate(self, pattern=None):
        """使缓存失效"""
        if self.cache_type == "redis":
            try:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    self.redis_client.flushdb()
            except Exception as e:
                print(f"[CacheManager] Redis失效失败: {e}")
        else:
            if pattern:
                # 简单的模式匹配
                keys_to_remove = [k for k in self.cache if pattern in k]
                for key in keys_to_remove:
                    self._remove(key)
            else:
                self.cache.clear()
                self.metadata.clear()
    
    def get_stats(self):
        """获取缓存统计信息"""
        if self.cache_type == "redis":
            try:
                info = self.redis_client.info()
                return {
                    "type": "redis",
                    "keys": int(info.get("db0", {}).get("keys", 0)),
                    "memory_used": info.get("used_memory_human", "N/A")
                }
            except Exception as e:
                return {"type": "redis", "error": str(e)}
        else:
            return {
                "type": "memory",
                "size": len(self.cache),
                "capacity": self.capacity,
                "expiration": self.expiration
            }
    
    def clear(self):
        """清空缓存"""
        self.invalidate()


# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager(cache_type="memory", capacity=10000, expiration=86400):
    """获取缓存管理器单例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(cache_type, capacity, expiration)
    return _cache_manager


if __name__ == "__main__":
    # 测试缓存管理器
    cache = get_cache_manager()
    
    # 测试设置和获取
    key = "test_key"
    value = "test_value"
    cache.set(key, value)
    result = cache.get(key)
    print(f"设置缓存: {value}, 获取缓存: {result}")
    
    # 测试LRU
    for i in range(11):
        cache.set(f"key{i}", f"value{i}")
    print(f"缓存大小: {len(cache.cache)}")
    print(f"key0是否存在: {'key0' in cache.cache}")
    print(f"key10是否存在: {'key10' in cache.cache}")
    
    # 测试过期
    cache.set("expire_key", "expire_value", expire=1)
    print(f"过期前: {cache.get('expire_key')}")
    time.sleep(2)
    print(f"过期后: {cache.get('expire_key')}")
    
    # 测试统计信息
    print(f"缓存统计: {cache.get_stats()}")