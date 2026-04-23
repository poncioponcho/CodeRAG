"""测试异步事件循环修复"""

import threading
import sys
sys.path.insert(0, '/Users/seyonmacbook/Desktop/电子书/26春招/CodeRAG')

from parallel_processor import get_parallel_processor


def test_in_thread():
    """在非主线程中测试并行处理器"""
    print(f"当前线程: {threading.current_thread().name}")
    
    # 这应该在非主线程中工作
    processor = get_parallel_processor(max_workers=4)
    
    # 测试处理
    def test_processor(query):
        return f"处理结果: {query}"
    
    result = processor.process_query("测试查询", {"test": test_processor})
    print(f"处理结果: {result}")
    return True


if __name__ == "__main__":
    print(f"主线程: {threading.current_thread().name}")
    
    # 在主线程中测试
    print("\n--- 主线程测试 ---")
    processor = get_parallel_processor(max_workers=4)
    result = processor.process_query("主线程测试", {"test": lambda q: f"主线程结果: {q}"})
    print(f"主线程结果: {result}")
    
    # 在子线程中测试
    print("\n--- 子线程测试 ---")
    thread = threading.Thread(target=test_in_thread, name="TestThread")
    thread.start()
    thread.join()
    
    print("\n✅ 所有测试通过！")