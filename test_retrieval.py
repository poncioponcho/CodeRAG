"""检索效果测试：验证优化后的分词逻辑对检索覆盖率的影响"""

import time
import pickle
import sys


def test_dd_formula_retrieval():
    """测试关键查询: d*d 公式含义 的检索效果"""
    print("\n" + "=" * 80)
    print("检索测试：验证 'd*d 公式含义' 查询的 BM25 召回能力")
    print("=" * 80)
    
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from retrieval_core import HybridRetriever, hybrid_tokenize
        
        print("\n加载模型和数据...")
        
        embedding = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh",
            model_kwargs={"local_files_only": False},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vectorstore = FAISS.load_local("./faiss_index", embedding, allow_dangerous_deserialization=True)
        
        with open("./chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        
        print(f"✓ 加载完成: {len(chunks)} chunks")
        
        print("\n构建 HybridRetriever（使用优化后的混合分词）...")
        start_time = time.time()
        retriever = HybridRetriever(vectorstore, chunks, vec_k=10, bm25_k=10)
        build_time = time.time() - start_time
        print(f"✓ 索引构建完成，耗时: {build_time:.2f}秒")
        
        query = "d*d 公式含义"
        print(f"\n执行查询: '{query}'")
        print("-" * 80)
        
        print("\n分词结果:")
        query_tokens = hybrid_tokenize(query)
        print(f"  Tokens: {query_tokens}")
        assert "d*d" in query_tokens, "✗ 关键token 'd*d' 未被保留"
        print("  ✓ 'd*d' 被正确保留为完整token")
        
        print("\n执行检索...")
        start_time = time.time()
        results = retriever.invoke(query)
        retrieval_time = time.time() - start_time
        print(f"✓ 检索完成，耗时: {retrieval_time*1000:.2f}ms")
        print(f"✓ 召回文档数: {len(results)}")
        
        if len(results) > 0:
            print("\n召回结果（Top 5）:")
            print("-" * 80)
            for i, doc in enumerate(results[:5], 1):
                content_preview = doc.page_content[:150].replace('\n', ' ')
                source = doc.metadata.get('source', '未知')
                print(f"\n[{i}] 来源: {source}")
                print(f"    内容: {content_preview}...")
                
                if 'd*d' in doc.page_content:
                    print(f"    ★ 包含目标公式 'd*d'")
            
            dd_count = sum(1 for doc in results if 'd*d' in doc.page_content)
            print(f"\n{'='*80}")
            print(f"检索结果统计:")
            print(f"  • 总召回文档数: {len(results)}")
            print(f"  • 包含 'd*d' 的文档数: {dd_count}")
            print(f"  • 召回率: {dd_count/len(results)*100:.1f}%")
            
            if dd_count > 0:
                print(f"\n✅ 测试通过！成功召回包含 'd*d' 公式的文档")
                return True
            else:
                print(f"\n⚠️  未找到包含 'd*d' 的文档（可能数据集中不包含此内容）")
                return False
        else:
            print("⚠️  未召回任何文档")
            return False
            
    except FileNotFoundError as e:
        print(f"\n❌ 数据文件未找到: {e}")
        print("请确保已运行数据准备脚本生成 faiss_index 和 chunks.pkl")
        return False
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_old_vs_new_tokenization():
    """对比新旧分词方法在真实文档上的表现"""
    print("\n" + "=" * 80)
    print("新旧分词方法对比测试")
    print("=" * 80)
    
    try:
        import jieba
        from rank_bm25 import BM25Okapi
        from retrieval_core import hybrid_tokenize
        import pickle
        
        with open("./chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        
        sample_chunks = [chunk.page_content for chunk in chunks[:50]]
        
        print(f"\n使用 {len(sample_chunks)} 个样本chunks进行对比测试")
        
        old_tokenized = [list(jieba.cut(text)) for text in sample_chunks]
        new_tokenized = [hybrid_tokenize(text) for text in sample_chunks]
        
        test_queries = [
            "d*d 公式含义",
            "O(n^2) 复杂度",
            "ResNet-50 架构",
            "Attention机制"
        ]
        
        print("\n构建BM25索引...")
        old_bm25 = BM25Okapi(old_tokenized)
        new_bm25 = BM25Okapi(new_tokenized)
        
        print("\n查询对比:")
        print("-" * 80)
        print(f"{'查询':<25} {'旧方法命中':<15} {'新方法命中':<15} {'提升'}")
        print("-" * 80)
        
        improvements = []
        for query in test_queries:
            old_query_tokens = list(jieba.cut(query))
            new_query_tokens = hybrid_tokenize(query)
            
            old_scores = old_bm25.get_scores(old_query_tokens)
            new_scores = new_bm25.get_scores(new_query_tokens)
            
            old_hits = sum(1 for s in old_scores if s > 0)
            new_hits = sum(1 for s in new_scores if s > 0)
            
            improvement = ((new_hits - old_hits) / old_hits * 100) if old_hits > 0 else (100 if new_hits > 0 else 0)
            improvements.append(improvement)
            
            improvement_str = f"+{improvement:.0f}%" if improvement >= 0 else f"{improvement:.0f}%"
            print(f"{query:<25} {old_hits:<15} {new_hits:<15} {improvement_str}")
        
        avg_improvement = sum(improvements) / len(improvements)
        print("-" * 80)
        print(f"平均提升: {avg_improvement:+.1f}%")
        
        return avg_improvement > 0
        
    except Exception as e:
        print(f"\n❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_retrieval_tests():
    """运行所有检索相关测试"""
    print("\n开始检索效果测试...")
    
    test1_pass = test_dd_formula_retrieval()
    test2_pass = compare_old_vs_new_tokenization()
    
    print("\n" + "=" * 80)
    print("检索测试总结:")
    print("=" * 80)
    
    if test1_pass:
        print("✅ 'd*d 公式含义' 查询测试通过")
    else:
        print("⚠️  'd*d 公式含义' 查询需要人工验证")
    
    if test2_pass:
        print("✅ 新旧分词方法对比显示正向改进")
    else:
        print("⚠️  改进效果不明显")
    
    return test1_pass or test2_pass


if __name__ == "__main__":
    success = run_retrieval_tests()
    sys.exit(0 if success else 1)
