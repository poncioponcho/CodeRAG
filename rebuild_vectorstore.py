"""重建向量库：使用bge-small-zh模型重新创建FAISS索引"""

import os
import shutil
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from retrieval_core import split_by_headings


def main():
    print("加载bge-small-zh embedding模型...")
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"local_files_only": False},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    print("读取文档...")
    all_docs = []
    for p in Path("docs").glob("*"):
        if p.suffix.lower() in [".txt", ".md"]:
            print(f"  读取 {p.name}")
            all_docs.append(Document(
                page_content=p.read_text(encoding="utf-8"),
                metadata={"source": p.name}
            ))
    
    print(f"\n切分文档为chunks...")
    chunks = []
    for doc in all_docs:
        doc_chunks = split_by_headings(doc.page_content, doc.metadata["source"])
        chunks.extend(doc_chunks)
    
    print(f"共生成 {len(chunks)} 个chunks")
    
    # 删除旧的向量库
    if os.path.exists("./faiss_index"):
        print("删除旧的向量库...")
        shutil.rmtree("./faiss_index", ignore_errors=True)
    
    print("创建FAISS向量库...")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("./faiss_index")
    
    print("\n✅ 向量库重建完成！")
    return chunks


if __name__ == "__main__":
    chunks = main()
    # 保存chunks到文件
    import pickle
    with open("./chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("chunks.pkl 已保存")
