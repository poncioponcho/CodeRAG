import os
import warnings
import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)


def init_docs():
    os.makedirs("docs", exist_ok=True)
    files = {
        "attention.md": """# 注意力机制详解
## Self-Attention 公式
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
## 面试常考点
为什么要除以 sqrt(d_k)？防止点积过大导致 softmax 梯度消失。
时间和空间复杂度都是 O(n^2)，长文本优化用 KV Cache。
""",
        "pytorch_tips.md": """# PyTorch 工程实践
## 显存优化三板斧
1. gradient checkpointing：用时间换空间
2. mixed precision：torch.cuda.amp
3. DataLoader 的 pin_memory=True 加速 CPU→GPU 传输
## 分布式训练
DDP 启动命令：torchrun --nproc_per_node=4 train.py
""",
        "agent_patterns.md": """# Agent 设计模式
## ReAct 循环
Reason（推理）→ Act（行动）→ Observation（观察）→ 重复
## 记忆设计
LangGraph 的 MemorySaver 按 thread_id 隔离会话。
## RAG vs Fine-tuning
RAG 适合知识频繁更新；Fine-tuning 适合改变模型行为。
"""
    }
    for name, content in files.items():
        path = f"docs/{name}"
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)


def build_knowledge_base():
    docs = []
    for filename in os.listdir("docs"):
        if filename.endswith(".md"):
            with open(f"docs/{filename}", "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"共加载 {len(docs)} 个文档，切分为 {len(chunks)} 个 chunk")
    return chunks


def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="./chroma_db")
    return vectorstore


def ollama_generate(prompt: str, model: str = "qwen3", temperature: float = 0.1) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        }
    )
    resp.raise_for_status()
    return resp.json()["response"]


def main():
    init_docs()

    if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
        print(">>> 检测到已有向量库，直接加载")
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
    else:
        chunks = build_knowledge_base()
        vectorstore = build_vectorstore(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n>>> 本地 RAG 已启动，输入问题（exit 退出）")
    history = []

    while True:
        query = input("\n你: ").strip()
        if query.lower() in ["exit", "quit", "退出"]:
            break

        docs = retriever.invoke(query)
        context = "\n\n".join([f"[{d.metadata['source']}] {d.page_content}" for d in docs])

        history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history[-2:]])

        prompt = f"""你是技术面试助手。基于以下参考资料用中文回答问题。
如果参考资料中没有相关信息，请明确说明"资料中未提及"。

参考资料：
{context}

历史对话：
{history_str}

当前问题：{query}

请回答："""

        answer = ollama_generate(prompt)
        print(f"AI: {answer}")
        history.append((query, answer))


if __name__ == "__main__":
    main()