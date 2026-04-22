import os
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.documents import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# ========== 0. 自动生成测试文档 ==========
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

# ========== 1. 加载与切分（手动读取，绕过 DirectoryLoader 兼容 bug） ==========
def build_knowledge_base():
    docs = []
    for filename in os.listdir("docs"):
        if filename.endswith(".md"):
            with open(f"docs/{filename}", "r", encoding="utf-8") as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": filename}))
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"共加载 {len(docs)} 个文档，切分为 {len(chunks)} 个 chunk")
    if not chunks:
        raise ValueError("没有生成任何 chunk，请检查 docs/ 目录下的文件内容")
    return chunks

# ========== 2. 本地向量库 ==========
def build_vectorstore(chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        chunks, 
        embedding, 
        persist_directory="./chroma_db"
    )
    return vectorstore

# ========== 3. Agent 组装 ==========
def main():
    init_docs()
    chunks = build_knowledge_base()
    vectorstore = build_vectorstore(chunks)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    @tool
    def search_notes(query: str) -> str:
        """搜索本地技术笔记，用于回答编程、深度学习、Agent 开发相关问题。
        输入应该是简洁的搜索关键词或问题。
        """
        docs = retriever.invoke(query)
        if not docs:
            return "未找到相关笔记。"
        return "\n\n".join([f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in docs])
    
    model = ChatOpenAI(
        model="alibaba/qwen-2.5-7b-instruct:free",  
        api_key=os.getenv("KIMI_API_KEY"),  # type: ignore
        base_url="https://api.moonshot.cn/v1",
        temperature=0.1
    )
    
    tools = [search_notes]
    checkpointer = MemorySaver()
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer
    )
    
    thread_config = {"configurable": {"thread_id": "1"}}
    
    print(">>> 第一轮：问注意力机制面试考点")
    result1 = agent.invoke(
        {"messages": [HumanMessage(content="我面试被问到为什么要除以 sqrt(d_k)，怎么回答？")]},
        config=thread_config# type: ignore
    )
    print(result1["messages"][-1].content)
    
    print("\n>>> 第二轮：追问 PyTorch 显存优化")
    result2 = agent.invoke(
        {"messages": [HumanMessage(content="那 PyTorch 训练时显存不够怎么办？")]},
        config=thread_config# type: ignore
    )
    print(result2["messages"][-1].content)

if __name__ == "__main__":
    main()