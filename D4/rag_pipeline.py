#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG核心流水线

实现完整的RAG（Retrieval-Augmented Generation）流水线：
查询 → 检索 → 增强 → 生成

【RAG是什么？】
RAG = 检索增强生成，让LLM基于外部知识库回答问题，而不是仅靠训练数据。
就像"开卷考试"：先从书本中找到相关内容，再基于这些内容回答问题。

【RAG四步流程】
① 查询（Query）：用户提出问题
② 检索（Retrieve）：从向量数据库中找到与问题最相关的文档片段
③ 增强（Augment）：把检索到的文档片段拼入Prompt，作为上下文
④ 生成（Generate）：LLM基于增强后的Prompt生成回答

【核心组件】
- 文档加载器：读取txt/pdf/md等文件
- 文本切分器：将长文档切分为小片段（Chunking）
- 嵌入模型：将文本转为向量（Embedding）
- 向量数据库：存储和检索向量（ChromaDB）
- RAG Chain：用LCEL构建的完整流水线
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ========== 会话存储（复用D3的设计） ==========
session_store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def clear_session(session_id: str):
    if session_id in session_store:
        session_store[session_id].clear()
        print(f"🗑️ 会话 {session_id} 的历史已清空")


# ========== 第一步：创建LLM和嵌入模型 ==========

def create_llm(
    model: str = "qwen-plus",
    temperature: float = 0.7,
    max_retries: int = 3,
    **kwargs
) -> ChatOpenAI:
    """
    创建LangChain的ChatOpenAI实例（与D3相同）
    """
    load_dotenv()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("API Key未设置！请在.env文件中配置DASHSCOPE_API_KEY")
    
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    print(f"✅ LLM初始化成功（模型: {model}）")
    return llm


def create_embeddings(
    model: str = "text-embedding-v3",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAIEmbeddings:
    """
    创建嵌入模型（Embedding Model）
    
    【什么是嵌入（Embedding）？】
    嵌入就是把文本转换成一组数字（向量），比如：
      "机器学习" → [0.12, -0.34, 0.56, ..., 0.78]  （1024维向量）
    
    语义相近的文本，向量也相近：
      "机器学习" 和 "深度学习" → 向量距离很近
      "机器学习" 和 "今天天气" → 向量距离很远
    
    这样就可以通过计算向量距离来找"最相关的文档片段"
    
    【与D3的对比】
    D3只有LLM（生成模型），D4新增了Embedding（嵌入模型）
    - LLM：文本 → 文本（问答、翻译、摘要）
    - Embedding：文本 → 向量（相似度检索）
    
    参数说明：
    model : str, 默认值 "text-embedding-v3"
        阿里云百炼支持的嵌入模型：
        - text-embedding-v3: 最新版，支持1024/768/512维度
        - text-embedding-v2: 上一版，1536维度
    """
    load_dotenv()
    resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not resolved_api_key:
        raise ValueError("API Key未设置！")
    
    resolved_base_url = base_url or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    embeddings = OpenAIEmbeddings(
        model=model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        check_embedding_ctx_length=False,
    )
    print(f"✅ 嵌入模型初始化成功（模型: {model}）")
    return embeddings


# ========== 第二步：文档加载与切分 ==========

def load_documents(
    path: str,
    glob_pattern: str = "**/*.txt",
) -> list:
    """
    加载文档
    
    【什么是文档加载？】
    把各种格式的文件（txt、pdf、md等）读取为LangChain的Document对象。
    每个Document包含：
    - page_content: 文本内容
    - metadata: 元数据（文件名、页码等）
    
    参数说明：
    path : str
        文件路径或目录路径
        - 如果是文件：直接加载该文件
        - 如果是目录：加载目录下所有匹配的文件
    
    glob_pattern : str, 默认值 "**/*.txt"
        文件匹配模式（仅在path为目录时生效）
        - "**/*.txt": 所有txt文件
        - "**/*.md": 所有Markdown文件
        - "**/*.{txt,md}": 所有txt和md文件
    """
    if os.path.isfile(path):
        # 单文件加载
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        print(f"📄 已加载文件: {path}（{len(docs)}个文档）")
    elif os.path.isdir(path):
        # 目录加载：自动扫描目录下所有匹配的文件
        loader = DirectoryLoader(
            path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        print(f"📄 已加载目录: {path}（{len(docs)}个文档）")
    else:
        raise FileNotFoundError(f"路径不存在: {path}")
    
    return docs


def split_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> list:
    """
    切分文档（Chunking）
    
    【为什么要切分？】
    1. LLM有上下文长度限制，不能把整本书塞进去
    2. 切成小片段后，可以只检索最相关的片段，提高准确性
    3. 小片段的嵌入向量更精确，语义更集中
    
    【切分策略：递归字符切分】
    RecursiveCharacterTextSplitter 的切分逻辑：
    1. 先按段落（\n\n）切分
    2. 如果段落太长，按句子（\n）切分
    3. 如果句子太长，按句号（。）切分
    4. 最后按字符切分
    
    这样可以尽量保持语义完整性，不会把一句话切成两半。
    
    参数说明：
    documents : list
        待切分的文档列表（由 load_documents 返回）
    
    chunk_size : int, 默认值 500
        每个片段的最大字符数
        - 太大：检索不精确，可能包含无关内容
        - 太小：语义不完整，可能缺少上下文
        - 建议：300-1000
    
    chunk_overlap : int, 默认值 50
        相邻片段的重叠字符数
        - 作用：避免关键信息正好在切分点被截断
        - 例如：片段1的末尾50字 = 片段2的开头50字
        - 建议：chunk_size的10%-20%
    
    separators : Optional[List[str]], 可选
        自定义切分分隔符列表
        - 默认：["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        - 可以根据文档特点自定义
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "，", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 文档已切分: {len(documents)}个文档 → {len(chunks)}个片段")
    print(f"   - 片段大小: {chunk_size}字符")
    print(f"   - 重叠大小: {chunk_overlap}字符")
    
    return chunks


# ========== 第三步：向量存储与检索 ==========

def create_vector_store(
    chunks: list,
    embeddings: OpenAIEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
) -> Chroma:
    """
    创建ChromaDB向量存储并保存文档片段
    
    【什么是向量数据库？】
    向量数据库专门存储和检索向量（高维数字数组）。
    它的核心能力是"相似度搜索"：给定一个查询向量，快速找到最相似的向量。
    
    【ChromaDB是什么？】
    ChromaDB是一个轻量级的开源向量数据库：
    - 不需要单独部署服务器（嵌入式运行）
    - 数据可以持久化到本地磁盘
    - 与LangChain无缝集成
    
    【数据存储过程】
    1. 对每个文档片段调用Embedding模型，得到向量
    2. 将向量和原文一起存入ChromaDB
    3. ChromaDB自动建立索引，加速检索
    
    参数说明：
    chunks : list
        切分后的文档片段列表（由 split_documents 返回）
    
    embeddings : OpenAIEmbeddings
        嵌入模型实例（由 create_embeddings 返回）
    
    persist_directory : str, 默认值 "./chroma_db"
        ChromaDB数据的持久化目录
        - 数据会保存到这个目录，下次可以直接加载，不需要重新嵌入
    
    collection_name : str, 默认值 "default"
        集合名称，类似于数据库中的"表名"
        - 不同的集合存储不同类型的文档
        - 例如："ai_knowledge"、"company_docs"
    """
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    
    print(f"💾 向量存储已创建:")
    print(f"   - 集合: {collection_name}")
    print(f"   - 片段数: {len(chunks)}")
    print(f"   - 持久化目录: {persist_directory}")
    
    return vector_store


def load_vector_store(
    embeddings: OpenAIEmbeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
) -> Chroma:
    """
    加载已有的ChromaDB向量存储
    
    与 create_vector_store 的区别：
    - create_vector_store：从文档片段新建向量存储（需要嵌入，较慢）
    - load_vector_store：从磁盘加载已有向量存储（不需要嵌入，很快）
    
    使用场景：
    - 第一次运行：create_vector_store（建库）
    - 后续运行：load_vector_store（直接用）
    """
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    
    count = vector_store._collection.count()
    print(f"📂 向量存储已加载:")
    print(f"   - 集合: {collection_name}")
    print(f"   - 片段数: {count}")
    
    return vector_store


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 3,
) -> list:
    """
    相似度检索
    
    【检索原理】
    1. 将查询文本转为向量（Embedding）
    2. 在向量数据库中计算与所有文档片段的向量距离
    3. 返回距离最近的k个文档片段
    
    【向量距离计算】
    常用的距离度量：
    - 余弦相似度：衡量向量方向的相似性（最常用）
    - 欧氏距离：衡量向量空间中的直线距离
    
    参数说明：
    vector_store : Chroma
        向量存储实例
    
    query : str
        查询文本（用户的问题）
    
    k : int, 默认值 3
        返回最相似的k个文档片段
        - k太小：可能遗漏重要信息
        - k太大：可能引入无关内容，且消耗更多token
        - 建议：3-5
    """
    results = vector_store.similarity_search(query, k=k)
    
    print(f"🔍 检索结果（查询: '{query[:30]}...'）:")
    for i, doc in enumerate(results):
        print(f"   [{i+1}] {doc.page_content[:80]}...")
    
    return results


# ========== 第四步：构建RAG Chain ==========

# RAG专用的Prompt模板
# 关键：{context} 是检索到的文档片段，{question} 是用户的问题
RAG_PROMPT_TEMPLATE = """你是一个知识库问答助手。请根据以下参考资料回答用户的问题。

要求：
1. 优先使用参考资料中的信息回答
2. 如果参考资料中没有相关信息，请诚实说明"知识库中暂无相关信息"
3. 不要编造参考资料中没有的内容

参考资料：
{context}

用户问题：{question}

回答："""


def create_rag_chain(
    llm: ChatOpenAI,
    vector_store: Chroma,
    k: int = 3,
):
    """
    构建完整的RAG Chain（用LCEL管道）
    
    【RAG Chain的数据流】
    输入: {"question": "什么是机器学习？"}
        ↓
    ① RunnableParallel 并行执行两个任务：
       - "context": retriever 检索相关文档片段
       - "question": RunnablePassthrough 直接传递原始问题
        ↓
    ② ChatPromptTemplate 将 context 和 question 填入模板
        ↓
    ③ ChatOpenAI 生成回答
        ↓
    ④ StrOutputParser 提取纯文本
    
    【与D3 Chain的对比】
    D3: prompt | llm | parser
        → 直接把用户输入传给LLM
    
    D4: 
        RunnableParallel({
            "context": retriever,          ← 新增：检索相关文档
            "question": RunnablePassthrough  ← 透传用户问题
        })
        | prompt | llm | parser
        → 先检索文档，再拼入Prompt，最后生成回答
    
    参数说明：
    llm : ChatOpenAI
        LLM实例
    
    vector_store : Chroma
        向量存储实例
    
    k : int, 默认值 3
        检索的文档片段数量
    """
    # 从向量存储创建检索器
    # retriever 是 LangChain 的检索器接口，封装了相似度搜索逻辑
    # invoke("查询文本") → 返回最相关的k个Document对象
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    
    # 创建RAG Prompt模板
    # {context} 会被检索到的文档片段填充
    # {question} 会被用户的原始问题填充
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE),
    ])
    
    # 用LCEL构建RAG Chain
    # RunnableParallel: 并行执行检索和透传问题
    #   - "context": retriever 检索相关文档
    #   - "question": RunnablePassthrough 直接传递原始输入中的question字段
    # 
    # 然后通过 | 管道传给 prompt → llm → parser
    rag_chain = (
        RunnableParallel({
            "context": retriever | (lambda docs: "\n\n---\n\n".join(
                doc.page_content for doc in docs
            )),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"🔗 RAG Chain已构建（检索top-{k}）")
    return rag_chain


def create_rag_chain_with_memory(
    llm: ChatOpenAI,
    vector_store: Chroma,
    k: int = 3,
):
    """
    构建带Memory的RAG Chain
    
    【与普通RAG Chain的区别】
    普通RAG Chain：每次提问都是独立的，不记得之前问过什么
    带Memory的RAG Chain：记住对话历史，可以追问
    
    例如：
    第1轮: "什么是机器学习？" → AI回答
    第2轮: "它有哪些应用？" → AI知道"它"指的是"机器学习"
    
    【实现方式】
    在RAG Chain的基础上，用 RunnableWithMessageHistory 包装
    同时在Prompt中加入 MessagesPlaceholder("history") 来插入历史消息
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    
    # 带Memory的Prompt模板
    # MessagesPlaceholder("history") 会被自动替换为对话历史
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT_TEMPLATE),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    
    # 构建Chain
    chain = (
        RunnableParallel({
            "context": retriever | (lambda docs: "\n\n---\n\n".join(
                doc.page_content for doc in docs
            )),
            "question": RunnablePassthrough(),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 用 RunnableWithMessageHistory 包装，添加Memory功能
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    print(f"🔗 带Memory的RAG Chain已构建（检索top-{k}）")
    return chain_with_memory
