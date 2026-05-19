#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例1：基础RAG流水线

完整演示RAG的四步流程：查询 → 检索 → 增强 → 生成

【运行流程】
1. 加载知识库文档
2. 切分文档为小片段
3. 嵌入并存储到ChromaDB
4. 构建RAG Chain
5. 提问并获取基于知识库的回答
"""

import os
from rag_pipeline import (
    create_llm,
    create_embeddings,
    load_documents,
    split_documents,
    create_vector_store,
    load_vector_store,
    similarity_search,
    create_rag_chain,
)


def main():
    print("=" * 60)
    print("示例1：基础RAG流水线")
    print("=" * 60)
    print()
    
    # ========== 步骤1：创建LLM和嵌入模型 ==========
    print("📋 步骤1：初始化模型")
    print("-" * 50)
    
    llm = create_llm(model="qwen-plus", temperature=0.1)
    embeddings = create_embeddings(model="text-embedding-v3")
    print()
    
    # ========== 步骤2：加载并切分文档 ==========
    print("📋 步骤2：加载并切分文档")
    print("-" * 50)
    
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    
    # 加载 knowledge_base 目录下的所有txt文件
    docs = load_documents(knowledge_dir, glob_pattern="**/*.txt")
    
    # 切分文档
    # chunk_size=500: 每个片段最多500字符
    # chunk_overlap=50: 相邻片段重叠50字符
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    print()
    
    # ========== 步骤3：创建向量存储 ==========
    print("📋 步骤3：创建向量存储")
    print("-" * 50)
    
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    
    # 检查是否已有向量存储
    # 如果有，直接加载（节省时间和API调用）
    # 如果没有，创建新的（需要调用Embedding API）
    if os.path.exists(db_dir) and os.listdir(db_dir):
        print("检测到已有向量存储，直接加载...")
        vector_store = load_vector_store(embeddings, db_dir)
    else:
        print("首次运行，创建新的向量存储...")
        vector_store = create_vector_store(chunks, embeddings, db_dir)
    print()
    
    # ========== 步骤4：测试相似度检索 ==========
    print("📋 步骤4：测试相似度检索")
    print("-" * 50)
    
    # 先单独测试检索功能，看看能找到什么
    # 这一步不是RAG必需的，但有助于理解检索效果
    results = similarity_search(vector_store, "什么是机器学习？", k=2)
    print()
    
    # ========== 步骤5：构建RAG Chain并提问 ==========
    print("📋 步骤5：构建RAG Chain并提问")
    print("-" * 50)
    
    rag_chain = create_rag_chain(llm, vector_store, k=3)
    print()
    
    # 提问1：知识库中有答案的问题
    print("🤔 问题1：什么是机器学习？")
    answer = rag_chain.invoke("什么是机器学习？")
    print(f"💡 回答：{answer}")
    print()
    
    # 提问2：知识库中有答案的问题
    print("🤔 问题2：深度学习有哪些应用？")
    answer = rag_chain.invoke("深度学习有哪些应用？")
    print(f"💡 回答：{answer}")
    print()
    
    # 提问3：知识库中可能没有答案的问题
    print("🤔 问题3：如何制作披萨？")
    answer = rag_chain.invoke("如何制作披萨？")
    print(f"💡 回答：{answer}")
    print()
    
    # ========== 对比：有RAG vs 无RAG ==========
    print("=" * 60)
    print("对比：有RAG vs 无RAG")
    print("=" * 60)
    print()
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    # 无RAG：直接问LLM
    no_rag_chain = (
        ChatPromptTemplate.from_messages([
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    
    question = "阿里云百炼平台支持哪些嵌入模型？"
    
    print(f"🤔 问题：{question}")
    print()
    
    print("📝 无RAG回答（仅靠LLM训练数据）：")
    no_rag_answer = no_rag_chain.invoke({"question": question})
    print(f"   {no_rag_answer}")
    print()
    
    print("📝 有RAG回答（基于知识库检索）：")
    rag_answer = rag_chain.invoke(question)
    print(f"   {rag_answer}")
    print()


if __name__ == "__main__":
    main()
