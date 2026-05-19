#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例2：带Memory的RAG

结合D3的对话历史管理，实现多轮对话+知识库检索。

【与普通RAG的区别】
普通RAG：每次提问都是独立的，不记得之前问过什么
带Memory的RAG：记住对话历史，可以追问

【使用场景】
- 知识库问答：先问一个概念，再追问细节
- 客服系统：多轮对话中持续检索知识库
- 学习助手：基于教材进行连续提问
"""

import os
from rag_pipeline import (
    create_llm,
    create_embeddings,
    load_documents,
    split_documents,
    create_vector_store,
    load_vector_store,
    create_rag_chain_with_memory,
    clear_session,
)


def main():
    print("=" * 60)
    print("示例2：带Memory的RAG")
    print("=" * 60)
    print()
    
    # ========== 初始化 ==========
    llm = create_llm(model="qwen-plus", temperature=0.1)
    embeddings = create_embeddings(model="text-embedding-v3")
    
    knowledge_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    docs = load_documents(knowledge_dir, glob_pattern="**/*.txt")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    
    db_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    if os.path.exists(db_dir) and os.listdir(db_dir):
        vector_store = load_vector_store(embeddings, db_dir)
    else:
        vector_store = create_vector_store(chunks, embeddings, db_dir)
    print()
    
    # ========== 构建带Memory的RAG Chain ==========
    rag_chain = create_rag_chain_with_memory(llm, vector_store, k=3)
    print()
    
    # ========== 多轮对话演示 ==========
    config = {"configurable": {"session_id": "rag_session_1"}}
    
    # 第一轮：问一个概念
    print("-" * 50)
    print("第一轮对话")
    print("-" * 50)
    
    question1 = "什么是自然语言处理？"
    print(f"🤔 用户: {question1}")
    answer1 = rag_chain.invoke({"question": question1}, config=config)
    print(f"💡 AI: {answer1}")
    print()
    
    # 第二轮：追问细节（AI知道"它"指的是"自然语言处理"）
    question2 = "它有哪些主要任务？"
    print(f"🤔 用户: {question2}")
    answer2 = rag_chain.invoke({"question": question2}, config=config)
    print(f"💡 AI: {answer2}")
    print()
    
    # 第三轮：继续追问
    question3 = "请举一个实际应用的例子"
    print(f"🤔 用户: {question3}")
    answer3 = rag_chain.invoke({"question": question3}, config=config)
    print(f"💡 AI: {answer3}")
    print()
    
    # ========== 新session测试隔离 ==========
    print("-" * 50)
    print("新session测试（AI不记得之前的对话）")
    print("-" * 50)
    
    config2 = {"configurable": {"session_id": "rag_session_2"}}
    
    question4 = "它有哪些主要任务？"
    print(f"🤔 用户: {question4}")
    answer4 = rag_chain.invoke({"question": question4}, config=config2)
    print(f"💡 AI: {answer4}")
    print()
    
    # ========== 清空历史重新开始 ==========
    print("-" * 50)
    print("清空历史重新开始")
    print("-" * 50)
    
    clear_session("rag_session_1")
    
    question5 = "我们之前聊了什么？"
    print(f"🤔 用户: {question5}")
    answer5 = rag_chain.invoke({"question": question5}, config=config)
    print(f"💡 AI: {answer5}")
    print()


if __name__ == "__main__":
    main()
