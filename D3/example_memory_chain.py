#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例2：带Memory的对话Chain

展示LangChain的对话历史管理（Memory），实现多轮对话。

【什么是Memory？】
Memory就是"记忆"，让AI能记住之前说过的话。
D1中用 conversation_history 列表手动管理，D3中用 LangChain 的 Memory 自动管理。

【核心组件】
1. ChatMessageHistory  — 存储单个会话的消息记录
2. session_store       — 字典，存储多个会话（key=session_id, value=ChatMessageHistory）
3. RunnableWithMessageHistory — 包装器，自动在调用前后处理历史消息

【与D1的对比】
D1: 
    llm = LLMClient()
    llm.chat("我叫小明")           # 手动追加到 conversation_history
    llm.chat("我叫什么？")         # 手动从 conversation_history 取出
    llm.clear_conversation()       # 手动清空 conversation_history

D3:
    chain = create_chain(llm, use_memory=True)
    chain.invoke({"input": "我叫小明"}, config={"configurable": {"session_id": "s1"}})
    chain.invoke({"input": "我叫什么？"}, config={"configurable": {"session_id": "s1"}})
    # 自动从 session_store["s1"] 取历史、保存回复
"""

from llm_langchain import create_llm, create_chain, clear_session


def main():
    print("=" * 60)
    print("示例2：带Memory的对话Chain")
    print("=" * 60)
    print()
    
    # ========== 步骤1：创建带Memory的Chain ==========
    # use_memory=True 会自动用 RunnableWithMessageHistory 包装Chain
    # 调用时需要通过 config 指定 session_id
    llm = create_llm(model="qwen-plus", temperature=0.7)
    chain = create_chain(llm, use_memory=True)
    print()
    
    # ========== 步骤2：多轮对话（同一session） ==========
    print("-" * 50)
    print("2.1 多轮对话 — 同一session_id，AI会记住上下文")
    print("-" * 50)
    
    # config 中的 session_id 决定了使用哪份对话历史
    # 相同的 session_id = 同一个对话，AI会记住之前说的内容
    # 不同的 session_id = 不同的对话，互不影响
    config = {"configurable": {"session_id": "user_001"}}
    
    # 第一轮：告诉AI你的名字
    reply = chain.invoke({"input": "我叫小明，今年25岁，我是一名程序员"}, config=config)
    print(f"用户: 我叫小明，今年25岁，我是一名程序员")
    print(f"AI: {reply}")
    print()
    
    # 第二轮：问AI你的名字，它应该能记住
    reply = chain.invoke({"input": "我叫什么名字？今年多大？做什么工作？"}, config=config)
    print(f"用户: 我叫什么名字？今年多大？做什么工作？")
    print(f"AI: {reply}")
    print()
    
    # 第三轮：继续在同一session中对话
    reply = chain.invoke({"input": "给我推荐一门适合我学习的编程语言"}, config=config)
    print(f"用户: 给我推荐一门适合我学习的编程语言")
    print(f"AI: {reply}")
    print()
    
    # ========== 步骤3：不同session隔离 ==========
    print("-" * 50)
    print("2.2 不同session隔离 — 不同session_id互不影响")
    print("-" * 50)
    
    # 使用新的 session_id，这是一个全新的对话
    # AI不会知道之前 session "user_001" 中说的内容
    config_user2 = {"configurable": {"session_id": "user_002"}}
    
    reply = chain.invoke({"input": "我叫什么名字？"}, config=config_user2)
    print(f"用户(user_002): 我叫什么名字？")
    print(f"AI: {reply}")
    print()
    
    # 但 user_001 的对话还在，切回去仍然记得
    reply = chain.invoke({"input": "你还记得我的名字吗？"}, config=config)
    print(f"用户(user_001): 你还记得我的名字吗？")
    print(f"AI: {reply}")
    print()
    
    # ========== 步骤4：清空对话历史 ==========
    print("-" * 50)
    print("2.3 清空对话历史")
    print("-" * 50)
    
    # 清空 user_001 的对话历史
    # 相当于 D1 中的 llm.clear_conversation()
    clear_session("user_001")
    
    # 再问同样的问题，AI应该不记得了
    reply = chain.invoke({"input": "我叫什么名字？"}, config=config)
    print(f"用户(user_001): 我叫什么名字？")
    print(f"AI: {reply}")
    print()
    
    # ========== 步骤5：带System Prompt的Memory Chain ==========
    print("-" * 50)
    print("2.4 带System Prompt的Memory Chain")
    print("-" * 50)
    
    # 创建一个带自定义System Prompt和Memory的Chain
    # 相当于 D1 中先 set_system_prompt() 再 chat()
    expert_chain = create_chain(
        llm,
        system_prompt="你是一个友好的英语老师，用简单易懂的方式教英语，适当用中文解释。",
        use_memory=True
    )
    
    config_teacher = {"configurable": {"session_id": "english_class"}}
    
    reply = expert_chain.invoke({"input": "我想学习英语时态"}, config=config_teacher)
    print(f"用户: 我想学习英语时态")
    print(f"英语老师: {reply}")
    print()
    
    reply = expert_chain.invoke({"input": "能给我举个例子吗？"}, config=config_teacher)
    print(f"用户: 能给我举个例子吗？")
    print(f"英语老师: {reply}")
    print()


if __name__ == "__main__":
    main()
