#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例2：高级用法
展示如何使用不同参数、系统提示词、流式输出等高级功能
"""

from llm_client import LLMClient


def main():
    print("=" * 50)
    print("示例2：高级用法")
    print("=" * 50)
    
    # 初始化客户端
    llm = LLMClient(default_model="qwen-plus")
    print()
    
    # ========== 示例1：使用系统提示词 ==========
    print("-" * 50)
    print("3.1 使用系统提示词设定AI角色")
    print("-" * 50)
    
    # 设置系统提示词，让AI扮演一个Python编程专家
    system_prompt = """你是一个专业的Python编程助手，具备以下特点：
1. 回答简洁明了，重点突出
2. 提供的代码必须是可运行的
3. 会详细解释代码的每一部分
4. 如果有多种解法，会给出最优方案
"""
    
    reply = llm.chat(
        "如何用Python快速排序一个列表？",
        system_prompt=system_prompt,
        temperature=0.1  # 低温度，更精确
    )
    print(f"AI编程助手: {reply}")   
    print()
    
    # ========== 示例2：调整温度参数 ==========
    print("-" * 50)
    print("3.2 调整temperature参数（控制随机性）")
    print("-" * 50)
    
    llm.clear_conversation()
    
    # 低温度（0.1）：回答更确定、一致
    print("🔧 temperature=0.1（精确模式）:")
    reply_low = llm.chat(
        "用一句话描述春天",
        temperature=0.1,
        use_history=False
    )
    print(f"   {reply_low}")
    print()
    
    # 高温度（1.5）：回答更随机、有创意
    print("🔧 temperature=1.5（创意模式）:")
    reply_high = llm.chat(
        "用一句话描述春天",
        temperature=1.5,
        use_history=False
    )
    print(f"   {reply_high}")
    print()
    
    # ========== 示例3：流式输出 ==========
    print("-" * 50)
    print("3.3 流式输出（逐字显示）")
    print("-" * 50)
    
    llm.clear_conversation()
    
    print("AI正在回复（流式输出）: ", end="", flush=True)
    # 使用 chat_stream 方法逐字获取回复
    for chunk in llm.chat_stream(
        "讲一个关于程序员的小笑话",
        temperature=0.9
    ):
        print(chunk, end="", flush=True)
    print("\n")
    
    # ========== 示例4：使用不同的模型 ==========
    print("-" * 50)
    print("3.4 使用不同的模型")
    print("-" * 50)
    
    llm.clear_conversation()
    
    # 尝试使用 qwen-turbo（更快）
    print("🚀 使用 qwen-turbo 模型:")
    reply_turbo = llm.chat(
        "你好，请简短介绍一下自己",
        model="qwen-turbo",
        use_history=False
    )
    print(f"   {reply_turbo}")
    print()
    
    # ========== 示例5：限制输出长度 ==========
    print("-" * 50)
    print("3.5 限制输出长度（max_tokens）")
    print("-" * 50)
    
    llm.clear_conversation()
    
    print("🔤 限制最多50个token:")
    reply_short = llm.chat(
        "介绍一下人工智能",
        max_tokens=50,
        use_history=False
    )
    print(f"   {reply_short}")
    print()


if __name__ == "__main__":
    main()
