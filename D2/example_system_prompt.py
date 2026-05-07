#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例5：System Prompt与用户输入的组合使用

【什么是System Prompt？】
System Prompt是在对话开始前给模型的"角色设定"和"行为准则"。
它告诉模型：你是谁？你应该怎么做？

【System Prompt的作用】
1. 设定AI的身份和角色（如专家、助手、老师等）
2. 规定AI的回答风格和格式
3. 提供背景信息和约束条件

【常用的System Prompt模板】
- 通用助手："你是一个乐于助人的AI助手..."
- 专家模式："你是一位领域专家..."
- 创意写作："你是一位创意作家..."
- 教育导师："你是一位耐心的教育导师..."
"""

from llm_client import LLMClient


def main():
    print("=" * 60)
    print("示例5：System Prompt与用户输入的组合")
    print("=" * 60)
    print()
    
    # ========== 场景1：通用助手角色 ==========
    print("-" * 50)
    print("场景1：通用助手角色")
    print("-" * 50)
    
    # 创建LLM客户端
    llm1 = LLMClient()
    
    # 设置System Prompt
    system_prompt = """你是一个乐于助人的AI助手。
请用自然、友好的语言回答问题。
保持回答简洁明了，不要过于冗长。
如果不确定答案，请诚实地说明。"""
    
    print(f"System Prompt：{system_prompt}")
    print()
    
    # 提问
    question = "什么是Python？"
    print(f"问题：{question}")
    
    reply = llm1.chat(question, system_prompt=system_prompt, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景2：专家角色 ==========
    print("-" * 50)
    print("场景2：专家角色")
    print("-" * 50)
    
    llm2 = LLMClient()
    
    system_prompt = """你是一位资深的Python编程专家。
请提供专业、准确、详细的技术回答。
提供代码示例和具体实现步骤。
解释技术原理和最佳实践。"""
    
    print(f"System Prompt：{system_prompt}")
    print()
    
    question = "什么是Python？"
    print(f"问题：{question}")
    
    reply = llm2.chat(question, system_prompt=system_prompt, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景3：教育导师角色 ==========
    print("-" * 50)
    print("场景3：教育导师角色")
    print("-" * 50)
    
    llm3 = LLMClient()
    
    system_prompt = """你是一位耐心的编程导师。
请用通俗易懂的语言解释复杂概念。
分步骤讲解，确保理解。
鼓励提问，提供练习建议。"""
    
    print(f"System Prompt：{system_prompt}")
    print()
    
    question = "什么是Python？"
    print(f"问题：{question}")
    
    reply = llm3.chat(question, system_prompt=system_prompt, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景4：创意作家角色 ==========
    print("-" * 50)
    print("场景4：创意作家角色")
    print("-" * 50)
    
    llm4 = LLMClient()
    
    system_prompt = """你是一位富有想象力的创意作家。
请用生动、优美的语言写作。
使用恰当的比喻和形象的描述。
保持风格独特，富有感染力。"""
    
    print(f"System Prompt：{system_prompt}")
    print()
    
    question = "描写一下春天"
    print(f"问题：{question}")
    
    reply = llm4.chat(question, system_prompt=system_prompt, temperature=1.2, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景5：自定义规则 ==========
    print("-" * 50)
    print("场景5：自定义规则")
    print("-" * 50)
    
    llm5 = LLMClient()
    
    system_prompt = """你是一个AI助手。
请严格遵守以下规则：
1. 回答不超过50字
2. 使用中文
3. 不要使用专业术语
4. 保持口语化"""
    
    print(f"System Prompt：{system_prompt}")
    print()
    
    question = "什么是人工智能？"
    print(f"问题：{question}")
    
    reply = llm5.chat(question, system_prompt=system_prompt, use_history=False)
    print(f"回答：{reply}")
    print(f"回答长度：{len(reply)}字")
    print()


if __name__ == "__main__":
    main()
