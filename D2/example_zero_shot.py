#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例1：零样本学习 (Zero-shot Learning)

【什么是零样本学习？】
直接向模型提问，不提供任何示例。
模型依靠自身的预训练知识来回答问题。

【适用场景】
- 简单的事实性问题
- 日常对话
- 不需要特定格式的问题

【优缺点】
✅ 优点：简单直接，不需要准备示例
❌ 缺点：复杂问题可能回答不准确，格式难以控制
"""

from llm_client import LLMClient


def main():
    print("=" * 60)
    print("示例1：零样本学习")
    print("=" * 60)
    print()
    
    # 创建LLM客户端
    llm = LLMClient()
    
    # ========== 场景1：简单事实问答 ==========
    print("-" * 50)
    print("场景1：简单事实问答")
    print("-" * 50)
    
    # 直接提问，不需要任何示例
    question = "什么是人工智能？"
    print(f"问题：{question}")
    
    # 调用chat方法，直接传入问题
    reply = llm.chat(question, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景2：日常对话 ==========
    print("-" * 50)
    print("场景2：日常对话")
    print("-" * 50)
    
    question = "今天天气怎么样？"
    print(f"问题：{question}")
    
    reply = llm.chat(question, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景3：翻译任务 ==========
    print("-" * 50)
    print("场景3：翻译任务")
    print("-" * 50)
    
    question = "请将'Hello World'翻译成中文"
    print(f"问题：{question}")
    
    reply = llm.chat(question, use_history=False)
    print(f"回答：{reply}")
    print()
    
    # ========== 场景4：创意写作 ==========
    print("-" * 50)
    print("场景4：创意写作")
    print("-" * 50)
    
    question = "写一首关于春天的诗"
    print(f"问题：{question}")
    
    reply = llm.chat(question, temperature=1.2, use_history=False)
    print(f"回答：{reply}")
    print()


if __name__ == "__main__":
    main()
