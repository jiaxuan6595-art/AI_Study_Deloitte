#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例3：思维链 (Chain of Thought)

【什么是思维链？】
引导模型在回答之前，先一步步地展示推理过程。
通过"让我思考一下"、"首先"、"其次"等提示词，
促使模型产生中间推理步骤，从而提高复杂问题的准确性。

【适用场景】
- 数学问题
- 逻辑推理问题
- 需要多步骤思考的复杂问题
- 需要解释答案由来的场景

【优缺点】
✅ 优点：提高复杂推理问题的准确性
✅ 优点：答案可解释，便于理解推理过程
❌ 缺点：输出更长，消耗更多token
❌ 缺点：推理过程可能包含错误步骤

【常用的思维链提示词】
- "让我一步步思考这个问题："
- "首先，我需要..."
- "让我分析一下："
- "请给出详细的推理过程："
"""

from llm_client import LLMClient


def main():
    print("=" * 60)
    print("示例3：思维链 (Chain of Thought)")
    print("=" * 60)
    print()
    
    # 创建LLM客户端
    llm = LLMClient()
    
    # ========== 场景1：数学问题 ==========
    print("-" * 50)
    print("场景1：数学问题")
    print("-" * 50)
    
    # 在问题中加入思维链引导词
    question = """让我一步步思考这个问题：

问题：一个水池有两个进水管和一个出水管。
单开甲管6小时注满，单开乙管8小时注满，单开丙管12小时放完。
三管同时打开，几小时注满水池？

思考过程：
"""
    
    print("问题：一个水池有两个进水管和一个出水管。")
    print("      单开甲管6小时注满，单开乙管8小时注满，单开丙管12小时放完。")
    print("      三管同时打开，几小时注满水池？")
    print()
    
    reply = llm.chat(question, temperature=0.1, use_history=False)
    print(f"思考过程+答案：\n{reply}")
    print()
    
    # ========== 场景2：逻辑推理 ==========
    print("-" * 50)
    print("场景2：逻辑推理")
    print("-" * 50)
    
    question = """请一步步分析：

问题：甲、乙、丙三人中有一人是老师，一人是医生，一人是工程师。
已知：
1. 甲比医生年龄大
2. 乙和工程师不同岁
3. 工程师比丙年龄小

请问：谁是老师？谁是医生？谁是工程师？

分析过程：
"""
    
    print("问题：甲、乙、丙三人中有一人是老师，一人是医生，一人是工程师。")
    print("已知：")
    print("1. 甲比医生年龄大")
    print("2. 乙和工程师不同岁")
    print("3. 工程师比丙年龄小")
    print("请问：谁是老师？谁是医生？谁是工程师？")
    print()
    
    reply = llm.chat(question, temperature=0.1, use_history=False)
    print(f"分析过程+答案：\n{reply}")
    print()
    
    # ========== 场景3：复杂问答 ==========
    print("-" * 50)
    print("场景3：复杂问答")
    print("-" * 50)
    
    question = """让我仔细分析一下：

问题：为什么夏天白天比冬天长？

思考步骤：
"""
    
    print("问题：为什么夏天白天比冬天长？")
    print()
    
    reply = llm.chat(question, temperature=0.7, use_history=False)
    print(f"分析过程+答案：\n{reply}")
    print()


if __name__ == "__main__":
    main()
