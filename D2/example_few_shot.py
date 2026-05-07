#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例2：少样本学习 (Few-shot Learning)

【什么是少样本学习？】
在用户问题前提供几个示例（示范），让模型学习期望的格式和回答方式。

【适用场景】
- 需要特定格式输出的任务（如分类、提取）
- 需要遵循特定规则的任务
- 模型在零样本情况下表现不佳的复杂任务

【优缺点】
✅ 优点：可以引导模型输出特定格式，提高准确性
✅ 优点：少量示例即可显著提升效果
❌ 缺点：需要人工准备示例
❌ 缺点：增加了Prompt长度，消耗更多token
"""

from llm_client import LLMClient


def main():
    print("=" * 60)
    print("示例2：少样本学习")
    print("=" * 60)
    print()
    
    # 创建LLM客户端
    llm = LLMClient()
    
    # ========== 场景1：文本分类 ==========
    print("-" * 50)
    print("场景1：文本情感分类")
    print("-" * 50)
    
    # 准备示例（告诉模型什么样的输入对应什么样的输出）
    examples = """
示例1：
输入：这家餐厅的服务太棒了！
输出：正面

示例2：
输入：电影很无聊，浪费时间。
输出：负面

示例3：
输入：会议将在下午3点举行。
输出：中性

问题：
输入：产品质量一般，价格偏高。
输出：
"""
    
    print("示例：")
    print("输入：这家餐厅的服务太棒了！ -> 输出：正面")
    print("输入：电影很无聊，浪费时间。 -> 输出：负面")
    print("输入：会议将在下午3点举行。 -> 输出：中性")
    print()
    print(f"待分类文本：产品质量一般，价格偏高。")
    
    reply = llm.chat(examples, temperature=0.1, use_history=False)
    print(f"分类结果：{reply}")
    print()
    
    # ========== 场景2：实体提取 ==========
    print("-" * 50)
    print("场景2：实体提取")
    print("-" * 50)
    
    # 准备示例
    examples = """
示例1：
文本：苹果公司总部位于美国加州。
人物：无
地点：美国加州
组织：苹果公司

示例2：
文本：马云创立了阿里巴巴。
人物：马云
地点：无
组织：阿里巴巴

问题：
文本：张勇在杭州阿里园区发表演讲。
人物：
地点：
组织：
"""
    
    print("示例：")
    print("文本：苹果公司总部位于美国加州。 -> 人物：无，地点：美国加州，组织：苹果公司")
    print("文本：马云创立了阿里巴巴。 -> 人物：马云，地点：无，组织：阿里巴巴")
    print()
    print(f"待提取文本：张勇在杭州阿里园区发表演讲。")
    
    reply = llm.chat(examples, temperature=0.1, use_history=False)
    print(f"提取结果：\n{reply}")
    print()
    
    # ========== 场景3：格式转换 ==========
    print("-" * 50)
    print("场景3：格式转换（中文姓名转英文）")
    print("-" * 50)
    
    examples = """
示例1：
中文：张三
英文：Zhang San

示例2：
中文：李四
英文：Li Si

示例3：
中文：王小明
英文：Wang Xiaoming

问题：
中文：陈大伟
英文：
"""
    
    print("示例：")
    print("中文：张三 -> 英文：Zhang San")
    print("中文：李四 -> 英文：Li Si")
    print("中文：王小明 -> 英文：Wang Xiaoming")
    print()
    print(f"待转换：中文：陈大伟")
    
    reply = llm.chat(examples, temperature=0.1, use_history=False)
    print(f"转换结果：英文：{reply}")
    print()


if __name__ == "__main__":
    main()
