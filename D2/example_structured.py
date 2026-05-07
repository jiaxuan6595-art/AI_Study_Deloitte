#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例4：结构化输出

【什么是结构化输出？】
通过明确的格式要求，引导模型输出结构化的数据（如JSON、XML等）。

【适用场景】
- 需要将AI输出作为数据输入其他程序
- 需要解析和提取特定字段
- 需要标准化输出格式的场景

【优缺点】
✅ 优点：输出格式明确，易于程序解析
✅ 优点：可以定义字段约束和类型
❌ 缺点：需要明确定义输出格式
❌ 缺点：模型可能偶尔不遵守格式要求
"""

import json
from llm_client import LLMClient


def main():
    print("=" * 60)
    print("示例4：结构化输出")
    print("=" * 60)
    print()
    
    # 创建LLM客户端
    llm = LLMClient()
    
    # ========== 场景1：JSON格式输出 ==========
    print("-" * 50)
    print("场景1：JSON格式输出")
    print("-" * 50)
    
    # 直接在问题中指定JSON格式
    question = """请分析以下文章并输出JSON格式结果：

文章内容：人工智能正在改变我们的生活。从智能家居到自动驾驶，AI技术已经渗透到各个领域。机器学习算法能够从大量数据中学习模式，帮助我们做出更明智的决策。

请按照以下JSON格式输出：
{
    "title": "文章标题",
    "category": "文章类别",
    "summary": "文章摘要",
    "keywords": ["关键词1", "关键词2", "关键词3"],
    "confidence": 置信度(0-1)
}

输出：
"""
    
    print("文章内容：人工智能正在改变我们的生活...")
    print()
    print("要求输出格式：JSON")
    print()
    
    reply = llm.chat(question, temperature=0.1, use_history=False)
    print(f"输出结果：\n{reply}")
    print()
    
    # 解析JSON
    try:
        data = json.loads(reply)
        print("解析结果：")
        print(f"  标题: {data.get('title')}")
        print(f"  类别: {data.get('category')}")
        print(f"  摘要: {data.get('summary')}")
        print(f"  关键词: {data.get('keywords')}")
        print(f"  置信度: {data.get('confidence')}")
    except:
        print("JSON解析失败")
    print()
    
    # ========== 场景2：XML格式输出 ==========
    print("-" * 50)
    print("场景2：XML格式输出")
    print("-" * 50)
    
    question = """请将以下信息转换为XML格式：

产品名称：智能手机X1
价格：2999元
品牌：科技公司A
评分：4.5/5

请输出XML格式：
<product>
  <name>产品名称</name>
  <price>价格</price>
  <brand>品牌</brand>
  <rating>评分</rating>
</product>

输出：
"""
    
    print("输入信息：产品名称：智能手机X1，价格：2999元，品牌：科技公司A，评分：4.5/5")
    print()
    
    reply = llm.chat(question, temperature=0.1, use_history=False)
    print(f"输出结果：\n{reply}")
    print()
    
    # ========== 场景3：CSV格式输出 ==========
    print("-" * 50)
    print("场景3：CSV格式输出")
    print("-" * 50)
    
    question = """请列出以下城市信息，输出为CSV格式：

北京：人口2154万，面积16410平方公里
上海：人口2428万，面积6340平方公里
广州：人口1530万，面积7434平方公里

请输出CSV格式（第一行为表头）：
城市,人口,面积
北京,2154万,16410平方公里
...

输出：
"""
    
    print("输入信息：北京、上海、广州的人口和面积")
    print()
    
    reply = llm.chat(question, temperature=0.1, use_history=False)
    print(f"输出结果：\n{reply}")
    print()


if __name__ == "__main__":
    main()
