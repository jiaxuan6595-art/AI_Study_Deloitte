#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例4：结构化输出Chain（JSON解析）

展示如何用LangChain的 JsonOutputParser 让LLM输出结构化的JSON数据。

【什么是结构化输出？】
让LLM按照预定义的格式（如JSON Schema）输出，而不是自由文本。
这样程序就可以直接解析和使用LLM的输出。

【核心组件】
1. Pydantic模型 — 用Python类定义输出结构（字段名、类型、描述）
2. JsonOutputParser — 根据Pydantic模型生成格式说明，并解析LLM输出为字典

【与D2的对比】
D2中在prompt中手写JSON模板：
    question = "请输出JSON格式：{name: ..., age: ...}"
    reply = llm.chat(question)
    data = json.loads(reply)  # 手动解析，可能失败

D3中用JsonOutputParser自动处理：
    parser = JsonOutputParser(pydantic_object=PersonInfo)
    # parser自动在prompt中注入格式说明
    # parser自动解析LLM输出为Python字典
    data = chain.invoke({"input": "..."})  # 直接得到字典
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from llm_langchain import create_llm, create_custom_chain


# ========== 定义Pydantic模型 ==========
# Pydantic模型用来描述期望的输出结构
# 每个字段的 Field(description=...) 会告诉LLM这个字段应该填什么

class PersonInfo(BaseModel):
    """人物信息提取模型"""
    name: str = Field(description="人物姓名")
    age: int = Field(description="人物年龄")
    occupation: str = Field(description="职业")
    hobbies: list = Field(description="爱好列表")


class SentimentResult(BaseModel):
    """情感分析结果模型"""
    sentiment: str = Field(description="情感倾向：正面/负面/中性")
    confidence: float = Field(description="置信度，0到1之间的数值")
    reason: str = Field(description="判断理由的简要说明")


class ProductAnalysis(BaseModel):
    """产品分析模型"""
    product_name: str = Field(description="产品名称")
    pros: list = Field(description="产品优点列表")
    cons: list = Field(description="产品缺点列表")
    rating: int = Field(description="评分，1到5分")
    summary: str = Field(description="一句话总结")


def main():
    print("=" * 60)
    print("示例4：结构化输出Chain（JSON解析）")
    print("=" * 60)
    print()
    
    llm = create_llm(model="qwen-plus", temperature=0.1)
    print()
    
    # ========== 场景1：人物信息提取 ==========
    print("-" * 50)
    print("场景1：人物信息提取")
    print("-" * 50)
    
    # 创建JsonOutputParser，传入Pydantic模型
    # parser会做两件事：
    # 1. 生成格式说明（get_format_instructions()），告诉LLM应该输出什么格式的JSON
    # 2. 解析LLM的JSON输出，转为Python字典
    person_parser = JsonOutputParser(pydantic_object=PersonInfo)
    
    # 在prompt模板中加入 {format_instructions} 占位符
    # parser.get_format_instructions() 会生成类似这样的说明：
    #   "请按照以下JSON格式输出：
    #    {"name": "人物姓名", "age": 年龄, "occupation": "职业", "hobbies": ["爱好1", "爱好2"]}"
    person_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个信息提取专家，从文本中提取人物信息。"),
        ("human", "从以下文本中提取人物信息：\n\n{input}\n\n{format_instructions}"),
    ])
    
    # 构建Chain，注意传入自定义的parser
    person_chain = create_custom_chain(llm, person_prompt, person_parser)
    
    text = "李明今年28岁，是一名软件工程师，平时喜欢打篮球、看电影和编程。"
    print(f"输入文本: {text}")
    
    # invoke时传入 format_instructions 参数
    # parser.get_format_instructions() 自动生成格式说明
    result = person_chain.invoke({
        "input": text,
        "format_instructions": person_parser.get_format_instructions()
    })
    
    # result 已经是Python字典了，不需要手动 json.loads()
    print(f"提取结果（自动解析为字典）:")
    print(f"  姓名: {result['name']}")
    print(f"  年龄: {result['age']}")
    print(f"  职业: {result['occupation']}")
    print(f"  爱好: {result['hobbies']}")
    print()
    
    # ========== 场景2：情感分析 ==========
    print("-" * 50)
    print("场景2：情感分析")
    print("-" * 50)
    
    sentiment_parser = JsonOutputParser(pydantic_object=SentimentResult)
    
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感分析专家，分析文本的情感倾向。"),
        ("human", "分析以下文本的情感：\n\n{input}\n\n{format_instructions}"),
    ])
    
    sentiment_chain = create_custom_chain(llm, sentiment_prompt, sentiment_parser)
    
    texts = [
        "这家餐厅的菜品非常好吃，服务也很周到，下次还会来！",
        "产品质量太差了，用了两天就坏了，非常失望。",
        "今天天气一般，不好不坏。"
    ]
    
    for text in texts:
        result = sentiment_chain.invoke({
            "input": text,
            "format_instructions": sentiment_parser.get_format_instructions()
        })
        print(f"文本: {text}")
        print(f"  情感: {result['sentiment']} | 置信度: {result['confidence']} | 理由: {result['reason']}")
    print()
    
    # ========== 场景3：产品分析 ==========
    print("-" * 50)
    print("场景3：产品分析")
    print("-" * 50)
    
    product_parser = JsonOutputParser(pydantic_object=ProductAnalysis)
    
    product_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个产品分析专家，客观分析产品的优缺点。"),
        ("human", "分析以下产品评价：\n\n{input}\n\n{format_instructions}"),
    ])
    
    product_chain = create_custom_chain(llm, product_prompt, product_parser)
    
    review = """
    我买了这款笔记本电脑用了一个月了。优点是屏幕显示效果很好，键盘手感也不错，
    续航能达到8小时。缺点是风扇噪音有点大，而且只有8GB内存，运行大型软件时会卡。
    总体来说性价比还行，给4分吧。
    """
    
    result = product_chain.invoke({
        "input": review.strip(),
        "format_instructions": product_parser.get_format_instructions()
    })
    
    print(f"产品名称: {result['product_name']}")
    print(f"优点: {', '.join(result['pros'])}")
    print(f"缺点: {', '.join(result['cons'])}")
    print(f"评分: {result['rating']}/5")
    print(f"总结: {result['summary']}")
    print()


if __name__ == "__main__":
    main()
