#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例3：多种Chain类型

展示如何用LCEL构建不同类型的Chain，每种Chain有专用的prompt模板。

【什么是Chain类型？】
Chain就是"链条"，把多个处理步骤串起来。
不同类型的Chain = 不同的prompt模板 + 相同的LCEL管道语法

【本示例展示的Chain类型】
1. 翻译链 — 将文本从一种语言翻译为另一种语言
2. 摘要链 — 将长文本总结为简短摘要
3. 组合链 — 将多个Chain串联起来（先翻译再摘要）
4. 串行链 — 用LCEL的 | 管道符将多个Chain串成一条流水线
5. 并行链 — 用RunnableParallel让多个Chain同时处理同一输入

【LCEL的统一语法】
无论什么类型的Chain，构建方式都是一样的：
    chain = prompt | llm | parser
区别只在于 prompt 模板的内容不同。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from llm_langchain import create_custom_chain, create_llm


def main():
    print("=" * 60)
    print("示例3：多种Chain类型")
    print("=" * 60)
    print()
    
    llm = create_llm(model="qwen-plus", temperature=0.3)
    print()
    
    # ========== Chain类型1：翻译链 ==========
    print("-" * 50)
    print("Chain类型1：翻译链")
    print("-" * 50)
    
    # 翻译链的prompt模板
    # 包含3个变量：source_lang（源语言）、target_lang（目标语言）、text（待翻译文本）
    #
    # 【与D2的对比】
    # D2中直接把完整prompt写在字符串里：
    #   llm.chat("请将以下中文翻译为英文：\n你好世界")
    #
    # D3中用模板变量，可以复用同一个Chain翻译不同语言：
    #   translate_chain.invoke({"source_lang": "中文", "target_lang": "英文", "text": "你好世界"})
    #   translate_chain.invoke({"source_lang": "英文", "target_lang": "日文", "text": "Hello"})
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的翻译官，翻译准确、流畅、自然。"),
        ("human", "请将以下{source_lang}文本翻译为{target_lang}：\n\n{text}"),
    ])
    
    # 用LCEL构建翻译链
    translate_chain = create_custom_chain(llm, translate_prompt)
    
    # 中文 → 英文
    reply = translate_chain.invoke({
        "source_lang": "中文",
        "target_lang": "英文",
        "text": "人工智能正在改变我们的生活方式，从智能手机到自动驾驶，AI无处不在。"
    })
    print(f"中文 → 英文:")
    print(f"原文: 人工智能正在改变我们的生活方式，从智能手机到自动驾驶，AI无处不在。")
    print(f"译文: {reply}")
    print()
    
    # 中文 → 日文
    reply = translate_chain.invoke({
        "source_lang": "中文",
        "target_lang": "日文",
        "text": "今天天气真好，适合出去散步。"
    })
    print(f"中文 → 日文:")
    print(f"原文: 今天天气真好，适合出去散步。")
    print(f"译文: {reply}")
    print()
    
    # ========== Chain类型2：摘要链 ==========
    print("-" * 50)
    print("Chain类型2：摘要链")
    print("-" * 50)
    
    # 摘要链的prompt模板
    # 包含2个变量：word_count（摘要字数）、content（待摘要内容）
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的文本摘要助手，善于提取关键信息。"),
        ("human", "请用{word_count}字左右总结以下内容：\n\n{content}"),
    ])
    
    summarize_chain = create_custom_chain(llm, summarize_prompt)
    
    long_text = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，
    旨在开发能够模拟人类智能的系统和技术。AI的研究包括机器学习、深度学习、
    自然语言处理、计算机视觉等多个领域。近年来，随着大数据和计算能力的提升，
    AI技术取得了突破性进展，在医疗、金融、教育、交通等行业得到广泛应用。
    然而，AI的发展也带来了伦理、隐私、就业等方面的挑战，需要社会各界共同关注和应对。
    """
    
    reply = summarize_chain.invoke({
        "word_count": "50",
        "content": long_text.strip()
    })
    print(f"原文: {long_text.strip()[:80]}...")
    print(f"摘要: {reply}")
    print()
    
    # ========== Chain类型3：组合链（先翻译再摘要） ==========
    print("-" * 50)
    print("Chain类型3：组合链 — 先翻译再摘要")
    print("-" * 50)
    
    # 组合链的核心思路：把一个Chain的输出作为另一个Chain的输入
    # 这里用Python代码手动串联，展示Chain之间的数据流转
    #
    # 数据流：
    # 原文 → 翻译链 → 英文译文 → 摘要链 → 英文摘要
    
    chinese_text = "深度学习是机器学习的一个子领域，通过多层神经网络来学习数据的表示。它在图像识别、语音识别和自然语言处理等任务中取得了巨大成功。"
    
    # 第一步：翻译
    translated = translate_chain.invoke({
        "source_lang": "中文",
        "target_lang": "英文",
        "text": chinese_text
    })
    print(f"第一步 — 翻译结果:")
    print(f"  {translated}")
    print()
    
    # 第二步：对翻译结果做摘要
    summary = summarize_chain.invoke({
        "word_count": "30",
        "content": translated
    })
    print(f"第二步 — 摘要结果:")
    print(f"  {summary}")
    print()
    
    # ========== Chain类型4：角色扮演链 ==========
    print("-" * 50)
    print("Chain类型4：角色扮演链")
    print("-" * 50)
    
    # 角色扮演链：通过System Prompt设定AI的角色
    # 包含2个变量：role（角色描述）、question（用户问题）
    #
    # 【与D2的对比】
    # D2中在 system_prompt 参数中写死角色：
    #   llm.chat("问题", system_prompt="你是编程专家")
    #
    # D3中用模板变量，同一个Chain可以扮演不同角色：
    #   role_chain.invoke({"role": "编程专家", "question": "..."})
    #   role_chain.invoke({"role": "营养师", "question": "..."})
    role_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。请用专业但易懂的方式回答问题。"),
        ("human", "{question}"),
    ])
    
    role_chain = create_custom_chain(llm, role_prompt)
    
    # 扮演编程专家
    reply = role_chain.invoke({
        "role": "资深Python编程专家",
        "question": "Python中列表和元组有什么区别？"
    })
    print(f"角色: 资深Python编程专家")
    print(f"问题: Python中列表和元组有什么区别？")
    print(f"回答: {reply}")
    print()
    
    # 扮演营养师
    reply = role_chain.invoke({
        "role": "专业营养师",
        "question": "每天应该喝多少水？"
    })
    print(f"角色: 专业营养师")
    print(f"问题: 每天应该喝多少水？")
    print(f"回答: {reply}")
    print()

    # ========== Chain类型5：串行链 — 用 | 管道符串联 ==========
    print("-" * 50)
    print("Chain类型5：串行链 — 翻译→摘要一条龙")
    print("-" * 50)
    
    # 串行链的核心思路：用LCEL的 | 管道符把多个Chain串起来
    # Chain A 的输出自动变成 Chain B 的输入，像流水线一样
    #
    # 【与Chain类型3（手动组合）的对比】
    # Chain类型3是手动分两步调用的：
    #   translated = translate_chain.invoke({...})
    #   summary = summarize_chain.invoke({"content": translated, ...})
    #
    # 串行链是用 | 管道符把他们连成一条Chain，一步完成：
    #   pipeline = translate_chain | map_output | summarize_chain
    #   result = pipeline.invoke({...})  # 一步得到最终结果
    #
    # 数据流：
    # 输入 {"source_lang", "target_lang", "text"}
    #   → translate_chain 输出 字符串（翻译结果）
    #     → lambda 把字符串映射为 {"word_count": "30", "content": 翻译结果}
    #       → summarize_chain 输出 字符串（摘要结果）
    
    # translate_chain 输出的是纯文本字符串
    # summarize_chain 需要接收 {"word_count": ..., "content": ...}
    # 所以中间加一个 lambda 函数做数据格式转换
    pipeline_chain = (
        translate_chain                                          # 第一步：翻译，输出字符串
        | (lambda text: {"word_count": "30", "content": text})  # 第二步：格式转换，把字符串包成字典
        | summarize_chain                                        # 第三步：摘要，输出字符串
    )
    
    chinese_text = "机器学习是人工智能的核心技术之一，它让计算机能够从数据中学习规律和模式，而不需要显式编程。深度学习作为机器学习的一个分支，使用多层神经网络处理复杂任务，在图像识别、语音识别和自然语言处理等领域取得了突破性进展。"
    
    # 串行链：一步调用，内部自动执行 翻译 → 格式转换 → 摘要
    final_result = pipeline_chain.invoke({
        "source_lang": "中文",
        "target_lang": "英文",
        "text": chinese_text
    })
    
    print(f"原文: {chinese_text[:60]}...")
    print(f"串行链最终结果（英文摘要）:")
    print(f"  {final_result}")
    print()
    
    # ========== Chain类型6：并行链 — 同时执行多个任务 ==========
    print("-" * 50)
    print("Chain类型6：并行链 — 同一文本多角度分析")
    print("-" * 50)
    
    # 并行链的核心思路：用 RunnableParallel 同时运行多个Chain
    # 所有Chain接收相同的输入，各自独立处理，最后把结果合并为一个字典
    #
    # 就像"一个人提问，多个专家同时回答"：
    #   输入 {"text": "..."}
    #         ├──→ 情感分析链 → "正面"（耗时2秒）
    #         ├──→ 关键词提取链 → "AI, 技术"（耗时2秒）
    #         └──→ 语言检测链 → "中文"（耗时2秒）
    #   总耗时 ≈ 2秒（并行执行），而不是 6秒（串行执行）
    #
    # RunnableParallel 的参数：
    #   每个 key=value 就是一个子任务
    #   key: 结果字典中的字段名
    #   value: 要执行的Chain
    
    # 创建3个专用的分析链
    # 1. 情感分析链（Parallel Chain中的子任务1）
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个情感分析专家，分析文本的情感倾向。"),
        ("human", "请分析以下文本的情感倾向（回答格式：正面/负面/中性）：\n\n{text}"),
    ])
    sentiment_chain = create_custom_chain(llm, sentiment_prompt)
    
    # 2. 关键词提取链（Parallel Chain中的子任务2）
    keyword_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个关键词提取专家，提取文本中的关键信息。"),
        ("human", "请提取以下文本中的3-5个关键词（用逗号分隔）：\n\n{text}"),
    ])
    keyword_chain = create_custom_chain(llm, keyword_prompt)
    
    # 3. 语言检测链（Parallel Chain中的子任务3）
    language_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个语言检测专家，准确识别文本使用的语言。"),
        ("human", "请识别以下文本使用的语言（只输出语言名称）：\n\n{text}"),
    ])
    language_chain = create_custom_chain(llm, language_prompt)
    
    # 用 RunnableParallel 创建并行链
    # 传入一个字典，每个 key=value 对定义一个子任务
    # 运行时会同时启动所有子任务，等待全部完成后返回结果字典
    parallel_chain = RunnableParallel({
        "sentiment": sentiment_chain,     # key="sentiment" → 情感分析结果
        "keywords": keyword_chain,        # key="keywords"  → 关键词结果
        "language": language_chain,       # key="language"  → 语言检测结果
    })
    
    text_to_analyze = "这款智能手机的屏幕显示效果非常出色，色彩鲜艳且亮度充足，但电池续航时间较短，需要频繁充电。"
    
    print(f"待分析文本: {text_to_analyze[:60]}...")
    print()
    
    # 并行调用：一次 invoke，三个子任务同时执行
    # 返回的 result 是一个字典，包含三个 key
    result = parallel_chain.invoke({"text": text_to_analyze})
    
    print(f"并行分析结果:")
    print(f"  情感分析: {result['sentiment']}")
    print(f"  关键词: {result['keywords']}")
    print(f"  语言检测: {result['language']}")
    print()
    
    # ========== Bonus：串行+并行混合使用 ==========
    print("-" * 50)
    print("Bonus：串行+并行混合 — 翻译后多角度分析")
    print("-" * 50)
    
    # 高级用法：串行链和并行链可以任意组合
    # 先翻译，再对翻译结果做并行分析
    #
    # 数据流：
    #   {"text": "中文原文", ...}
    #     → translate_chain（翻译为英文）
    #       → RunnableParallel（并行分析英文文本）
    #           ├──→ sentiment_chain → 情感
    #           ├──→ keyword_chain   → 关键词
    #           └──→ language_chain  → 语言
    #
    # 用 | 管道的串联能力 + RunnableParallel 的并联能力
    hybrid_chain = (
        translate_chain                                    # 第一步：翻译
        | (lambda text: {"text": text})                   # 第二步：格式转换
        | RunnableParallel({                              # 第三步：并行分析
            "sentiment": sentiment_chain,
            "keywords": keyword_chain,
            "language": language_chain,
        })
    )
    
    hybrid_input = {
        "source_lang": "中文",
        "target_lang": "英文",
        "text": "这家餐厅的服务态度非常好，服务员热情周到，但菜品价格偏高，性价比一般。"
    }
    
    hybrid_result = hybrid_chain.invoke(hybrid_input)
    
    print(f"原文: {hybrid_input['text']}")
    print(f"混合链结果（先翻译成英文，再并行分析）:")
    print(f"  情感分析: {hybrid_result['sentiment']}")
    print(f"  关键词: {hybrid_result['keywords']}")
    print(f"  语言检测: {hybrid_result['language']}")
    print()


if __name__ == "__main__":
    main()
