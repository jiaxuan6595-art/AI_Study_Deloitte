"""
示例1：基础LCEL Chain

展示LangChain最核心的LCEL管道语法：prompt | llm | parser

【LCEL管道数据流】
输入 {"input": "你好"}
    → ChatPromptTemplate 生成消息列表 [{"role": "system", ...}, {"role": "human", ...}]
        → ChatOpenAI 调用大模型，返回 AIMessage 对象
            → StrOutputParser 提取纯文本字符串

【与D1的对比】
D1: llm.chat("你好", use_history=False)
    → 手动拼接 messages → 调用API → 手动提取 response.choices[0].message.content

D3: chain.invoke({"input": "你好"})
    → prompt模板自动生成messages → LLM自动调用 → parser自动提取文本
"""

from llm_langchain import create_llm, create_chain


def main():
    print("=" * 60)
    print("示例1：基础LCEL Chain")
    print("=" * 60)
    print()
    
    # ========== 步骤1：创建LLM和Chain ==========
    # create_llm() 相当于 D1 中的 LLMClient()
    # create_chain() 用LCEL管道构建 chain = prompt | llm | parser
    llm = create_llm(model="qwen-plus", temperature=0.7)
    chain = create_chain(llm, use_memory=False)
    print()
    
    # ========== 步骤2：单轮对话（invoke） ==========
    print("-" * 50)
    print("2.1 单轮对话 — chain.invoke()")
    print("-" * 50)
    
    # chain.invoke() 是同步调用，等待完整结果返回
    # 相当于 D1 中的 llm.chat("你好", use_history=False)
    #
    # {"input": "你好"} 中的 "input" 对应 ChatPromptTemplate 中的 "{input}" 变量
    # 模板会自动把 "你好" 填充到 human 消息中
    reply = chain.invoke({"input": "你好，请介绍一下你自己"})
    print(f"用户: 你好，请介绍一下你自己")
    print(f"AI: {reply}")
    print()
    
    # ========== 步骤3：流式输出（stream） ==========
    print("-" * 50)
    print("2.2 流式输出 — chain.stream()")
    print("-" * 50)
    
    # chain.stream() 返回一个生成器，逐字输出
    # 相当于 D1 中的 llm.chat_stream("...")
    #
    # 【与D1的对比】
    # D1: for chunk in llm.chat_stream("讲个笑话"): print(chunk, end="")
    # D3: for chunk in chain.stream({"input": "讲个笑话"}): print(chunk, end="")
    print("用户: 讲一个关于程序员的小笑话")
    print("AI: ", end="", flush=True)
    for chunk in chain.stream({"input": "讲一个关于程序员的小笑话"}):
        print(chunk, end="", flush=True)
    print("\n")
    
    # ========== 步骤4：自定义System Prompt ==========
    print("-" * 50)
    print("2.3 自定义System Prompt")
    print("-" * 50)
    
    # 创建一个带自定义System Prompt的Chain
    # 相当于 D1 中的 llm.chat("...", system_prompt="你是一个编程专家")
    expert_chain = create_chain(
        llm,
        system_prompt="你是一个专业的Python编程助手，回答简洁明了，提供可运行的代码。",
        use_memory=False
    )
    
    reply = expert_chain.invoke({"input": "如何用Python读取CSV文件？"})
    print(f"用户: 如何用Python读取CSV文件？")
    print(f"AI编程助手: {reply}")
    print()
    
    # ========== 步骤5：批量调用（batch） ==========
    print("-" * 50)
    print("2.4 批量调用 — chain.batch()")
    print("-" * 50)
    
    # chain.batch() 可以同时处理多个输入，返回结果列表
    # 这是 D1 中没有的功能，LangChain 自动处理并发
    #
    # 传入一个列表，每个元素是一个输入字典
    # 返回一个列表，每个元素是对应的回复字符串
    questions = [
        {"input": "1+1等于几？只回答数字"},
        {"input": "中国的首都是哪里？只回答城市名"},
        {"input": "水的化学式是什么？只回答化学式"},
    ]
    
    replies = chain.batch(questions)
    for q, r in zip(questions, replies):
        print(f"问: {q['input']}")
        print(f"答: {r}")
        print()


if __name__ == "__main__":
    main()
