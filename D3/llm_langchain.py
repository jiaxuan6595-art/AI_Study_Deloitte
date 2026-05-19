"""
LangChain版LLM客户端
将D1中手动实现的API调用、对话历史管理、重试机制，
迁移到LangChain框架，使用LCEL（LangChain Expression Language）构建Chain。

【D1 → D3 迁移对照】
D1: OpenAI(api_key=..., base_url=...)       → D3: ChatOpenAI(api_key=..., base_url=...)
D1: conversation_history 手动列表管理         → D3: InMemoryChatMessageHistory + session_store
D1: set_system_prompt() 手动插入system消息    → D3: ChatPromptTemplate 中定义system模板
D1: tenacity @retry 装饰器重试               → D3: ChatOpenAI(max_retries=3) 内置重试
D1: chat() / chat_stream() 手动拼接消息      → D3: chain.invoke() / chain.stream() LCEL管道
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# ========== 会话存储 ==========
# 用字典存储不同session的对话历史，key是session_id，value是ChatMessageHistory对象
# 这就是D1中 conversation_history 的LangChain替代方案
# D1中只有一个 conversation_history 列表，D3中可以有多个session，互不干扰
session_store: Dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    """
    根据session_id获取对应的对话历史
    
    【与D1的对比】
    D1中：self.conversation_history 是一个列表，只有一份历史记录
    D3中：session_store 是一个字典，可以同时维护多份独立的对话历史
          比如session1是和用户A的对话，session2是和用户B的对话，互不影响
    
    参数说明：
    session_id : str
        会话ID，用于区分不同的对话
        例如："user_001"、"session_abc" 等
        不同的session_id对应不同的对话历史，互不干扰
    """
    if session_id not in session_store:
        # 如果该session不存在，创建一个新的空历史记录
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def clear_session(session_id: str):
    """
    清空指定session的对话历史
    
    【与D1的对比】
    D1中：llm.clear_conversation() 清空唯一的对话历史
    D3中：clear_session("session1") 清空指定session的历史，其他session不受影响
    """
    if session_id in session_store:
        session_store[session_id].clear()
        print(f"🗑️ 会话 {session_id} 的历史已清空")


def create_llm(
    model: str = "qwen-plus",
    temperature: float = 0.7,
    max_retries: int = 3,
    timeout: int = 30,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    创建LangChain的ChatOpenAI实例（兼容阿里云百炼API）
    
    【与D1的对比】
    D1中：
        self.client = OpenAI(api_key=..., base_url=...)
        然后手动调用 self.client.chat.completions.create(...)
    
    D3中：
        llm = create_llm()
        然后通过LCEL管道 chain = prompt | llm | parser 自动调用
    
    【ChatOpenAI vs OpenAI SDK】
    ChatOpenAI 是 LangChain 对 OpenAI SDK 的封装，提供了：
    - 与LCEL管道无缝衔接（可以用 | 连接）
    - 内置重试机制（max_retries参数）
    - 内置回调系统（可以监控调用过程）
    - 自动处理流式输出
    
    参数说明：
    model : str, 默认值 "qwen-plus"
        模型名称，与D1中的 default_model 相同
        可选：qwen-turbo, qwen-plus, qwen-max 等
    
    temperature : float, 默认值 0.7
        采样温度，与D1中的 temperature 参数含义完全相同
    
    max_retries : int, 默认值 3
        最大重试次数
        【与D1的对比】D1用 tenacity 的 @retry 装饰器实现，D3用 ChatOpenAI 内置参数
    
    timeout : int, 默认值 30
        超时时间（秒），与D1中的 timeout 参数含义相同
    
    api_key : Optional[str], 可选
        API Key，优先级：传入参数 > 环境变量
    
    base_url : Optional[str], 可选
        API基础URL，默认使用阿里云百炼兼容模式地址
    
    **kwargs : dict
        其他参数，如 top_p, max_tokens 等，会直接传给ChatOpenAI
    """
    load_dotenv()
    
    resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not resolved_api_key:
        raise ValueError("API Key未设置！请在.env文件中配置DASHSCOPE_API_KEY")
    
    resolved_base_url = base_url or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        timeout=timeout,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        **kwargs
    )
    
    print(f"✅ LangChain LLM初始化成功！")
    print(f"   - 使用模型: {model}")
    print(f"   - API地址: {resolved_base_url}")
    
    return llm


def create_chain(
    llm: ChatOpenAI,
    system_prompt: str = "你是一个有帮助的AI助手。",
    use_memory: bool = False
):
    """
    用LCEL（LangChain Expression Language）构建Chain
    
    【什么是LCEL？】
    LCEL是LangChain的核心语法，用 | 管道符将组件连接起来：
    
        prompt | llm | parser
    
    就像Linux的管道一样，数据从左到右流过每个组件：
    1. prompt：根据模板和输入生成消息
    2. llm：调用大模型生成回复
    3. parser：解析输出格式
    
    【与D1的对比】
    D1中手动拼接消息、调用API、解析响应：
        messages = self.conversation_history.copy()
        messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(messages=messages, ...)
        reply = response.choices[0].message.content
    
    D3中用LCEL管道一行搞定：
        chain = prompt | llm | parser
        reply = chain.invoke({"input": "你好"})
    
    参数说明：
    llm : ChatOpenAI
        由 create_llm() 创建的LLM实例
    
    system_prompt : str, 默认值 "你是一个有帮助的AI助手。"
        系统提示词
        【与D1的对比】D1中用 set_system_prompt() 方法设置，D3中在创建Chain时直接传入
    
    use_memory : bool, 默认值 False
        是否启用对话历史（Memory）
        True：多轮对话，自动记住上下文
        False：单轮对话，每次都是全新的
    """
    
    # ========== 第一步：定义Prompt模板 ==========
    # ChatPromptTemplate 是 LangChain 的提示词模板类
    # 它可以自动将变量填充到模板中，生成完整的消息列表
    #
    # 【与D1的对比】
    # D1中手动构建消息列表：
    #   messages = []
    #   messages.append({"role": "system", "content": system_prompt})
    #   messages.append({"role": "user", "content": user_message})
    #
    # D3中用模板自动生成：
    #   prompt = ChatPromptTemplate.from_messages([...])
    #   prompt.invoke({"input": "你好"}) → 自动生成消息列表
    
    if use_memory:
        # 带Memory的模板：需要 MessagesPlaceholder 来插入历史消息
        # MessagesPlaceholder 是一个占位符，运行时会自动替换为实际的对话历史
        #
        # 消息结构：
        # [("system", system_prompt),   → 系统消息（固定）
        #  MessagesPlaceholder("history"), → 历史消息（自动填充）
        #  ("human", "{input}")]        → 当前用户输入（变量替换）
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
    else:
        # 不带Memory的模板：只有系统消息和用户输入
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
    
    # ========== 第二步：用LCEL管道构建Chain ==========
    # | 管道符将三个组件连接起来，数据从左到右流动：
    #   prompt → 生成消息列表
    #     → llm → 调用大模型
    #       → StrOutputParser() → 提取纯文本回复
    #
    # StrOutputParser 的作用：
    # LLM返回的是 AIMessage 对象，StrOutputParser 把它转成纯字符串
    # 相当于 D1 中的 response.choices[0].message.content
    chain = prompt | llm | StrOutputParser()
    
    # ========== 第三步：如果需要Memory，包装Chain ==========
    if use_memory:
        # RunnableWithMessageHistory 是 LangChain 的 Memory 包装器
        # 它会自动：
        # 1. 调用前：从 session_store 取出历史消息，填入 MessagesPlaceholder
        # 2. 调用后：把本次的输入和回复保存到 session_store
        #
        # 【与D1的对比】
        # D1中手动管理历史：
        #   self.conversation_history.append({"role": "user", "content": ...})
        #   self.conversation_history.append({"role": "assistant", "content": ...})
        #
        # D3中自动管理：
        #   chain_with_history.invoke({"input": "你好"}, config={"configurable": {"session_id": "s1"}})
        #   → 自动从session_store取历史、拼入消息、保存回复
        
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        print(f"📝 Chain已创建（带Memory，支持多轮对话）")
        return chain_with_history
    else:
        print(f"📝 Chain已创建（无Memory，单轮对话）")
        return chain


def create_custom_chain(
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    output_parser=None
):
    """
    创建自定义Chain，可以自由组合prompt模板和输出解析器
    
    这是 create_chain 的灵活版本，当你需要：
    - 自定义prompt模板（如翻译、摘要等专用模板）
    - 自定义输出解析器（如JsonOutputParser解析JSON）
    
    参数说明：
    llm : ChatOpenAI
        LLM实例
    
    prompt : ChatPromptTemplate
        自定义的prompt模板
    
    output_parser : 可选
        输出解析器，默认使用 StrOutputParser()
        可以传入 JsonOutputParser() 等自定义解析器
    """
    if output_parser is None:
        output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    print(f"📝 自定义Chain已创建")
    return chain
