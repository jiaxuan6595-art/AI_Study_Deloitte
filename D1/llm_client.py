import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMClient:
    """
    LLM API客户端封装类
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "qwen-plus",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        初始化LLM客户端
    
        参数说明：
        api_key : 阿里云百炼的API Key
        base_url : 默认使用：https://dashscope.aliyuncs.com/compatible-mode/v1
        default_model : 默认值 "qwen-plus"
            默认使用的模型名称
            可选模型列表（参考 https://help.aliyun.com/zh/model-studio/getting-started/models）：
            - qwen-turbo: 通义千问-Turbo，快速响应，适合简单任务
            - qwen-plus: 通义千问-Plus，平衡性能与速度
            - qwen-max: 通义千问-Max，最强能力，适合复杂任务
            - qwen-coder-plus: 通义千问-Coder-Plus，专门用于代码生成
        timeout : 默认值 30
            API请求的超时时间（秒）
            - 如果API响应时间超过这个值，会抛出超时异常
        max_retries :  默认值 3
            请求失败时的最大重试次数
            - 当遇到网络错误、API限流等问题时会自动重试
            - 重试间隔采用指数退避策略（1秒、2秒、4秒...）
        """
        
        # 加载.env文件中的环境变量
        load_dotenv()
        
        # 设置API Key
        # 优先级：传入的参数 > 环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        # 检查API Key是否存在
        if not self.api_key:
            raise ValueError(
                "API Key未设置！\n"
            )
        
        # 设置API基础URL
        # 优先级：传入的参数 > 环境变量 > 默认值
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 设置默认模型
        self.default_model = default_model
        
        # 设置超时时间
        self.timeout = timeout
        
        # 设置最大重试次数
        self.max_retries = max_retries
        
        # 初始化OpenAI客户端
        # 阿里云百炼提供了兼容OpenAI API的接口，所以可以直接使用OpenAI SDK
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 对话历史记录，用于多轮对话
        # 格式：[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.conversation_history: List[Dict[str, str]] = []
        
        # 打印初始化成功信息
        print(f"✅ LLM客户端初始化成功！")
        print(f"   - 使用模型: {self.default_model}")
        print(f"   - API地址: {self.base_url}")
        print(f"   - 超时时间: {self.timeout}秒")
        print(f"   - 最大重试: {self.max_retries}次")

    def clear_conversation(self):
        """
        清空对话历史
        当你想开始一个新的对话时，调用这个方法可以清空之前的对话记录
        """
        self.conversation_history = []
        print("🗑️ 对话历史已清空")

    def set_system_prompt(self, system_prompt: str):
        """
        设置系统提示词（system prompt）
        参数说明：
        ----------
        system_prompt : str
            系统提示词，用于设定AI助手的角色、行为准则等
            例如："你是一个专业的Python编程助手，擅长解决代码问题"
        系统提示词的作用：
        - 设定AI的身份和角色
        - 规定AI的回答风格和格式
        - 提供背景信息和约束条件
        """
        # 先检查是否已经有system消息
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            # 如果有，替换它
            self.conversation_history[0]["content"] = system_prompt
        else:
            # 如果没有，插入到最前面
            self.conversation_history.insert(0, {
                "role": "system",
                "content": system_prompt
            })
        print(f"📝 系统提示词已设置: {system_prompt[:50]}...")

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = True,
        **kwargs
    ) -> Any:
        """
        内部方法：调用API并自动重试
        这个方法封装了API调用逻辑，并添加了重试机制
        当遇到网络错误、API限流等问题时会自动重试
        参数说明：
        ----------
        messages : List[Dict[str, str]]
            对话消息列表，格式如下：
            [
                {"role": "system", "content": "系统提示词"},
                {"role": "user", "content": "用户消息"},
                {"role": "assistant", "content": "AI回复"},
                ...
            ]
            role的可选值：
            - system: 系统消息，设定AI的角色和行为
            - user: 用户消息，用户的提问
            - assistant: 助手消息，AI的回复
        
        model : Optional[str], 可选
            要使用的模型名称
            - 如果不提供，使用初始化时设置的 default_model
        
        temperature : float, 可选, 默认值 0.1
            采样温度，控制输出的随机性
            - 范围：0.0 到 2.0
            - 值越高，输出越随机、有创意（适合写诗、创作）
            - 值越低，输出越确定、一致（适合问答、编程）
            - 建议值：
              * 0.0-0.3：精确回答，事实性任务
              * 0.4-0.7：平衡，一般对话
              * 0.8-1.5：创意写作、头脑风暴
        
        top_p : float, 可选, 默认值 0.9
            核采样参数，控制输出的多样性
            - 范围：0.0 到 1.0
            - 模型会考虑累积概率达到 top_p 的token
            - 例如：top_p=0.9 表示只考虑累计概率占90%的最可能的token
            - 通常与 temperature 二选一使用，不同时调整两者
        
        max_tokens : Optional[int], 可选
            最大生成token数
            - 限制AI回复的长度
            - 不同模型有不同的最大限制
            - 例如：qwen-plus 最多支持 8k token
        
        stream : bool, 可选, 默认值 True
            是否使用流式输出
            - True: 逐字返回，适合实时显示
            - False: 一次性返回完整结果
        
        **kwargs : dict, 可选
            其他API参数，会直接传递给API
            常用的额外参数：
            - presence_penalty: float, 范围 -2.0 到 2.0，减少重复内容
            - frequency_penalty: float, 范围 -2.0 到 2.0，减少频繁出现的词
            - stop: List[str], 停止词，遇到这些词时停止生成
        
        返回值：
        ----------
        Any
            API的响应对象
        """
        
        # 使用装饰器定义重试逻辑
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True
        )
        def _make_request():
            """内部函数：实际执行API请求"""
            return self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream,
                timeout=self.timeout,
                **kwargs
            )
        
        try:
            # 执行请求，自动重试
            return _make_request()
        except Exception as e:
            # 所有重试都失败后，抛出异常
            raise Exception(f"API调用失败（已重试{self.max_retries}次）: {str(e)}")

    def chat(
        self,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        stream: bool = True,        
        **kwargs
    ) -> str:
        """
        发送消息给LLM并获取回复，支持单轮和多轮对话

        参数说明：
        ----------
        user_message : str
            用户的消息内容，即你要问的问题或说的话
        
        model : Optional[str], 可选
            要使用的模型名称，不设置则使用默认模型
        
        temperature : float, 可选, 默认值 0.1
            采样温度，详见 _call_api_with_retry 方法的说明
        
        top_p : float, 可选, 默认值 0.9
            核采样参数，详见 _call_api_with_retry 方法的说明
        
        max_tokens : Optional[int], 可选
            最大生成token数，详见 _call_api_with_retry 方法的说明
        
        system_prompt : Optional[str], 可选
            系统提示词
            - 如果提供，会设置或更新系统提示词
            - 如果不提供，使用已有的系统提示词
        
        use_history : bool, 可选, 默认值 True
            是否使用对话历史
            - True: 多轮对话，会记住之前的对话内容
            - False: 单轮对话，每次都是全新的对话
        
        stream : bool, 可选, 默认值 True
            是否使用流式输出
            - True: 逐字打印回复
            - False: 一次性返回完整回复
        
        **kwargs : dict, 可选
            其他API参数，详见 _call_api_with_retry 方法的说明
        
        返回值：
        ----------
        str
            AI的回复内容
        
        使用示例：
        ----------
        # 简单对话
        reply = llm.chat("你好！")
        
        # 多轮对话
        reply1 = llm.chat("我叫小明")
        reply2 = llm.chat("我叫什么名字？")  # AI会记得你叫小明
        
        # 单轮对话（不记住历史）
        reply = llm.chat("你好", use_history=False)
        
        # 自定义参数
        reply = llm.chat(
            "写一首关于春天的诗",
            temperature=1.2,  # 高温度，更有创意
            system_prompt="你是一个专业的诗人"
        )
        """
        
        # 如果提供了系统提示词，先设置
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        
        # 构建当前对话的消息列表
        if use_history:
            # 使用历史记录
            messages = self.conversation_history.copy()
        else:
            # 不使用历史记录，但保留system消息
            messages = []
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                messages.append(self.conversation_history[0])
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 调用API
        response = self._call_api_with_retry(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        if stream:
            # 流式输出
            assistant_reply = ""
            print("🤖 AI回复: ", end="", flush=True)
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    assistant_reply += content
                    print(content, end="", flush=True)
            print()
        else:
            # 非流式输出，获取完整回复
            assistant_reply = response.choices[0].message.content
        
        # 如果使用历史记录，保存对话
        if use_history:
            self.conversation_history = messages
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_reply
            })
        
        return assistant_reply

    def chat_stream(
        self,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        **kwargs
    ):
        """
        流式对话，逐字生成回复（生成器版本）
        
        这个方法返回一个生成器，可以逐字获取AI的回复
        适合在UI中实时显示打字效果
        
        参数说明：
        ----------
        与 chat 方法相同
        
        返回值：
        ----------
        Generator[str, None, None]
            生成器，每次产生一个字符串片段
        
        使用示例：
        ----------
        for chunk in llm.chat_stream("写一个故事"):
            print(chunk, end="", flush=True)
        """
        
        # 如果提供了系统提示词，先设置
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        
        # 构建当前对话的消息列表
        if use_history:
            messages = self.conversation_history.copy()
        else:
            messages = []
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                messages.append(self.conversation_history[0])
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 调用API（流式）
        response = self._call_api_with_retry(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        # 收集完整回复
        assistant_reply = ""
        
        # 逐字返回
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                assistant_reply += content
                yield content
        
        # 保存对话历史
        if use_history:
            self.conversation_history = messages
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_reply
            })
