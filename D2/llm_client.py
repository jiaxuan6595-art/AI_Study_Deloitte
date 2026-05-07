import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "qwen-plus",
        timeout: int = 30,
        max_retries: int = 3
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        if not self.api_key:
            raise ValueError("API Key未设置！")
        
        self.base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        self.conversation_history: List[Dict[str, str]] = []
        
        print(f"✅ LLM客户端初始化成功！")
        print(f"   - 使用模型: {self.default_model}")

    def clear_conversation(self):
        self.conversation_history = []

    def set_system_prompt(self, system_prompt: str):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = system_prompt
        else:
            self.conversation_history.insert(0, {
                "role": "system",
                "content": system_prompt
            })

    def _call_api_with_retry(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((Exception,)),
            reraise=True
        )
        def _make_request():
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
            return _make_request()
        except Exception as e:
            raise Exception(f"API调用失败（已重试{self.max_retries}次）: {str(e)}")

    def chat(
        self,
        user_message: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        use_history: bool = True,
        stream: bool = False,
        **kwargs
    ) -> str:
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        
        if use_history:
            messages = self.conversation_history.copy()
        else:
            messages = []
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                messages.append(self.conversation_history[0])
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
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
            assistant_reply = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    assistant_reply += content
        else:
            assistant_reply = response.choices[0].message.content
        
        if use_history:
            self.conversation_history = messages
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_reply
            })
        
        return assistant_reply
