# LangChain框架项目

本项目将D1的LLM API调用逻辑迁移至LangChain框架，使用LCEL（LangChain Expression Language）构建Chain，实现对话历史管理（Memory）、结构化输出、串行链和并行链。

## 📁 项目结构

```
D3/
├── llm_langchain.py              # LangChain版LLM客户端（迁移自D1）
├── example_basic_chain.py        # 基础LCEL Chain示例（invoke/stream/batch）
├── example_memory_chain.py       # 带Memory的对话Chain（多轮对话+session隔离）
├── example_multi_chain.py        # 多种Chain类型（翻译/摘要/角色扮演/串行/并行/混合）
├── example_structured_chain.py   # 结构化输出Chain（Pydantic+JsonOutputParser）
├── interactive_chat.py           # 交互式对话（LangChain版）
├── requirements.txt              # Python依赖包
├── .env                         # 环境变量配置
└── README.md                    # 项目说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API Key

打开 `.env` 文件，填入你的阿里云百炼API Key：

```env
DASHSCOPE_API_KEY=你的API密钥
```

### 3. 运行示例

```bash
# 基础LCEL Chain（invoke/stream/batch）
python example_basic_chain.py

# 带Memory的对话Chain
python example_memory_chain.py

# 多种Chain类型
python example_multi_chain.py

# 结构化输出Chain
python example_structured_chain.py

# 交互式对话
python interactive_chat.py
```

## ⚠️ 常见问题

### 导入报错

如果遇到 `ImportError: cannot import name 'BaseChatSession'`，原因是langchain_core版本不同。
已修复：直接删除了未使用的 `BaseChatSession` 导入。

## 🔄 D1 → D3 迁移对照

| D1（手动实现） | D3（LangChain实现） |
|---|---|
| `OpenAI` 客户端 | `ChatOpenAI`（LangChain封装） |
| `conversation_history` 列表 | `InMemoryChatMessageHistory` + `session_store` |
| `set_system_prompt()` | `ChatPromptTemplate` 中的system模板 |
| 手动拼接messages | LCEL管道 `prompt \| llm \| parser` |
| `tenacity` 重试 | `ChatOpenAI` 内置 `max_retries` |
| `chat()` / `chat_stream()` | `chain.invoke()` / `chain.stream()` |
| 单一对话历史 | 多session隔离（不同session_id互不影响） |

## 📚 核心概念

### 1. LCEL（LangChain Expression Language）

LCEL是LangChain的核心语法，用 `|` 管道符将组件连接起来：

```python
chain = prompt | llm | parser
```

数据从左到右流过每个组件：prompt生成消息 → llm调用模型 → parser解析输出

LCEL支持三种调用方式：

| 方法 | 说明 | 对应D1 |
|------|------|--------|
| `chain.invoke()` | 同步调用，等待完整结果 | `llm.chat()` |
| `chain.stream()` | 流式输出，逐字返回 | `llm.chat_stream()` |
| `chain.batch()` | 批量调用，同时处理多个输入 | D1不支持 |

### 2. Chain类型

| Chain类型 | 说明 | 核心组件 |
|-----------|------|---------|
| 基础对话链 | prompt \| llm \| parser | ChatPromptTemplate + ChatOpenAI + StrOutputParser |
| 翻译链 | 多变量prompt模板（source_lang/target_lang/text） | 自定义ChatPromptTemplate |
| 摘要链 | 控制输出字数（word_count/content） | 自定义ChatPromptTemplate |
| 角色扮演链 | System Prompt使用动态变量（role） | 变量化System Prompt |
| **串行链** | 用 `\|` 管道符串联多个Chain，前一个的输出是后一个的输入 | LCEL管道 + lambda格式转换 |
| **并行链** | 用 `RunnableParallel` 同时运行多个Chain，共享同一输入 | RunnableParallel |
| **混合链** | 串行+并行组合使用（如：先翻译然后并行分析） | RunnableParallel + LCEL管道 |

#### 串行链示例

```python
# 翻译 → 格式转换 → 摘要，用 | 串成一条流水线
pipeline_chain = (
    translate_chain                                          # 第一步：翻译，输出字符串
    | (lambda text: {"word_count": "30", "content": text})  # 第二步：格式转换
    | summarize_chain                                        # 第三步：摘要，输出字符串
)

# 一步调用，内部自动执行三个步骤
final_result = pipeline_chain.invoke({
    "source_lang": "中文",
    "target_lang": "英文",
    "text": "原文内容..."
})
```

#### 并行链示例

```python
# 多个Chain同时处理同一段文本
parallel_chain = RunnableParallel({
    "sentiment": sentiment_chain,    # 情感分析
    "keywords": keyword_chain,       # 关键词提取
    "language": language_chain,      # 语言检测
})

# 一次调用，三个子任务同时执行
result = parallel_chain.invoke({"text": "待分析文本..."})
# result = {"sentiment": "正面", "keywords": "AI,技术", "language": "中文"}
```

#### 混合链示例

```python
# 先串行（翻译）再并行（多角度分析）
hybrid_chain = (
    translate_chain
    | (lambda text: {"text": text})
    | RunnableParallel({
        "sentiment": sentiment_chain,
        "keywords": keyword_chain,
    })
)
```

### 3. Memory（对话历史管理）

```python
# 创建带Memory的Chain
chain = create_chain(llm, use_memory=True)

# 通过session_id区分不同对话
config = {"configurable": {"session_id": "user_001"}}

# 自动保存历史
chain.invoke({"input": "我叫小明"}, config=config)
chain.invoke({"input": "我叫什么名字？"}, config=config)  # AI还记得！

# 不同session互不干扰
config2 = {"configurable": {"session_id": "user_002"}}
chain.invoke({"input": "我叫什么名字？"}, config=config2)  # AI不知道
```

#### 多session隔离

| 功能 | D1 | D3 |
|------|----|----|
| 存储方式 | 1个 `conversation_history` 列表 | N个 `ChatMessageHistory` 对象的字典 |
| 会话隔离 | 不支持，只有一个对话 | `session_id` 区分不同用户/场景 |
| 清空历史 | `clear_conversation()` 清空全部 | `clear_session("id")` 只清空指定会话 |

### 4. 结构化输出

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# 定义输出结构
class PersonInfo(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")

# 创建解析器
parser = JsonOutputParser(pydantic_object=PersonInfo)

# 构建Chain（parser自动注入格式说明，自动解析JSON）
chain = prompt | llm | parser
result = chain.invoke({"input": "李明28岁是一名软件工程师"})
# result = {"name": "李明", "age": 28, "occupation": "软件工程师"}
```

## 📄 许可证

本项目仅供学习使用。