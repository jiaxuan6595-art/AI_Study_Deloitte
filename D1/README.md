# 阿里云百炼LLM API调用项目

这是一个基于阿里云百炼平台的大语言模型API调用封装项目，提供了简单易用的接口，支持多轮对话、错误处理和超时重试等功能。

## 📁 项目结构

```
D1/
├── llm_client.py          # 核心LLM客户端封装类（详细注释）
├── example_basic.py       # 基础用法示例
├── example_advanced.py    # 高级用法示例
├── interactive_chat.py    # 交互式对话程序
├── requirements.txt       # Python依赖包
├── .env.example           # 环境变量模板
└── README.md              # 项目说明文档
```

## 🚀 快速开始

### 1. 安装依赖

在项目目录下打开终端，运行：

```bash
pip install -r requirements.txt
```

### 2. 配置API Key

1. 复制 `.env.example` 文件，重命名为 `.env`
2. 打开 `.env` 文件，将 `sk-xxx` 替换为你的阿里云百炼API Key

```env
DASHSCOPE_API_KEY=你的API密钥在这里
```

**获取API Key：** 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)

### 3. 运行示例

#### 示例1：基础对话
```bash
python example_basic.py
```

#### 示例2：高级用法
```bash
python example_advanced.py
```

#### 交互式对话
```bash
python interactive_chat.py
```

## 📚 主要功能

### 1. LLMClient 类

核心封装类，提供以下功能：

- **API调用封装**：简化API调用流程
- **多轮对话**：自动维护对话历史
- **错误处理**：完善的异常处理机制
- **超时重试**：自动重试失败的请求
- **灵活配置**：支持多种参数调整

### 2. 主要参数说明

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | str | None | API Key，不填则从环境变量读取 |
| `base_url` | str | 阿里云百炼地址 | API基础地址 |
| `default_model` | str | "qwen-plus" | 默认模型 |
| `timeout` | int | 30 | 请求超时时间（秒） |
| `max_retries` | int | 3 | 最大重试次数 |

#### chat() 方法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_message` | str | 必填 | 用户输入的消息 |
| `model` | str | None | 使用的模型，不填则用默认值 |
| `temperature` | float | 0.7 | 采样温度（0.0-2.0） |
| `top_p` | float | 0.9 | 核采样参数（0.0-1.0） |
| `max_tokens` | int | None | 最大生成token数 |
| `system_prompt` | str | None | 系统提示词 |
| `use_history` | bool | True | 是否使用对话历史 |
| `stream` | bool | False | 是否流式输出 |

### 3. 常用模型

| 模型名称 | 说明 |
|----------|------|
| `qwen-turbo` | 通义千问-Turbo，快速响应 |
| `qwen-plus` | 通义千问-Plus，平衡性能 |
| `qwen-max` | 通义千问-Max，最强能力 |
| `qwen-coder-plus` | 通义千问-Coder-Plus，代码专用 |

更多模型请参考：[阿里云百炼模型列表](https://help.aliyun.com/zh/model-studio/getting-started/models)

## 💡 使用示例

### 基础用法

```python
from llm_client import LLMClient

# 初始化客户端
llm = LLMClient()

# 简单对话
reply = llm.chat("你好！")
print(reply)

# 多轮对话
llm.chat("我叫小明")
reply = llm.chat("我叫什么名字？")  # AI会记得你叫小明
```

### 设置系统提示词

```python
# 让AI扮演编程助手
system_prompt = "你是一个专业的Python编程助手"
reply = llm.chat(
    "如何写快速排序？",
    system_prompt=system_prompt,
    temperature=0.3
)
```

### 流式输出

```python
# 逐字显示回复
for chunk in llm.chat_stream("讲个故事"):
    print(chunk, end="", flush=True)
```

## ⚙️ 参数调优建议

### temperature（温度）

- **0.0 - 0.3**：精确回答，适合事实性问题、编程
- **0.4 - 0.7**：平衡，适合一般对话
- **0.8 - 1.5**：创意模式，适合写作、头脑风暴

### top_p（核采样）

- 通常与 temperature 二选一调整
- 值越小，输出越保守
- 值越大，输出越多样

## 📞 常见问题

**Q: API Key如何获取？**
A: 访问 https://bailian.console.aliyun.com/ 注册并获取

**Q: 为什么调用失败？**
A: 请检查：
1. API Key是否正确配置
2. 网络连接是否正常
3. 账户是否有余额

**Q: 如何清空对话历史？**
A: 调用 `llm.clear_conversation()` 方法

## 📄 许可证

本项目仅供学习使用。
