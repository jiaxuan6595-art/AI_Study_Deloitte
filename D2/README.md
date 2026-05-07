# Prompt工程实验项目

本项目专注于Prompt工程研究，提供了多种Prompt策略的示例代码，直接在代码中写Prompt，简单易懂。

## 📁 项目结构

```
D2/
├── llm_client.py              # 核心LLM客户端封装类
├── example_zero_shot.py       # 零样本学习示例
├── example_few_shot.py        # 少样本学习示例
├── example_cot.py             # 思维链示例
├── example_few_shot_cot.py    # 少样本+思维链组合示例
├── example_structured.py      # 结构化输出示例
├── example_system_prompt.py   # System Prompt示例
├── requirements.txt           # Python依赖包
├── .env                      # 环境变量配置
└── README.md                  # 项目说明文档
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
# 零样本学习
python example_zero_shot.py

# 少样本学习
python example_few_shot.py

# 思维链
python example_cot.py

# 少样本+思维链组合
python example_few_shot_cot.py

# 结构化输出
python example_structured.py

# System Prompt
python example_system_prompt.py
```

## 📚 Prompt策略说明

### 1. 零样本学习 (Zero-shot)

直接向模型提问，不提供任何示例。

```python
from llm_client import LLMClient

llm = LLMClient()
reply = llm.chat("什么是人工智能？", use_history=False)
```

### 2. 少样本学习 (Few-shot)

在用户问题前提供几个示例，让模型学习期望的格式。

```python
examples = """
示例1：
输入：这家餐厅很棒！
输出：正面

问题：
输入：产品质量一般。
输出：
"""
reply = llm.chat(examples, temperature=0.1, use_history=False)
```

### 3. 思维链 (Chain of Thought)

引导模型逐步展示推理过程。

```python
question = """让我一步步思考：

问题：水池注水问题...

思考过程：
"""
reply = llm.chat(question, temperature=0.1, use_history=False)
```

### 4. 少样本+思维链组合 (Few-shot + CoT)

结合少样本学习和思维链，提供示例让模型学习推理模式。

```python
few_shot_cot = """
示例1：
问题：甲管4小时注满，乙管6小时注满，同时开几小时注满？
思考过程：...
答案：2.4小时

问题：你的问题...
思考过程：
"""
reply = llm.chat(few_shot_cot, temperature=0.1, use_history=False)
```

### 5. 结构化输出

指定输出格式（JSON、XML、CSV等）。

```python
question = """请输出JSON格式：
{
    "title": "...",
    "summary": "..."
}
"""
reply = llm.chat(question, temperature=0.1, use_history=False)
```

### 6. System Prompt

设定AI的角色和行为准则。

```python
system_prompt = """你是一位编程专家。
请提供详细的技术回答。"""
reply = llm.chat("什么是Python？", system_prompt=system_prompt)
```

## 💡 Prompt工程最佳实践

1. **明确任务目标**：清楚你想要模型做什么
2. **提供清晰示例**：少样本学习通常能显著提升效果
3. **控制输出格式**：使用结构化输出便于程序处理
4. **设定合适角色**：根据任务选择合适的System Prompt
5. **调整温度参数**：数学问题用低温度(0.1)，创意写作用高温度(1.2)
6. **使用思维链**：对于复杂推理问题，引导模型逐步思考

## 📄 许可证

本项目仅供学习使用。
