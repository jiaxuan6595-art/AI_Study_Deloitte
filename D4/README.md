# RAG知识库问答项目

本项目构建完整的RAG（Retrieval-Augmented Generation）流水线，实现知识库问答。使用ChromaDB作为向量数据库，LangChain框架构建检索增强生成链。

## 📁 项目结构

```
D4/
├── rag_pipeline.py              # RAG核心流水线（加载/切分/嵌入/存储/检索/生成）
├── example_basic_rag.py         # 基础RAG示例（完整流水线+有RAG vs 无RAG对比）
├── example_rag_with_memory.py   # 带Memory的RAG（多轮对话+知识库检索）
├── knowledge_base/              # 知识库目录
│   └── ai_knowledge.txt         # 示例知识文档（AI相关）
├── chroma_db/                   # ChromaDB数据目录（运行后自动生成）
├── requirements.txt             # Python依赖包
├── .env                        # 环境变量配置
└── README.md                   # 项目说明文档
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
# 基础RAG流水线
python example_basic_rag.py

# 带Memory的RAG
python example_rag_with_memory.py
```

## 🔄 RAG四步流程

```
用户提问 "什么是机器学习？"
    ↓
① 查询（Query）：将问题转为向量（Embedding）
    ↓
② 检索（Retrieve）：在ChromaDB中找最相似的文档片段
    ↓
③ 增强（Augment）：把检索到的文档片段拼入Prompt
    ↓
④ 生成（Generate）：LLM基于增强后的Prompt生成回答
```

## 📚 核心概念

### 1. 文档加载与切分（Chunking）

```python
# 加载文档
docs = load_documents("./knowledge_base", glob_pattern="**/*.txt")

# 切分文档
chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
```

| 参数 | 说明 | 建议值 |
|------|------|--------|
| chunk_size | 每个片段最大字符数 | 300-1000 |
| chunk_overlap | 相邻片段重叠字符数 | chunk_size的10%-20% |

### 2. 文档嵌入（Embedding）

```python
embeddings = create_embeddings(model="text-embedding-v3")
```

嵌入模型将文本转为向量，语义相近的文本向量也相近：
- "机器学习" 和 "深度学习" → 向量距离近
- "机器学习" 和 "今天天气" → 向量距离远

### 3. 向量存储与检索（ChromaDB）

```python
# 创建向量存储（首次运行）
vector_store = create_vector_store(chunks, embeddings, "./chroma_db")

# 加载已有向量存储（后续运行）
vector_store = load_vector_store(embeddings, "./chroma_db")

# 相似度检索
results = similarity_search(vector_store, "什么是机器学习？", k=3)
```

### 4. RAG Chain（LCEL管道）

```python
rag_chain = create_rag_chain(llm, vector_store, k=3)

# 提问
answer = rag_chain.invoke("什么是机器学习？")
```

RAG Chain的LCEL数据流：

```
{"question": "什么是机器学习？"}
    ↓
RunnableParallel({
    "context": retriever → 检索相关文档片段,
    "question": 透传原始问题
})
    ↓
ChatPromptTemplate → 拼入Prompt模板
    ↓
ChatOpenAI → 生成回答
    ↓
StrOutputParser → 提取纯文本
```

### 5. 带Memory的RAG

```python
rag_chain = create_rag_chain_with_memory(llm, vector_store, k=3)

# 多轮对话
config = {"configurable": {"session_id": "session1"}}
rag_chain.invoke({"question": "什么是NLP？"}, config=config)
rag_chain.invoke({"question": "它有哪些任务？"}, config=config)  # AI知道"它"是NLP
```

## 🔧 自定义知识库

将你的文档放入 `knowledge_base/` 目录，支持以下格式：

| 格式 | 说明 | 需要的包 |
|------|------|---------|
| .txt | 纯文本 | 内置 |
| .md | Markdown | 内置 |
| .pdf | PDF文档 | `pypdf` |

添加新文档后，删除 `chroma_db/` 目录重新运行即可重建索引。

## 📄 许可证

本项目仅供学习使用。
