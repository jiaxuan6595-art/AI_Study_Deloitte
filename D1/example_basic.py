#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例1：基础对话
展示最基本的单轮和多轮对话功能
"""

from llm_client import LLMClient


def main():
    # ========== 步骤1：初始化客户端 ==========
    print("=" * 50)
    print("示例1：基础对话")
    print("=" * 50)
    
    # 创建LLM客户端实例
    # 这里会自动从.env文件中读取API Key
    llm = LLMClient(
        default_model="qwen-plus",  # 默认使用 qwen-plus 模型
        timeout=30,                 # 超时时间30秒
        max_retries=3               # 失败时最多重试3次
    )
    
    print()
    
    # ========== 步骤2：简单的单轮对话 ==========
    print("-" * 50)
    print("2.1 简单单轮对话")
    print("-" * 50)
    
    # 发送第一条消息
    reply = llm.chat("你好，请介绍一下你自己", use_history=False)
    print(f"用户: 你好，请介绍一下你自己")
    print(f"AI: {reply}")
    print()
    
    # ========== 步骤3：多轮对话（记住上下文） ==========
    print("-" * 50)
    print("2.2 多轮对话（AI会记住之前的对话）")
    print("-" * 50)
    
    # 告诉AI你的名字
    reply = llm.chat("我叫小明，今年25岁")
    print(f"用户: 我叫小明，今年25岁")
    print(f"AI: {reply}")
    print()
    
    # 问AI刚才说的名字，AI应该能记住
    reply = llm.chat("我刚才说我叫什么名字？今年多大？")
    print(f"用户: 我刚才说我叫什么名字？今年多大？")
    print(f"AI: {reply}")
    print()
    
    # ========== 步骤4：清空对话历史 ==========
    print("-" * 50)
    print("2.3 清空对话历史，开始新对话")
    print("-" * 50)
    
    # 清空历史
    llm.clear_conversation()
    
    # 再问同样的问题，AI应该不记得了
    reply = llm.chat("我叫什么名字？")
    print(f"用户: 我叫什么名字？")
    print(f"AI: {reply}")
    print()


if __name__ == "__main__":
    main()
