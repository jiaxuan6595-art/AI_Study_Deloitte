#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式对话程序（LangChain版）

用LangChain的带Memory Chain实现交互式聊天。
与D1的 interactive_chat.py 功能相同，但底层实现完全不同。

【与D1的对比】
D1: llm.chat_stream(user_input)  → 手动管理 conversation_history
D3: chain.stream({"input": ...}, config=...)  → 自动管理 session_store
"""

from llm_langchain import create_llm, create_chain, clear_session


def main():
    print("=" * 60)
    print("💬 交互式对话程序（LangChain版）")
    print("=" * 60)
    print("指令说明:")
    print("  - 直接输入文字与AI对话")
    print("  - 输入 'quit' 或 'exit' 退出程序")
    print("  - 输入 'clear' 清空对话历史")
    print("=" * 60)
    print()
    
    try:
        llm = create_llm(model="qwen-plus", timeout=60)
        chain = create_chain(
            llm,
            system_prompt="你是一个友好的AI助手，善于倾听和交流。",
            use_memory=True
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 使用固定的session_id，保持同一个对话上下文
    session_id = "interactive_chat"
    config = {"configurable": {"session_id": session_id}}
    
    print()
    print("AI: 你好！有什么我可以帮助你的吗？")
    print()
    
    while True:
        try:
            user_input = input("你: ").strip()
            
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break
            
            if user_input.lower() in ["clear", "清空"]:
                clear_session(session_id)
                print("对话历史已清空！\n")
                continue
            
            if not user_input:
                continue
            
            # 使用 chain.stream() 流式输出
            # 相当于 D1 中的 llm.chat_stream(user_input)
            # 但不需要手动管理 conversation_history
            print("AI: ", end="", flush=True)
            for chunk in chain.stream({"input": user_input}, config=config):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print()


if __name__ == "__main__":
    main()
