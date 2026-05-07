#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式对话程序
可以在终端中与AI进行持续对话
输入 'quit' 或 'exit' 退出
输入 'clear' 清空对话历史
"""

from llm_client import LLMClient


def main():
    print("=" * 60)
    print("💬 交互式对话程序")
    print("=" * 60)
    print("指令说明:")
    print("  - 直接输入文字与AI对话")
    print("  - 输入 'quit' 或 'exit' 退出程序")
    print("  - 输入 'clear' 清空对话历史")
    print("=" * 60)
    print()
    
    # 初始化客户端
    try:
        llm = LLMClient(
            default_model="qwen-plus",
            timeout=60
        )
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    print()
    print("AI: 你好！有什么我可以帮助你的吗？")
    print()
    
    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ["quit", "exit", "退出"]:
                print("再见！")
                break
            
            # 检查清空命令
            if user_input.lower() in ["clear", "清空"]:
                llm.clear_conversation()
                print()
                continue
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 发送消息并获取回复（流式输出）
            print("AI: ", end="", flush=True)
            for chunk in llm.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
            
        except KeyboardInterrupt:
            # Ctrl+C 中断
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print()


if __name__ == "__main__":
    main()
