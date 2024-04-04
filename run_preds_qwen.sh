#!/bin/bash
sleep 1
# 运行main.py并传递第二个模型参数，将所有输出重定向到b.out
python pred_vllm.py --model qwen15_14b_chat_int4 --quantize > qwen15_14b_chat_int4.out 2>&1

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model qwen15_4b_chat > qwen15_4b_chat.out 2>&1

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model qwen15_7b_chat > qwen15_7b_chat.out 2>&1

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model qwen15_14b_chat > qwen15_14b_chat.out 2>&1



