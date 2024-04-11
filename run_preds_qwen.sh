#!/bin/bash

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model qwen15_0-5b_chat > qwen15_0-5b_chat.out 2>&1

python pred_vllm.py --model qwen15_1-8b_chat > qwen15_1-8b_chat.out 2>&1

python pred_vllm.py --model qwen15_4b_chat > qwen15_4b_chat.out 2>&1

python pred_vllm.py --model qwen15_7b_chat > qwen15_7b_chat.out 2>&1



