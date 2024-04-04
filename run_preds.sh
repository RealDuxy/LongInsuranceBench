#!/bin/bash

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model chatglm3-6b --max_samples 10 > a.out 2>&1

# 运行main.py并传递第二个模型参数，将所有输出重定向到b.out
python pred_vllm.py --model qwen15_14b_chat_int4 --max_samples 10 > b.out 2>&1

