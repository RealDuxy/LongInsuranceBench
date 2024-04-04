#!/bin/bash

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model chatglm3-6b > chatglm3-6b.out 2>&1

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model chatglm3-6b-32k > chatglm3-6b-32k.out 2>&1

# 运行main.py并传递第一个模型参数，将所有输出重定向到a.out
python pred_vllm.py --model longalign-6b-64k > longalign-6b-64k.out 2>&1

