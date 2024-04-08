#!/bin/bash

# 运行main.py并传递第二个模型参数，将所有输出重定向到b.out

python pred_litellm.py --model qwen15_14b_chat_int4 --dataset product_retrieval_summary > qwen15_14b_chat_int4_litellm_1.out 2>&1 &

python pred_litellm.py --model qwen15_14b_chat_int4 --dataset product_count > qwen15_14b_chat_int4_litellm_2.out 2>&1 &

python pred_litellm.py --model qwen15_14b_chat_int4 --dataset multi_product_qa > qwen15_14b_chat_int4_litellm_3.out 2>&1 &

python pred_litellm.py --model qwen15_14b_chat_int4 --dataset product_retrieval_question > qwen15_14b_chat_int4_litellm_4.out 2>&1 &

#python pred_litellm.py --model chatglm3-6b > chatglm3-6b_litellm.out 2>&1


