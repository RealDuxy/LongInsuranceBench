#!/bin/bash

# 运行main.py并传递第二个模型参数，将所有输出重定向到b.out

#python pred_litellm.py --model qwen15_14b_chat_int4 --dataset product_retrieval_summary > qwen15_14b_chat_int4_litellm_1.out 2>&1 &
#
#python pred_litellm.py --model qwen15_14b_chat_int4 --dataset product_count > qwen15_14b_chat_int4_litellm_2.out 2>&1 &
#
#python pred_litellm.py --model qwen15_14b_chat_int4 --dataset multi_product_qa > qwen15_14b_chat_int4_litellm_3.out 2>&1 &
#
#python pred_litellm.py --model qwen15_14b_chat_int4 --dataset product_retrieval_question > qwen15_14b_chat_int4_litellm_4.out 2>&1 &

nohup python pred_litellm.py --model qwen15_14b_chat_int4 --port 8081 --pred_dir pred_litellm > qwen15_14b_chat_int4_litellm.out 2>&1

nohup python pred_litellm.py --model chatglm3-6b --port 8082 --pred_dir pred_litellm > chatglm3-6b_litellm.out 2>&1

nohup python pred_litellm.py --model qwen15_32b_chat_int4 --port 8081 --pred_dir pred_litellm > qwen15_32b_chat_int4_litellm.out 2>&1

nohup python pred_litellm.py --model qwen15_72b_chat_int4 --port 8081 --pred_dir pred_litellm > qwen15_72b_chat_int4_litellm.out 2>&1

#python pred_litellm.py --model chatglm3-6b --dataset repeat_product > chatglm3-6b_1.out 2>&1 &
#python pred_litellm.py --model chatglm3-6b --dataset deny_multi_product_qa > chatglm3-6b_2.out 2>&1 &
#python pred_litellm.py --model chatglm3-6b --dataset multi_product_qa,product_retrieval_question > chatglm3-6b_3.out 2>&1 &
#python pred_litellm.py --model chatglm3-6b --dataset product_count,product_retrieval_summary > chatglm3-6b_4.out 2>&1 &
