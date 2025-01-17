import os
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from litellm import LLM


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="chatglm3-6b", )
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--s', help='model size in B')
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    parser.add_argument('--max_samples', type=int, required=False)
    # parser.add_argument('--checkpoint', type=str, help="checkpoint_path")
    parser.add_argument('--quantize', action='store_true', help="Debug mode")
    parser.add_argument('--dataset',type=str, required=False)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--port', type=int, required=True)

    return parser.parse_args(args)

def load_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return tokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name or "longalign":
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def build_input(tokenizer, **kwargs):
    prompt = prompt_format.format(**kwargs)
    # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if "chatglm3" in model_name or "longalign":
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[
            0]
    # if "qwen15" in model_name:
    #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids

    if len(tokenized_prompt) > max_source_length:
        # print(f"当前数据大于{max_length}, 过长，需要进行截断")
        # print(f"prompt length: {len(prompt)}")
        # print(f"tokenized_prompt length: {len(tokenized_prompt)}")
        half = int(max_source_length / 2) - 2
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized_prompt[-half:], skip_special_tokens=True)
        # print(f"截断后：")
        # print(f"prompt length: {len(prompt)}")
        # tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # print(f"tokenized_prompt length: {len(tokenized_prompt)}")

    # prompt = build_chat(tokenizer, prompt, model_name)

    # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
    #                    "repobench-p"]:  # chat models are better off without build prompts on these tasks
    #     prompt = build_chat(tokenizer, prompt, model_name)
    # if "chatglm3" in model_name:
    #     if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
    #         input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    #     else:
    #         input = prompt.to(device)
    # else:
    #     input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

    # context_length = input.input_ids.shape[-1]

    # prompt = tokenizer.batch_decode(prompt.input_ids, skip_special_tokens=True)
    return [prompt]

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response.strip()

    # dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    print(args)

    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    if args.debug:
        config_dir = "config"
    else:
        config_dir = "config_tsr"

    if args.max_samples:
        test_split = f"test[:{args.max_samples}]"
    else:
        test_split = "test"

    model2path = json.load(open(f"{config_dir}/model2path.json", "r"))
    model2maxlen = json.load(open(f"{config_dir}/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]

    dataset_list = ["deny_multi_product_qa", "product_retrieval_question", "product_retrieval_summary", "product_count",
                    "multi_product_qa", "repeat_product"]
    if args.e:
        if args.dataset:
            datasets = [x.strip() for x in args.dataset.split(",")]
        else:
            datasets = ["repeat_product", "deny_multi_product_qa", "product_retrieval_question", "product_retrieval_summary", "product_count",
                    "multi_product_qa"]
    else:
        if args.dataset:
            datasets = [x.strip() for x in args.dataset.split(",")]
        else:
            datasets = ["repeat_product", "deny_multi_product_qa", "product_retrieval_question", "product_retrieval_summary", "product_count",
                    "multi_product_qa"]

    print(datasets)
        # datasets = ["repeat_product"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open(f"{config_dir}/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(f"{config_dir}/dataset2maxlen.json", "r"))
    # predict on each dataset

    pred_dir = args.pred_dir

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    model = LLM(port=int(args.port))

    data_script = "LongInsuranceBench/LongInsuranceBench.py"
    for dataset in datasets:
        print(f"处理数据集：{dataset}")
        data = load_dataset(data_script, dataset, split=test_split)
        if not os.path.exists(f"{pred_dir}/{model_name}"):
            os.makedirs(f"{pred_dir}/{model_name}")
        out_path = f"{pred_dir}/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        print(f"word_size: {world_size}")

        data_split = 2
        data_subsets = [data_all[i: i+data_split]for i in range(0, len(data_all), data_split)]
        # data_subsets = [data_all[i::data_split] for i in range(data_split)]
        max_new_tokens = max(min(dataset2maxlen[dataset], max_length), 64)
        print(f"max_new_tokens: {max_new_tokens}")
        max_source_length = max(max_length - max_new_tokens, max_length-64)
        print(f"max_source_length: {max_source_length}")
        # sampling_params = SamplingParams(max_tokens=max_new_tokens, use_beam_search=False, temperature=0.0)
        tokenizer = load_tokenizer(model2path[model_name], model_name)
        # for json_obj in tqdm(data):
        #     prompt = build_input(tokenizer, **json_obj)
        #     output = model.chat(prompt[0], max_new_tokens=max_new_tokens)
        #     pred = output
        #     if pred == '':
        #         print(output)
        #     pred = post_process(pred, model_name)
        #     with open(out_path, "a", encoding="utf-8") as f:
        #         json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
        #                    "length": json_obj["length"]}, f, ensure_ascii=False)
        #         f.write('\n')

        with open(out_path, "a", encoding="utf-8") as f:
            for json_obj_list in tqdm(data_subsets):
                prompts = [build_input(tokenizer, **json_obj)[0] for json_obj in json_obj_list]
                # prompt = build_input(tokenizer, **json_obj)
                outputs = model.parallel_chat(prompts, max_new_tokens=max_new_tokens)
                for output, json_obj in zip(outputs, json_obj_list):
                    pred = post_process(output, model_name)
                    json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                               "length": json_obj["length"]}, f, ensure_ascii=False)
                    f.write('\n')
            # get_pred(0,1,data_all,max_length,max_gen,prompt_format,dataset,device,model_name,model2path,out_path)
