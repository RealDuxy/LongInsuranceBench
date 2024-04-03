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
from vllm import LLM, SamplingParams


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="qwen15_7b_chat", )
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--s', help='model size in B')
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    # parser.add_argument('--checkpoint', type=str, help="checkpoint_path")
    parser.add_argument('--quantize', action='store_true', help="Debug mode")

    return parser.parse_args(args)

def load_tokenizer(path, model_name):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name or "qwen15" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return tokenizer


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
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
    if "chatglm3" in model_name:
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[
            0]
    # if "qwen15" in model_name:
    #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids
    if len(tokenized_prompt) > max_length:
        print(f"当前数据大于{max_length}, 过长，需要进行截断")
        print(f"prompt length: {len(prompt)}")
        print(f"tokenized_prompt length: {len(tokenized_prompt)}")
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
            tokenized_prompt[-half:], skip_special_tokens=True)
        print(f"截断后：")
        print(f"prompt length: {len(prompt)}")
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[
            0]
        print(f"tokenized_prompt length: {len(tokenized_prompt)}")

    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc",
                       "repobench-p"]:  # chat models are better off without build prompts on these tasks
        prompt = build_chat(tokenizer, prompt, model_name)
    if "chatglm3" in model_name:
        if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            input = prompt.to(device)
    else:
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
    context_length = input.input_ids.shape[-1]
    return input, context_length

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

    model2path = json.load(open("config_tsr/model2path.json", "r"))
    model2maxlen = json.load(open("config_tsr/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["product_retrieval_summary", "product_retrieval_question", "product_count", "multi_product_qa",
                    "deny_multi_product_qa", "repeat_product"]
    else:
        datasets = ["deny_multi_product_qa", "product_retrieval_question", "product_retrieval_summary", "product_count",
                    "multi_product_qa", "repeat_product"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config_tsr/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config_tsr/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    # print(args.checkpoint)

    if args.quantize:
        model = LLM(model=model2path[model_name],
                    trust_remote_code=True,
                    quantization="GPTQ",
                    max_model_len=max_length)
    else:
        model = LLM(model=model2path[model_name], tensor_parallel_size=world_size, trust_remote_code=True,
                    max_model_len=max_length)

    data_script = "LongInsuranceBench/LongInsuranceBench.py"
    for dataset in datasets:
        if args.e:
            data = load_dataset(data_script, f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset(data_script, dataset, split='test[:4]')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        print(f"word_size: {world_size}")

        if args.debug:
            world_size = 1

        data_split = 1
        data_subsets = [data_all[i::data_split] for i in range(data_split)]
        sampling_params = SamplingParams(max_tokens=dataset2maxlen[dataset], use_beam_search=False, temperature=0.0)
        model, tokenizer = load_tokenizer(model2path[model_name], model_name)
        for json_obj in tqdm(data):
            prompt = build_input(tokenizer, **json_obj)
            output = model.generate(prompt, sampling_params)
            pred = output[0].outputs[0].text
            if pred == '':
                print(output)
            pred = post_process(pred, model_name)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"],
                           "length": json_obj["length"]}, f, ensure_ascii=False)
                f.write('\n')
            # get_pred(0,1,data_all,max_length,max_gen,prompt_format,dataset,device,model_name,model2path,out_path)
