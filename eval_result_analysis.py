import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    # code_sim_score,
    retrieval_product_zh_score, count_product_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    # "lcc": code_sim_score,
    # "repobench-p": code_sim_score,

    "product_count": count_product_score,
    "product_retrieval_summary": retrieval_product_zh_score,
    "product_retrieval_question": retrieval_product_zh_score,
    "deny_multi_product_qa": rouge_zh_score,
    "multi_product_qa": rouge_zh_score,
    "repeat_product": rouge_zh_score
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="longalign-6b-64k")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k-12k": [], "12k-16k": [], "overall": []}
    for (prediction, ground_truths, length) in tqdm(zip(predictions, answers, lengths)):
        score = 0.
        # if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        #     prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        elif length < 12000:
            scores["8k-12k"].append(score)
        else:
            scores["12k-16k"].append(score)
        scores["overall"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

def aggregate_scorer():
    pred_dir = f"pred_litellm/"
    model_list = [
        "chatglm3-6b",
        # "chatglm3-6b-32k",
        # "longalign-6b-64k",
        # "qwen15_4b_chat",
        # "qwen15_7b_chat",
        # "qwen15_14b_chat",
        # "qwen15_14b_chat_int4"
    ]
    # model_list = os.listdir(pred_dir)

    model_result_map = {}

    print("\n".join(model_list))
    final_results_display = defaultdict(list) # {task: []}

    for model in model_list:
        model_dir = os.path.join(pred_dir, model)
        result_file = os.path.join(model_dir, "result.json")
        result = json.load(open(result_file))
        model_result_map[model] = result
        for task, info in result.items():
            for range, score in info.items():
                final_results_display[f"{task}_{range}"].append(score)

    pd.DataFrame(final_results_display).to_excel("final_result.xlsx")






if __name__ == '__main__':
    aggregate_scorer()