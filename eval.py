import os
import json
import argparse
from json import JSONDecodeError

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
    retrieval_product_zh_score, count_product_score, translation_edit_distance,
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
    "repeat_product": translation_edit_distance
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="chatglm3-6b")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes, label_scores):
    scores = {"0-4k": [], "4-8k": [], "8k-12k": [], "12k-16k": [], "overall": []}
    label_scores = [None]*len(predictions) if not label_scores else label_scores
    for (prediction, ground_truths, length, label_score) in tqdm(zip(predictions, answers, lengths, label_scores)):
        score = 0.
        # if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        #     prediction = prediction.lstrip('\n').split('\n')[0]
        if label_score:
            score = label_score
        else:
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
        # print(f"计算分数：")
        # print(f"{key}: {scores[key][:]}")
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

def pred2excel(data_file):
    lines = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            line.update({"label_score": 0})
            lines.append(line)
    pd.DataFrame(lines).to_excel(data_file.replace(".jsonl", ".xlsx"))

def excel2pred(data_file):
    lines = pd.read_excel(data_file, index_col=None).to_dict("records")
    with open(data_file.replace(".xlsx", ".jsonl"), "a", encoding="utf-8") as f:
        for line in lines:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

if __name__ == '__main__':
    pred_dir = "pred/longalign-6b-64k"
    # pred_file = os.path.join(pred_dir, "deny_multi_product_qa.jsonl")
    # pred2excel(pred_file)

    excel_file = os.path.join(pred_dir,"deny_multi_product_qa.xlsx")
    excel2pred(excel_file)

    args = parse_args()
    for model in [
                # "chatglm3-6b",
                #   "chatglm3-6b-32k",
                #   "longalign-6b-64k",

                  "qwen15_0-5b_chat",
                  # "qwen15_1-8b_chat",
                  # "qwen15_4b_chat",
                  # "qwen15_7b_chat",
                  # "qwen15_14b_chat",
                  # "qwen15_14b_chat_int4",
                  # "qwen15_32b_chat_int4"
    ]:
        print("=="*20, f"评估模型 {model}", "=="*20)
        args.model = model
        scores = dict()
        pred_dir = "pred"

        path = os.path.join(pred_dir, f"{args.model}/")

        all_files = os.listdir(path)
        print("Evaluating on:", all_files)
        for filename in all_files:
            print(f"evaluating {filename}")
            if not filename.endswith("jsonl"):
                continue
            predictions, answers, lengths = [], [], []
            label_scores = []
            dataset = filename.split('.')[0]
            with open(f"{path}{filename}", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    # except JSONDecodeError:
                    #     print(line)
                    predictions.append(data["pred"])
                    answers.append(data["answers"])
                    all_classes = data["all_classes"]
                    if "length" in data:
                        lengths.append(data["length"])
                    if "label_score" in data:
                        label_scores.append(data["label_score"])
            # if args.e:
            #     score = scorer_e(dataset, predictions, answers, lengths, all_classes)
            # else:
            #     score = scorer(dataset, predictions, answers, all_classes)
            score = scorer_e(dataset, predictions, answers, lengths, all_classes,label_scores)
            scores[dataset] = score
        out_path = os.path.join(pred_dir, f"{args.model}/") + "result.json"
        with open(out_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)

        # pd.DataFrame(scores).to_excel(f"pred/{args.model}/result.xlsx")
