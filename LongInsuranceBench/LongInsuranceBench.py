# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import datasets
import json


_DESCRIPTION = """\
LongInsuranceBench 是一个针对保险长文本场景下的长文本任务基准, 旨在测试评估大型预训练模型理解生成长文本的能力。 数据集目前主要包括，产品回答拒绝、多产品问答、产品计数、问题召回产品、摘要召回产品、产品问答复述。"""

_HOMEPAGE = "https://github.com/RealDuxy/LongInsuranceBench"

_URL = r"./"

task_list =  [
    "product_retrieval_summary",
    "product_retrieval_question",
    "product_count",
    "multi_product_qa",
    "deny_multi_product_qa",
    "repeat_product"
]


class LongInsuranceBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class LongInsuranceBench(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        LongInsuranceBenchConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "input": datasets.Value("string"),
                "context": datasets.Value("string"),
                "answers": [datasets.Value("string")],
                "length": datasets.Value("int32"),
                "dataset": datasets.Value("string"),
                "language": datasets.Value("string"),
                "all_classes": [datasets.Value("string")],
                "_id": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "data", f"{task_name}.jsonl"
                    ),
                },
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                key = f"{self.config.name}-{idx}"
                item = json.loads(line)
                yield key, {
                    "input": item["input"],
                    "context": item["context"],
                    "answers": item["answers"],
                    "length": item["length"],
                    "dataset": item["dataset"],
                    "language": item["language"],
                    "_id": item["_id"],
                    "all_classes": item["all_classes"],
                }