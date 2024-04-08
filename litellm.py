# -*- encoding: utf-8 -*-
"""
@File    : litellm.py
@Time    : 7/4/2024 15:39
@Author  : Duxy
@Email   : du.xi.yang@qq.com
@Software: PyCharm
"""
import json
import random
from concurrent.futures.thread import ThreadPoolExecutor

import requests


class LLM:
    def __init__(self, port):
        self.url = f"http://localhost:{port}/chat"

    def chat(self, prompt, history=[], **kwargs):
        record_id = kwargs.get("record_id", random.randint(0, 99999999))
        headers = {
            "Content-Type": "application/json",
            "cache_control": "no-cache"
        }
        temperature = kwargs.get("temperature", 0.01)
        max_new_tokens = kwargs.get("max_new_tokens", 64)
        # max_length = kwargs
        adapter_name = kwargs.get("adapter_name", "")
        skip_lora = (adapter_name == "original" or adapter_name == "")
        seed = kwargs.get("seed", 42)
        prefix_token_ids = kwargs.get("prefix_token_ids", [])

        input_data = {
            "query": prompt,
            "history": history,
            "request_id": record_id,
            "gen_kwarg": {
                "seed": seed,
                "prefix_token_ids": prefix_token_ids,
                "temperature": temperature,
                "skip_lora": skip_lora,
                "adapter_name": adapter_name,
                "return_raw": True,
                "max_tokens": max_new_tokens
            }
        }

        response = requests.post(url=self.url, headers=headers, data=json.dumps(input_data))

        if response.status_code != 200:
            return ""
        else:
            resp = response.json()
            return resp["response"]

    def parallel_chat(self, prompts, **kwargs):
        results = []
        excutor = ThreadPoolExecutor(max_workers=4)

        input_kwargs = [
            {"prompt": prompts[i], "history": []} for i in range(len(prompts))
        ]

        for kwarg in input_kwargs:
            kwarg.update(kwargs)

        print(input_kwargs[0].keys())
        print(input_kwargs[0]["prompt"][:10])
        for i, result in enumerate(excutor.map(lambda x: self.chat(**x), input_kwargs)):
            results.append(result)
        print(results)
        return results

if __name__ == '__main__':
    model = LLM(port=80801)

    prompts = ["prompt"] * 2
    history = ["history"] * 2
    kwargs = {"1":1, "2": 2}
    print(model.parallel_chat(prompts, **kwargs))


