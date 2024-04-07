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

import requests


class LLM:
    url: str = "http://localhost:8081/chat"

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

    def para_chat(self, prompt, history, **kwargs):
        record_id = kwargs.get("record_id", random.randint(0, 99999999))
        headers = {
            "Content-Type": "application/json",
            "cache_control": "no-cache"
        }
        temperature = kwargs.get("temperature", 0.01)
        adapter_name = kwargs.get("adapter_name", "")
        skip_lora = (adapter_name == "original" or adapter_name == "")
        seed = kwargs.get("seed", 42)
        prefix_token_ids = kwargs.get("prefix_token_ids", [])

        input_data = {
            "prompt": prompt,
            "history": history,
            "gen_kwarg": {
                "seed": seed,
                "prefix_token_ids": prefix_token_ids,
                "temperature": temperature,
                "skip_lora": skip_lora,
                "adapter_name": adapter_name,
                "return_raw": True
            }
        }

        response = requests.post(url=self.url, headers=headers, data=json.dumps(input_data))

        if response.status_code != 200:
            return "error"
        else:
            resp = response.json()
            return resp["response"]
