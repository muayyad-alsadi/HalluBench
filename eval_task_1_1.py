import sys
import os
import json
import numpy as np
import pandas as pd
import statistics
import HalluBench

from dotenv import load_dotenv
# from any_llm import completion
from my_any_llm import completion

def llm_response(provider: str, model_name: str, sys_prompt: str, text: str):
    key_env = provider.upper()+'_API_KEY'
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"Please set {key_env} environment variable")
    response = completion(
        api_key=api_key,
        provider=provider,
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text},
        ])
    return response.choices[0].message.content

load_dotenv()

def eval_one(provider, model_name, df_in):
    sys_prompt = HalluBench.task1_prompt
    csv_str = HalluBench.df_to_csv_str(df_in)
    text = f"```csv\n{csv_str}\n```"
    res = llm_response(provider, model_name, sys_prompt, text)
    df_out = HalluBench.csv_str_to_df(res)
    print("output size:", len(df_out))
    e1_rate, d1 = HalluBench.eval_hallucination_rate(df_in, df_out)
    e2_rate, d2 = HalluBench.eval_correspondence(df_in, df_out)
    e_task, d3 = HalluBench.eval_sort_task(df_out)
    print("hallucination rate: ", e1_rate)
    print("corr mismatch rate: ", e2_rate)
    print("task errors", e_task)
    total = d1 + d2 + d3
    w1, w2, w3 = d1/total, d2/total, d3/total
    e = e1_rate*w1 + e2_rate*w2 + e_task*w3
    print("avg error rate", e)
    tol = 1e-16
    hr = statistics.harmonic_mean([max(tol, 1.0-e1_rate*w1) , max(tol, 1.0-e2_rate*w2), max(tol, 1.0-e_task*w3)])
    print("hr score rate", hr , " (higher is better)")


def main():
    n = int(sys.argv[1]) if len(sys.argv)==2 else 50
    with open('config.json', 'r', encoding='utf-8') as f:
        conf = json.load(f)
    df_in = HalluBench.get_df1(n=n)
    print("input CSV:\n", df_in)
    print("size of input CSV:\n", len(df_in))
    for provider, models in conf["enabled_models"].items():
        for model_name in models:
            print(f"** II ** evaluating {provider} {model_name}:... (error rates: lower is better)")
            try: eval_one(provider, model_name, df_in)
            except Exception as e:
                print(f"** II ** evaluating {provider} {model_name}: FAILED")
                print(e)

if __name__=="__main__":
    main()
