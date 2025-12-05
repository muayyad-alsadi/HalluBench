
"""
HalluBench (or HalluMeter) a measure of LLM Hallucination Rate
"""

import hashlib
import random
import io
import datetime
import time
import math

from uuid import uuid4

import numpy as np
import pandas as pd

def csv_str_to_df(s: str):
    s=s.strip()
    if "```csv" in s and not s.startswith("```csv"):
        s=s.split('```csv', 1)[1].strip()
    if "```" in s and not s.startswith("```"):
        s=s.split('```', 1)[1].strip()
    buf=io.StringIO(s.strip().replace('```csv', '```').strip('`').strip())
    return pd.read_csv(buf)

def df_to_csv_str(df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()

def rnd_date(days_n=60):
    d=datetime.date.today()
    i=d.toordinal()
    return datetime.date.fromordinal(random.randint(i-days_n, i)).isoformat()

def rnd_ts(days_n=60):
    t0 = int(time.time())
    ts = random.randint(t0 - days_n*3600*24, t0)
    return datetime.datetime.fromtimestamp(ts).isoformat()+"Z"

def gen_item(i, signal_n=30, days_n=60, uuid_pool=None, md5_pool=None):
    uuid = str(uuid4()) if uuid_pool is None else random.choice(uuid_pool)
    md5 = hashlib.md5(uuid4().bytes).hexdigest() if md5_pool is None else random.choice(md5_pool)
    return {
        "id":i,
        "uuid": uuid,
        "md5":md5,
        "date": rnd_date(days_n),
        "update_timestamp": rnd_ts(days_n),
        "signal":random.randint(-signal_n,signal_n)}

def get_df1(n=50, skip_n=None):
    if skip_n is None:
        skip_n = math.ceil(n*0.05)
    signal_n = n*2//3
    skip = sorted(random.choices(list(range(n//2, n)), k=skip_n))
    data = [gen_item(i, signal_n) for i in range(1,n+1) if i not in skip]
    return pd.DataFrame(data, columns=data[0].keys())

def get_df2(n=50, skip_n=3, per_group=5):
    divisor = n // per_group
    signal_n = n // 2 // per_group
    skip = sorted(random.choices(list(range(n//2, n)), k=skip_n))
    data = [gen_item(i, signal_n, n // per_group) for i in range(1,n+1) if i not in skip]
    return pd.DataFrame(data, columns=data[0].keys())

def get_df3(n=50, skip_n=3, per_group=5, uuid_count=None, md5_count=None):
    divisor = n // per_group
    signal_n = n // 2 // per_group
    if uuid_count is None:
        uuid_count = divisor
    if md5_count is None:
        md5_count = divisor
    uuid_pool = [str(uuid4()) for _ in range(uuid_count)]
    md5_pool = [hashlib.md5(uuid4().bytes).hexdigest() for i in range(md5_count)]
    skip = sorted(random.choices(list(range(n//2, n)), k=skip_n))
    data = [gen_item(i, signal_n, divisor, uuid_pool, md5_pool) for i in range(1,n+1) if i not in skip]
    return pd.DataFrame(data, columns=data[0].keys())

def eval_hallucination_rate(df_true, df_out):
    cols = set(df_true.columns) - {'signal'}
    u=set()
    i=set()
    for key in cols:
        u |= set(df_true[key])
        u |= set(df_out[key])
        i |= set(df_true[key]) & set(df_out[key])
    print(u-i)
    d=len(u)
    e=len(u-i)
    key = 'signal'
    u = set(df_true[key]) | set(df_out[key])
    i = set(df_true[key]) & set(df_out[key])
    print(u-i)
    d += len(u)
    e += len(u-i)
    print(e, d, e/d)
    return e/d, d

def eval_correspondence(df_true, df_out):
    df1 = df_true.set_index(['id'])
    df2 = df_out.set_index(['id'])
    ids = set(df_true["id"]) | set(df_out["id"])
    ls = []
    e = 0
    for i in ids:
        r1 = df1.loc[i] if i in df1.index else None
        r2 = df2.loc[i] if i in df2.index else None
        if np.any(r1!=r2):
            ls.append(i)
            e+=1
    print(ls)
    d=len(ids)
    return e/d, d

def eval_correspondence_alt(df, df_out):
    d_in = {k:list(v)[1:] for k, v in zip(df["id"], df.itertuples())}
    d_out = {k:list(v)[1:] for k, v in zip(df_out["id"], df_out.itertuples())}
    ids = set(df["id"]) | set(df_out["id"])
    ls = []
    e = 0
    for i in ids:
        if d_in.get(i)!=d_out.get(i):
            ls.append(i)
            e+=1
    print(ls)
    d=len(ids)
    return e/d, d

def eval_sort_task(df_out, sort_by='signal', reverse=False):
    d = len(df_out)
    a = np.array(df_out[sort_by])
    if reverse:
        a = -a
    e_task = sum(np.argsort(a)!=np.arange(d)) / d
    return e_task, d

task1_prompt = "You will be given CSV with header, return the file ordered by value of `signal` column ascending. Give full and direct answer."

task1_2_prompt = """
You will be given CSV with header.
Your task is to construct a CSV such that:
* Rename `update_timestamp` into `update_date` and remove time part of it, keeping date part only. Move this column to be just after `id` (deleting original `update_timestamp`).
* Add a column named `hex4_id` just after newly add `update_date` having 4 hexadecimal digits formed by using the rightmost 2 digits from `md5` followed by the rightmost 2 digits from `uuid`.
* Drop `date` and `uuid` columns
* make `md5` uppercase while keeping `hex4_id` lowercase.
* Keep other columns as-is and in same order.
* rows should be returned in reverse order.
* Give full CSV output, and direct answer only no explanation.
""".strip()

def get_task_1_2_true(df_in):
    df_true = pd.DataFrame(df_in)
    df_true['update_date'] = [i[:10] for i in df_true['update_timestamp']]
    df_true['hex4_id'] = [i[-2:]+j[-2:] for i,j in zip(df_true['md5'], df_true['uuid'])]
    df_true['md5'] = [i.upper() for i in df_true['md5']]
    df_true = pd.DataFrame(df_true, columns=('id', 'update_date', 'hex4_id', 'md5', 'signal'))
    return df_true

def get_task2_prompt(grouping_key, grouping_val, example_1="1970-12-30", example_2="1970-12-31"):
    return f"""
You will be given CSV with header.
Your task is to group the items by values in `{grouping_key}` column.
Return a JSON object having each value from `{grouping_key}` coulmns as key paired with a list of the corresponding values of `{grouping_val}` column that belong to it
<example>
{{
"{example_1}":[...],
"{example_2}":[...],
...
}}
</exaple>
Only valid JSON.
""".strip()
