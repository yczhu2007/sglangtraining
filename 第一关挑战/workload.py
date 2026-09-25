import csv
import json
import random
import time
from itertools import islice

import ray
import requests

URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "Qwen/Qwen3-0.6B"
random.seed(25)

with open("mooncake_synthetic_trace.jsonl", encoding="utf-8") as source:
    trace = [json.loads(line) for line in islice(source, 12)]

@ray.remote
def send(index, raw_input, raw_output, target_input, target_output):
    prompt = "Briefly describe caching. /no_think\n" + "hello " * target_input
    started = time.perf_counter()
    try:
        response = requests.post(
            URL,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": target_output,
            },
            timeout=180,
        )
        latency = round(time.perf_counter() - started, 3)
        data = response.json()
        usage = data.get("usage") or {}
        return {
            "id": index,
            "trace_input": raw_input,
            "trace_output": raw_output,
            "target_input": target_input,
            "target_output": target_output,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "status": response.status_code,
            "latency_s": latency,
        }
    except Exception:
        return {
            "id": index,
            "trace_input": raw_input,
            "trace_output": raw_output,
            "target_input": target_input,
            "target_output": target_output,
            "input_tokens": 0,
            "output_tokens": 0,
            "status": "ERROR",
            "latency_s": round(time.perf_counter() - started, 3),
        }

ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR")
jobs = []

for index, item in enumerate(trace, start=1):
    time.sleep(random.expovariate(0.5))  # Poisson arrivals: mean gap 2 seconds
    raw_input = int(item["input_length"])
    raw_output = int(item["output_length"])
    target_input = min(320, max(32, raw_input // 128))
    target_output = min(48, max(4, raw_output))
    jobs.append(send.remote(index, raw_input, raw_output,
                            target_input, target_output))

results = ray.get(jobs)
columns = list(results[0])

with open("workload_results.csv", "w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)

print("id  trace_in trace_out target_in target_out actual_in actual_out status latency_s")
for row in results:
    print(
        f'{row["id"]:2}  {row["trace_input"]:8} {row["trace_output"]:9} '
        f'{row["target_input"]:9} {row["target_output"]:10} '
        f'{row["input_tokens"]:9} {row["output_tokens"]:10} '
        f'{str(row["status"]):>6} {row["latency_s"]:9}'
    )

ray.shutdown()
