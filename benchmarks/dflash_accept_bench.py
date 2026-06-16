#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
# DFlash per-request acceptance benchmark.
#
# Thin wrapper over ``vllm.benchmarks.serve`` (a.k.a. ``vllm bench serve``). It runs
# the IDENTICAL ShareGPT serving benchmark, but registers an extended openai-chat
# request function that also reads the per-request accepted/rejected prediction
# tokens that ``patch_dflash_accept_rate`` injects into the streamed ``usage``,
# then prints + saves avg/median/min/max acceptance rate and length plus an
# overall (all-requests) figure.
#
# Usage:
#   python -m benchmarks.dflash_accept_bench \
#       --backend openai-chat --base-url http://127.0.0.1:8000 \
#       --endpoint /v1/chat/completions \
#       --model <target> --served-model-name <name> \
#       --dataset-name sharegpt --dataset-path <sharegpt.json> \
#       --num-prompts 200 --num-speculative-tokens 15 [--save-result]
#
# NOTE: the extended request function is a copy of
# ``vllm.benchmarks.lib.endpoint_request_func.async_request_openai_chat_completions``
# with only the lines marked ``# DFLASH`` added. Re-sync it if you upgrade vLLM.
#
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback

import aiohttp

from vllm.benchmarks import serve as vbs
from vllm.benchmarks.lib import endpoint_request_func as erf

# Each successful chat request appends (accepted, rejected).
_ACCEPT_RECORDS: list[tuple[int, int]] = []


async def _async_request_openai_chat_with_accept(
    request_func_input,
    session: aiohttp.ClientSession,
    pbar=None,
    mm_position: str = "last",
):
    api_url = request_func_input.api_url
    erf._validate_api_url(api_url, "OpenAI Chat Completions API", "chat/completions")
    content = erf._get_chat_content(request_func_input, mm_position=mm_position)

    payload = {
        "model": request_func_input.model_name or request_func_input.model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": request_func_input.output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    erf._update_payload_common(payload, request_func_input)
    headers = erf._get_headers("application/json")
    erf._update_headers_common(headers, request_func_input)

    output = erf.RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len

    generated_text = ""
    ttft = 0.0
    accepted = None  # DFLASH
    rejected = None  # DFLASH
    st = time.perf_counter()
    output.start_time = st
    most_recent_timestamp = st
    try:
        async with session.post(url=api_url, json=payload, headers=headers) as response:
            if response.status == 200:
                handler = erf.StreamedResponseHandler()
                async for chunk_bytes in response.content.iter_any():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    for message in handler.add_chunk(chunk_bytes):
                        if message.startswith(":"):
                            continue
                        chunk = message.removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue
                        timestamp = time.perf_counter()
                        data = json.loads(chunk)
                        if choices := data.get("choices"):
                            delta_content = choices[0]["delta"].get("content")
                            if ttft == 0.0:
                                ttft = timestamp - st
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            generated_text += delta_content or ""
                        elif usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")
                            if (pt := usage.get("prompt_tokens")) is not None:
                                output.prompt_len = pt
                            # DFLASH: capture per-request accept fields if present.
                            details = usage.get("completion_tokens_details") or {}
                            a = details.get("accepted_prediction_tokens")
                            r = details.get("rejected_prediction_tokens")
                            if a is not None and r is not None:
                                accepted, rejected = int(a), int(r)
                        most_recent_timestamp = timestamp
                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - st
                if accepted is not None:  # DFLASH
                    _ACCEPT_RECORDS.append((accepted, rejected))
            else:
                output.error = response.reason or ""
                output.success = False
    except Exception:
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
    if pbar is not None:
        pbar.update(1)
    return output


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _report_acceptance(num_speculative_tokens: int) -> dict:
    # Per request: draft = accepted + rejected; rate = accepted / draft;
    # length (mean accepted tokens per verify step, incl. bonus) = 1 + k * rate,
    # where k = num_speculative_tokens drafted per step (constant for DFlash).
    per_req_rate, per_req_len = [], []
    total_accepted, total_draft = 0, 0
    for accepted, rejected in _ACCEPT_RECORDS:
        draft = accepted + rejected
        if draft <= 0:
            continue
        rate = accepted / draft
        per_req_rate.append(rate)
        per_req_len.append(1.0 + num_speculative_tokens * rate)
        total_accepted += accepted
        total_draft += draft

    overall_rate = (total_accepted / total_draft) if total_draft else 0.0
    report = {
        "dflash_num_requests_with_spec": len(per_req_rate),
        "dflash_overall_acceptance_rate": overall_rate,
        "dflash_overall_acceptance_length": 1.0 + num_speculative_tokens * overall_rate,
        "dflash_per_request_acceptance_rate": _summary(per_req_rate),
        "dflash_per_request_acceptance_length": _summary(per_req_len),
    }

    print("{s:{c}^{n}}".format(s=" DFlash Acceptance (per request) ", n=60, c="="))
    if not per_req_rate:
        print(
            "No speculative-decoding requests captured "
            "(server not in dflash mode, or patch_dflash_accept_rate not loaded)."
        )
    else:
        rs = report["dflash_per_request_acceptance_rate"]
        ls = report["dflash_per_request_acceptance_length"]
        print("{:<34} {:<10}".format("Requests w/ spec decode:", len(per_req_rate)))
        print(
            "{:<34} avg {:.3f}  med {:.3f}  min {:.3f}  max {:.3f}".format(
                "Acceptance rate (per req):", rs["avg"], rs["median"], rs["min"], rs["max"]
            )
        )
        print(
            "{:<34} avg {:.2f}  med {:.2f}  min {:.2f}  max {:.2f}".format(
                "Acceptance length (per req):", ls["avg"], ls["median"], ls["min"], ls["max"]
            )
        )
        print("{:<34} {:.3f}".format("Overall acceptance rate:", overall_rate))
        print("{:<34} {:.2f}".format("Overall acceptance length:", report["dflash_overall_acceptance_length"]))
    print("=" * 60)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DFlash per-request acceptance benchmark")
    vbs.add_cli_args(parser)
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        required=True,
        help="Block draft length k (num_speculative_tokens) configured on the server; "
        "used to convert acceptance rate -> acceptance length.",
    )
    args = parser.parse_args()

    # Register our extended openai-chat request func (mutates the shared registry
    # that vllm.benchmarks.serve reads).
    erf.ASYNC_REQUEST_FUNCS["openai-chat"] = _async_request_openai_chat_with_accept

    vbs.main(args)  # identical vllm bench serve flow, prints its own summary

    report = _report_acceptance(args.num_speculative_tokens)

    # Merge into the saved result JSON if vbs wrote one.
    result_path = getattr(args, "result_filename", None)
    if getattr(args, "save_result", False) and result_path:
        try:
            with open(result_path) as f:
                saved = json.load(f)
            saved.update(report)
            with open(result_path, "w") as f:
                json.dump(saved, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[dflash_accept_bench] could not merge into {result_path}: {exc}")


if __name__ == "__main__":
    main()
