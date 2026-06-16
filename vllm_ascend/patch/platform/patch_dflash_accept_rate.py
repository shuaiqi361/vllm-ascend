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
# Per-request DFlash / speculative-decoding acceptance accounting.
#
# Why: vLLM core computes a per-request accepted-token count each decode step
# (scheduler: ``num_accepted = len(generated_token_ids) - 1``) but folds it into
# an engine-wide aggregate and discards the per-request value. To report
# per-request acceptance from ``vllm bench serve``, we (1) accumulate per-request
# (draft_total, accepted_total) in the scheduler, (2) ship them to the API-server
# process via the existing ``EngineCoreOutput.kv_transfer_params`` dict
# side-channel (the only monkeypatch-safe carrier across the engine-core ->
# api-server process boundary; ``EngineCoreOutput`` is a positional ``msgspec``
# struct, so adding a typed field by monkeypatch would desync ser/deser), and
# (3) surface them on ``usage.completion_tokens_details.{accepted,rejected}_prediction_tokens``
# (the OpenAI "Predicted Outputs" fields, also added by
# ``patch_minimax_usage_accounting``).
#
# Every hook is guarded: if a vLLM symbol it depends on is missing, the feature
# disables itself with a warning rather than breaking server startup.
#
from __future__ import annotations

import json

from vllm.logger import logger

# Namespaced key so we never collide with real P/D KV-transfer payloads.
_DFLASH_KEY = "dflash_accept"


# =====================================================================
# Part 1 — Scheduler: accumulate per-request counts, inject on finish
# =====================================================================
def _install_scheduler_patch() -> bool:
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception as exc:  # noqa: BLE001
        logger.warning("DFlash accept patch: cannot import Scheduler (%s); skipping.", exc)
        return False

    if not hasattr(Scheduler, "make_spec_decoding_stats") or not hasattr(Scheduler, "update_from_output"):
        logger.warning("DFlash accept patch: Scheduler missing expected methods; skipping scheduler hook.")
        return False

    def _accept_store(scheduler) -> dict:
        # Per-scheduler store mapping request_id -> [draft_total, accepted_total].
        # Lives on the instance so it persists across a request's many decode steps.
        store = getattr(scheduler, "_dflash_accept_store", None)
        if store is None:
            store = {}
            scheduler._dflash_accept_store = store
        return store

    # --- Capture: wrap make_spec_decoding_stats. The scheduler calls it per
    #     request per spec-decode step with num_draft_tokens/num_accepted_tokens/
    #     request_id (kwargs at scheduler.py:1385-1391), regardless of log_stats.
    if not hasattr(Scheduler, "_ascend_orig_make_spec_decoding_stats"):
        Scheduler._ascend_orig_make_spec_decoding_stats = Scheduler.make_spec_decoding_stats

    def _patched_make_spec_decoding_stats(self, spec_decoding_stats, *args, **kwargs):
        num_draft = kwargs.get("num_draft_tokens", args[0] if len(args) >= 1 else None)
        num_accepted = kwargs.get("num_accepted_tokens", args[1] if len(args) >= 2 else None)
        req_id = kwargs.get("request_id", args[3] if len(args) >= 4 else None)
        if req_id is not None and num_draft is not None and num_accepted is not None:
            totals = _accept_store(self).setdefault(req_id, [0, 0])
            totals[0] += int(num_draft)
            totals[1] += int(num_accepted)
        return self._ascend_orig_make_spec_decoding_stats(spec_decoding_stats, *args, **kwargs)

    Scheduler.make_spec_decoding_stats = _patched_make_spec_decoding_stats

    # --- Ship: wrap update_from_output; attach final totals to each FINISHED
    #     request's outgoing EngineCoreOutput.kv_transfer_params.
    if not hasattr(Scheduler, "_ascend_orig_update_from_output"):
        Scheduler._ascend_orig_update_from_output = Scheduler.update_from_output

    def _iter_engine_core_outputs(ret):
        # update_from_output returns either an EngineCoreOutputs or a
        # dict[client_index -> EngineCoreOutputs]; yield each EngineCoreOutput.
        groups = ret.values() if isinstance(ret, dict) else [ret]
        for group in groups:
            for out in getattr(group, "outputs", None) or []:
                yield out

    def _patched_update_from_output(self, *args, **kwargs):
        ret = self._ascend_orig_update_from_output(*args, **kwargs)
        store = getattr(self, "_dflash_accept_store", None)
        if store:
            try:
                for out in _iter_engine_core_outputs(ret):
                    if getattr(out, "finish_reason", None) is None:
                        continue
                    totals = store.pop(getattr(out, "request_id", None), None)
                    if totals is None:
                        continue
                    draft_total, accepted_total = totals
                    payload = {
                        "draft": draft_total,
                        "accepted": accepted_total,
                        "rejected": max(0, draft_total - accepted_total),
                    }
                    if getattr(out, "kv_transfer_params", None) is None:
                        out.kv_transfer_params = {_DFLASH_KEY: payload}
                    else:
                        out.kv_transfer_params[_DFLASH_KEY] = payload
            except Exception as exc:  # noqa: BLE001  — never let accounting break decoding
                logger.warning("DFlash accept patch: inject failed (%s).", exc)
        return ret

    Scheduler.update_from_output = _patched_update_from_output
    return True


# =====================================================================
# Part 2 — Chat serving: surface counts on usage.completion_tokens_details
# =====================================================================
def _install_serving_patch() -> bool:
    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.engine import protocol as engine_protocol
    except Exception as exc:  # noqa: BLE001
        logger.warning("DFlash accept patch: cannot import chat serving (%s); skipping serving hook.", exc)
        return False

    if not hasattr(OpenAIServingChat, "chat_completion_stream_generator") or not hasattr(
        OpenAIServingChat, "chat_completion_full_generator"
    ):
        logger.warning("DFlash accept patch: OpenAIServingChat missing generators; skipping serving hook.")
        return False

    class _State:
        def __init__(self) -> None:
            self.payload = None

    def _extract(res):
        kv = getattr(res, "kv_transfer_params", None)
        if isinstance(kv, dict):
            payload = kv.get(_DFLASH_KEY)
            if isinstance(payload, dict):
                return payload
        return None

    async def _track(result_generator, state):
        async for res in result_generator:
            payload = _extract(res)
            if payload is not None:
                state.payload = payload
            yield res

    def _set_usage_details(usage_obj, payload) -> None:
        details = getattr(usage_obj, "completion_tokens_details", None)
        if details is None:
            ctu = getattr(engine_protocol, "CompletionTokenUsageInfo", None)
            if ctu is None:
                return  # minimax usage patch not loaded; nothing to attach to
            details = ctu()
            usage_obj.completion_tokens_details = details
        try:
            details.accepted_prediction_tokens = payload["accepted"]
            details.rejected_prediction_tokens = payload["rejected"]
        except Exception:  # noqa: BLE001
            pass

    def _inject_stream(data, state):
        if state.payload is None or not data.startswith("data: "):
            return data
        payload_str = data[len("data: "):]
        if payload_str.endswith("\n\n"):
            payload_str = payload_str[:-2]
        if payload_str == "[DONE]":
            return data
        try:
            chunk = json.loads(payload_str)
        except json.JSONDecodeError:
            return data
        usage = chunk.get("usage")
        if not isinstance(usage, dict):
            return data
        details = usage.get("completion_tokens_details")
        if not isinstance(details, dict):
            details = {}
        details["accepted_prediction_tokens"] = state.payload["accepted"]
        details["rejected_prediction_tokens"] = state.payload["rejected"]
        usage["completion_tokens_details"] = details
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Compose on top of whatever generators are installed (e.g. the minimax patch).
    if not hasattr(OpenAIServingChat, "_dflash_orig_stream_generator"):
        OpenAIServingChat._dflash_orig_stream_generator = OpenAIServingChat.chat_completion_stream_generator
    if not hasattr(OpenAIServingChat, "_dflash_orig_full_generator"):
        OpenAIServingChat._dflash_orig_full_generator = OpenAIServingChat.chat_completion_full_generator

    async def _stream_generator(self, request, result_generator, *args, **kwargs):
        state = _State()
        async for data in self._dflash_orig_stream_generator(
            request, _track(result_generator, state), *args, **kwargs
        ):
            yield _inject_stream(data, state)

    async def _full_generator(self, request, result_generator, *args, **kwargs):
        state = _State()
        response = await self._dflash_orig_full_generator(
            request, _track(result_generator, state), *args, **kwargs
        )
        if state.payload is not None and getattr(response, "usage", None) is not None:
            _set_usage_details(response.usage, state.payload)
        return response

    _stream_generator.__module__ = OpenAIServingChat.__module__
    _stream_generator.__qualname__ = f"{OpenAIServingChat.__qualname__}.chat_completion_stream_generator"
    _full_generator.__module__ = OpenAIServingChat.__module__
    _full_generator.__qualname__ = f"{OpenAIServingChat.__qualname__}.chat_completion_full_generator"

    OpenAIServingChat.chat_completion_stream_generator = _stream_generator
    OpenAIServingChat.chat_completion_full_generator = _full_generator
    return True


_sched_ok = _install_scheduler_patch()
_serve_ok = _install_serving_patch()
if _sched_ok and _serve_ok:
    logger.info("vllm-ascend: DFlash per-request acceptance accounting patch applied.")
else:
    logger.warning(
        "vllm-ascend: DFlash acceptance patch only partially applied (scheduler=%s, serving=%s).",
        _sched_ok,
        _serve_ok,
    )
