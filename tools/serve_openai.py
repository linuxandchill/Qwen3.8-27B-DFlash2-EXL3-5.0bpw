#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the EXL3 serving target.

Drafter: MTP by default (`-dm mtp`; the draft head lives inside the target
checkpoint). Alternatives: a DFlash2 draft model directory (`-dm <dir>`),
or no drafting (`-dm none`). The start.sh launcher maps the .env `DRAFT`
knob onto these.

Endpoints:
  GET  /v1/models
  GET  /health
  POST /v1/chat/completions   (stream and non-stream, tool calling)

Defaults use Qwen3.8's low reasoning effort, top-k 20 and top-p 0.95.
Requests can disable thinking or select low/medium/xhigh through the official
chat-template controls. Speculative drafting remains active by default.
Concurrency: requests are serialized (batch-1 draft); concurrent callers queue.

Tool calling (Qwen3.8 XML format):
  - `tools` (OpenAI function specs) are rendered by the model's HF chat template
    (system "# Tools" section). `tool_choice` is accepted; required/specific
    choices are enforced with an explicit system directive.
  - assistant history with `tool_calls` is re-rendered natively by the template
    (arguments are converted JSON-string -> dict, as the template expects).
  - `role:"tool"` messages render as `<tool_response>` blocks natively.
  - Model output `<tool_call><function=name><parameter=k>v</parameter>
    </function></tool_call>` is parsed back into OpenAI `tool_calls` objects;
    generation stops at `</tool_call>`, finish_reason = "tool_calls".
  - Tool-call arguments are typed per the request's own JSON schemas
    (integer/number/boolean/array/object), strings kept on mismatch.

Launch (from repo root):
  .venv/bin/python tools/serve_openai.py \
      -m models/Qwen3.8-27B-EXL3-3.5bpw -gs 22 -cs 262144 -cq nvfp4 --port 8888
"""
import argparse, json, os, re, sys, time, threading, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiohttp import web

MODEL_DIR = "test_models/Qwen3.8-27B-exl3-3.5bpw-wm"
DRAFT_DIR = "mtp"   # default drafting method: MTP head (no external draft model)
PORT = 8888

gen_lock = threading.Lock()          # serialize generation (batch-1 draft)
stats_lock = threading.Lock()
log_lock = threading.Lock()
# Cumulative counters for sparkDash live tok/s (GET /health).
stats = {
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "context_length": None,
}

def _bump_stats(prompt=0, completion=0):
    if prompt <= 0 and completion <= 0:
        return
    with stats_lock:
        if prompt > 0:
            stats["prompt_tokens_total"] += int(prompt)
        if completion > 0:
            stats["completion_tokens_total"] += int(completion)

def _result_new_tokens(r):
    ids = r.get("token_ids") if isinstance(r, dict) else None
    if ids is None:
        return 0
    try:
        return int(ids.shape[-1])
    except Exception:
        return 0

def _log_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    value = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./:@,+-]+", value):
        return value
    return json.dumps(value, ensure_ascii = False)


def log_request(request_id, event, **fields):
    """Write one atomic, grep-friendly request lifecycle line."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    details = " ".join(f"{key}={_log_value(value)}" for key, value in fields.items())
    line = f"{timestamp} [{request_id}] {event}"
    if details:
        line += " " + details
    with log_lock:
        print(line, flush = True)


def _draft_method(generator):
    if getattr(generator, "dflash_draft", False):
        return "dflash2"
    if getattr(generator, "mtp_draft", False):
        return "mtp"
    if getattr(generator, "draft_model", None) is not None:
        return "draft_model"
    if getattr(generator, "ngram_match_min", 0):
        return "ngram"
    return "none"

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
HOLD_BACK = 16                       # marker-safe holdback for streamed text
REASONING_EFFORTS = ("low", "medium", "xhigh")
CHAT_TEMPLATE_RESERVED = {
    "add_generation_prompt", "tokenize", "return_dict", "return_tensors", "tools"
}


def build_model(argv, use_draft = True):
    from argparse import ArgumentParser

    # The one-time JIT build of the CUDA extension can look like a hang;
    # say so before the import below blocks on it.
    try:
        import importlib.util, os
        if importlib.util.find_spec("exllamav3_ext") is None:
            _root = os.environ.get("TORCH_EXTENSIONS_DIR",
                                   os.path.expanduser("~/.cache/torch_extensions"))
            if not (os.path.isdir(_root) and
                    any(d == "exllamav3_ext"
                        for _, _dirs, _ in os.walk(_root) for d in _dirs)):
                print(" == compiling the CUDA extension "
                      "(one-time; a few minutes of silence is normal) ...", flush = True)
    except Exception:
        pass

    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = use_draft)
    args = parser.parse_args(argv)
    if use_draft:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(args, progress = True)
        generator = Generator(
            model, cache, tokenizer,
            draft_model = draft_model, draft_cache = draft_cache,
            # num_draft_tokens defaults to the draft model's arch-declared
            # default_draft_size (DFlash2: block_size - 1 = 7, MTP: 4). Must
            # match model_init's max_history sizing, which reads the same caps.
        )
    else:
        model, config, cache, tokenizer = model_init.init(args, progress = True)
        generator = Generator(model, cache, tokenizer)
    return generator, tokenizer


def normalize_messages(messages):
    """OpenAI history -> template-compatible dicts (tool_calls args str->dict)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                fn["arguments"] = args
                calls.append({"function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def resolve_chat_template_kwargs(body, defaults = None):
    """Merge server defaults with Qwen/OpenAI thinking controls."""
    nested = body.get("chat_template_kwargs")
    if nested is None:
        nested = {}
    elif not isinstance(nested, dict):
        return None, "`chat_template_kwargs` must be an object"

    reserved = sorted(CHAT_TEMPLATE_RESERVED.intersection(nested))
    if reserved:
        return None, (
            "`chat_template_kwargs` cannot override server-owned fields: "
            + ", ".join(reserved)
        )

    kwargs = dict(defaults or {})
    kwargs.update(nested)
    for key in ("enable_thinking", "preserve_thinking", "reasoning_effort"):
        if key not in body:
            continue
        if key in nested and nested[key] != body[key]:
            return None, f"conflicting top-level and chat_template_kwargs `{key}`"
        kwargs[key] = body[key]

    for key in ("enable_thinking", "preserve_thinking"):
        if key in kwargs and not isinstance(kwargs[key], bool):
            return None, f"`{key}` must be a boolean"

    explicit_effort = (
        "reasoning_effort" in body or "reasoning_effort" in nested
    )
    explicit_enable = (
        "enable_thinking" in body or "enable_thinking" in nested
    )
    effort = kwargs.get("reasoning_effort")
    if effort is not None:
        if not isinstance(effort, str):
            return None, "`reasoning_effort` must be a string"
        effort = effort.lower()
        if effort == "none":
            if explicit_enable and kwargs.get("enable_thinking"):
                return None, "`reasoning_effort: none` conflicts with `enable_thinking: true`"
            kwargs["enable_thinking"] = False
            kwargs.pop("reasoning_effort", None)
        elif effort not in REASONING_EFFORTS:
            supported = ", ".join(REASONING_EFFORTS)
            return None, f"`reasoning_effort` must be one of: none, {supported}"
        else:
            if explicit_effort and explicit_enable and kwargs.get("enable_thinking") is False:
                return None, (
                    f"`reasoning_effort: {effort}` conflicts with "
                    "`enable_thinking: false`"
                )
            kwargs["reasoning_effort"] = effort
            if explicit_effort:
                kwargs["enable_thinking"] = True

    if kwargs.get("enable_thinking") is False:
        kwargs.pop("reasoning_effort", None)
    return kwargs, None


def render_chat_prompt(tokenizer, messages, tools, chat_template_kwargs):
    """Render one request with the model's native HF chat template."""
    return tokenizer.hf_chat_template(
        messages, add_generation_prompt = True, tools = tools,
        **chat_template_kwargs)


def split_reasoning(text):
    """Split Qwen reasoning from content when a think block was generated.

    Returns (reasoning, content) with markers stripped."""
    close = text.find("</think>")
    if close >= 0:
        reasoning = text[:close]
        content = text[close + len("</think>"):]
        return reasoning.lstrip().removeprefix("<think>").strip(), content.strip("\n")
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[len("<think>"):].strip(), ""
    return "", text


def build_tool_schemas(tools):
    """OpenAI tools list -> {function_name: {param_name: json-schema type}}."""
    schemas = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if name and isinstance(props, dict):
            schemas[name] = {k: v.get("type") for k, v in props.items()
                             if isinstance(v, dict)}
    return schemas


def _coerce_value(value, jtype):
    """Coerce one XML string parameter to the schema-declared JSON type.
    Lossless: on any mismatch the original string is returned unchanged."""
    v = value.strip()
    if not v:
        return value
    try:
        if jtype == "integer":
            return int(v)
        if jtype == "number":
            try:
                return int(v)
            except ValueError:
                return float(v)
        if jtype == "boolean":
            if v.lower() == "true": return True
            if v.lower() == "false": return False
        if jtype == "array":
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        if jtype == "object":
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    return value


def coerce_tool_args(args, fn_schema):
    """Qwen's XML tool format delivers every parameter value as a string;
    OpenAI tool_calls arguments are typed JSON. Coerce each value using the
    request's own tool schema; undeclared params and failed coercions keep
    the raw string."""
    if not fn_schema:
        return args
    out = {}
    for k, v in args.items():
        t = fn_schema.get(k)
        types = t if isinstance(t, list) else [t]
        for tt in types:
            if isinstance(tt, str) and tt in ("integer", "number", "boolean",
                                              "array", "object"):
                cv = _coerce_value(v, tt)
                if not isinstance(cv, str):
                    v = cv
                    break
        out[k] = v
    return out


def parse_tool_calls(text, tool_schemas = None):
    """Parse Qwen XML tool calls. Returns (content_without_calls, [calls]).
    A <tool_call> block left unterminated is treated as complete: the
    </tool_call> stop-condition strips the closing tag from generated text."""
    calls = []
    content = text

    def parse_block(block):
        fm = re.search(r"<function=([^>]+)>", block)
        if not fm:
            return None
        name = fm.group(1).strip()
        args = {}
        for pm in re.finditer(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
                              block[fm.end():], flags = re.S):
            args[pm.group(1).strip()] = pm.group(2)
        if tool_schemas:
            args = coerce_tool_args(args, tool_schemas.get(name))
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    while True:
        i = content.find(TOOL_CALL_OPEN)
        if i < 0:
            break
        j = content.find(TOOL_CALL_CLOSE, i)
        if j < 0:
            # truncated close (stop string consumed): parse the remainder
            call = parse_block(content[i + len(TOOL_CALL_OPEN):])
            if call:
                calls.append(call)
            content = content[:i]
            break
        call = parse_block(content[i + len(TOOL_CALL_OPEN):j])
        if call:
            calls.append(call)
        content = content[:i] + content[j + len(TOOL_CALL_CLOSE):]
    return content, calls


def tool_choice_directive(tool_choice, tools):
    """OpenAI tool_choice -> (tools_to_render, extra system directive or None).
    The Qwen template has no tool_choice support, so required/specific are
    enforced with an explicit instruction appended to the history."""
    if tool_choice in (None, "auto"):
        return tools, None
    if tool_choice == "none":
        return None, None
    names = [t["function"]["name"] for t in (tools or [])
             if isinstance(t, dict) and t.get("type") == "function"]
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        return tools, (f"You must call the function `{name}` now. Reply ONLY with "
                       f"the <tool_call> block for `{name}` and nothing else.")
    if tool_choice == "required":
        one_of = " or ".join(f"`{n}`" for n in names)
        return tools, (f"You must call one of the available functions ({one_of}) "
                       "now. Reply ONLY with the <tool_call> block and nothing else.")
    return tools, None


def generate_full(generator, tokenizer, messages, max_tokens, sampling, seed,
                  tools, tool_choice = None, stop = None, on_text = None,
                  on_start = None, chat_template_kwargs = None):
    """Blocking generation; returns text/result fields plus timing metrics."""
    operation_started = time.perf_counter()
    schemas = build_tool_schemas(tools)
    tools, directive = tool_choice_directive(tool_choice, tools)
    if directive:
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            # Qwen template allows only ONE leading system message — merge
            first = dict(messages[0])
            first["content"] = (first.get("content") or "").rstrip() + "\n\n" + directive
            messages[0] = first
        else:
            messages = [{"role": "system", "content": directive}] + messages
    tokenize_started = time.perf_counter()
    input_ids = render_chat_prompt(
        tokenizer, messages, tools, chat_template_kwargs or {})
    tokenize_seconds = time.perf_counter() - tokenize_started
    prompt_toks = int(input_ids.shape[-1])
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""
    attempts = []

    def run_once(attempt):
        nonlocal text, reason
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(**sampling)
        stop_conditions = ["<|im_end|>", tokenizer.eos_token_id] + (stop or [])
        job = Job(input_ids = input_ids, max_new_tokens = max_tokens,
                  stop_conditions = stop_conditions,
                  sampler = sampler, seed = seed)
        prefill_seen = 0
        final_result = {}
        queue_started = time.perf_counter()
        with gen_lock:
            queue_seconds = time.perf_counter() - queue_started
            if on_start is not None:
                on_start(attempt, prompt_toks, queue_seconds)
            generator.enqueue(job)
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r.get("stage") == "prefill":
                        curr = int(r.get("curr_progress") or 0)
                        if curr > prefill_seen:
                            _bump_stats(prompt=curr - prefill_seen)
                            prefill_seen = curr
                    else:
                        new_tokens = _result_new_tokens(r)
                        if new_tokens:
                            _bump_stats(completion=new_tokens)
                    chunk = r.get("text", "")
                    if chunk:
                        text += chunk
                        if on_text is not None:
                            on_text(chunk)
                    if r.get("eos"):
                        reason = r.get("eos_reason", reason)
                        final_result = r
            if prefill_seen < prompt_toks:
                _bump_stats(prompt=prompt_toks - prefill_seen)

        accepted = int(final_result.get(
            "accepted_draft_tokens",
            getattr(job, "accepted_draft_tokens", 0)) or 0)
        rejected = int(final_result.get(
            "rejected_draft_tokens",
            getattr(job, "rejected_draft_tokens", 0)) or 0)
        attempts.append({
            "queue_seconds": queue_seconds,
            "prefill_seconds": float(final_result.get(
                "time_prefill", getattr(job, "time_prefill", 0.0)) or 0.0),
            "generation_seconds": float(final_result.get(
                "time_generate", getattr(job, "time_generate", 0.0)) or 0.0),
            "accepted_draft_tokens": accepted,
            "rejected_draft_tokens": rejected,
            "cached_prompt_tokens": int(final_result.get(
                "cached_tokens", getattr(job, "cached_tokens", 0)) or 0),
        })
        return job

    job = run_once(1)
    # Forced tool_choice is a prompt nudge; at temperature > 0 the model can
    # occasionally skip the call. One greedy retry makes it deterministic.
    if forced_choice and not parse_tool_calls(text, schemas)[1]:
        temperature = 0.0
        job = run_once(2)
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    content, calls = parse_tool_calls(text, schemas)
    if calls:
        finish = "tool_calls"
    else:
        finish = {"max_new_tokens": "length", "eos": "stop",
                  "stop_condition": "stop", "banned": "content_filter"}.get(
                      reason, "stop")
    reasoning, content = split_reasoning(content)

    accepted = sum(a["accepted_draft_tokens"] for a in attempts)
    rejected = sum(a["rejected_draft_tokens"] for a in attempts)
    drafted = accepted + rejected
    metrics = {
        "attempts": len(attempts),
        "tokenize_seconds": tokenize_seconds,
        "queue_seconds": sum(a["queue_seconds"] for a in attempts),
        "prefill_seconds": sum(a["prefill_seconds"] for a in attempts),
        "generation_seconds": sum(a["generation_seconds"] for a in attempts),
        "cached_prompt_tokens": attempts[-1]["cached_prompt_tokens"],
        "draft_method": _draft_method(generator),
        "draft_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "draft_acceptance_rate": accepted / drafted if drafted else None,
    }
    return text, calls, finish, prompt_toks, out_toks, reasoning, content, metrics


async def models(request):
    ctx = stats.get("context_length")
    return web.json_response({"object": "list", "data": [{
        "id": "qwen3.8-27b-exl3-3.5bpw-wm",
        "object": "model",
        "owned_by": "exl3",
        **({"max_model_len": ctx} if ctx else {}),
    }]})


async def health(request):
    with stats_lock:
        return web.json_response({
            "ok": True,
            "busy": gen_lock.locked(),
            "backend": "exl3",
            "prompt_tokens_total": stats["prompt_tokens_total"],
            "completion_tokens_total": stats["completion_tokens_total"],
            "context_length": stats["context_length"],
        })


def parse_request(body, template_defaults = None):
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"
    chat_template_kwargs, template_error = resolve_chat_template_kwargs(
        body, template_defaults)
    if template_error:
        return None, template_error
    max_tokens = int(body.get("max_tokens") or
                     body.get("max_completion_tokens") or 1024)
    enable_thinking = chat_template_kwargs.get("enable_thinking", True)
    temperature = float(body.get("temperature", 1.0 if enable_thinking else 0.7))
    top_p = float(body.get("top_p", 0.95 if enable_thinking else 0.80))
    top_k = int(body.get("top_k", 20))
    sampling = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": float(body.get("min_p", 0.0)),
        "pres_p": float(body.get(
            "presence_penalty", 0.0 if enable_thinking else 1.5)),
        "freq_p": float(body.get("frequency_penalty", 0.0)),
        "rep_p": float(body.get("repetition_penalty", 1.0)),
    }
    seed = body.get("seed")
    tools = body.get("tools") or None
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None
    return dict(
        messages = normalize_messages(messages),
        max_tokens = max_tokens, sampling = sampling,
        seed = int(seed) if seed is not None else None,
        tools = tools,
        tool_choice = body.get("tool_choice"),
        stop = stop,
        stream = bool(body.get("stream", False)),
        model_id = body.get("model", "qwen3.8-27b-exl3-3.5bpw-wm"),
        chat_template_kwargs = chat_template_kwargs,
        enable_thinking = enable_thinking,
    ), None



def _message_summary(messages):
    roles = {}
    input_chars = 0
    for message in messages:
        role = str(message.get("role", "unknown"))
        roles[role] = roles.get(role, 0) + 1
        content = message.get("content")
        if isinstance(content, str):
            input_chars += len(content)
        elif content is not None:
            try:
                input_chars += len(json.dumps(content, ensure_ascii = False))
            except (TypeError, ValueError):
                pass
    return len(messages), ",".join(f"{role}:{count}" for role, count in roles.items()), input_chars


def _log_completion(request_id, request_started, stream, finish, prompt_toks,
                    out_toks, metrics):
    generation_seconds = metrics["generation_seconds"]
    prefill_seconds = metrics["prefill_seconds"]
    uncached_prompt_toks = max(0, prompt_toks - metrics["cached_prompt_tokens"])
    tok_s = out_toks / generation_seconds if generation_seconds > 0 else None
    fields = {
        "status": 200,
        "stream": stream,
        "finish": finish,
        "prompt_tokens": prompt_toks,
        "completion_tokens": out_toks,
        "total_tokens": prompt_toks + out_toks,
        "request_ms": round((time.perf_counter() - request_started) * 1000, 1),
        "tokenize_ms": round(metrics["tokenize_seconds"] * 1000, 1),
        "queue_ms": round(metrics["queue_seconds"] * 1000, 1),
        "prefill_ms": round(prefill_seconds * 1000, 1),
        "decode_ms": round(generation_seconds * 1000, 1),
        "tok_s": round(tok_s, 2) if tok_s is not None else None,
        "ms_per_token": round(1000 / tok_s, 2) if tok_s else None,
        "prefill_tok_s": round(uncached_prompt_toks / prefill_seconds, 2)
                         if prefill_seconds > 0 else None,
        "cached_prompt_tokens": metrics["cached_prompt_tokens"],
        "attempts": metrics["attempts"],
        "draft": metrics["draft_method"],
    }
    if metrics["draft_method"] != "none":
        fields.update({
            "draft_tokens": metrics["draft_tokens"],
            "draft_accepted": metrics["accepted_draft_tokens"],
            "draft_rejected": metrics["rejected_draft_tokens"],
            "draft_acceptance_pct": round(metrics["draft_acceptance_rate"] * 100, 2)
                                    if metrics["draft_acceptance_rate"] is not None else None,
        })
    log_request(request_id, "completed", **fields)


async def chat_completions(request):
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
    request_started = time.perf_counter()
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    log_request(cid, "received", method = request.method, path = request.path,
                remote = request.remote or "unknown",
                content_bytes = request.content_length)
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        message = f"request body exceeds {request.app['max_body_mb']} MiB limit"
        log_request(cid, "rejected", status = 413, reason = message)
        # aiohttp enforces client_max_size inside request.json(); without this
        # branch it falls into the generic handler below and gets misreported
        # as "invalid JSON" (400) even though the body parsed fine.
        return web.json_response(
            {"error": {"message": message,
                       "type": "invalid_request_error",
                       "code": "request_entity_too_large"}},
            status = 413)
    except Exception as e:
        log_request(cid, "rejected", status = 400, reason = "invalid_json",
                    error = type(e).__name__)
        return web.json_response({"error": {"message": "invalid JSON"}}, status = 400)
    req, err = parse_request(body, request.app["chat_template_defaults"])
    if err:
        log_request(cid, "rejected", status = 400, reason = err)
        return web.json_response({"error": {"message": err}}, status = 400)

    message_count, roles, input_chars = _message_summary(req["messages"])
    log_request(
        cid, "queued",
        model = req["model_id"],
        stream = req["stream"],
        messages = message_count,
        roles = roles,
        input_chars = input_chars,
        max_tokens = req["max_tokens"],
        temperature = req["sampling"]["temperature"],
        top_p = req["sampling"]["top_p"],
        top_k = req["sampling"]["top_k"],
        min_p = req["sampling"]["min_p"],
        presence_penalty = req["sampling"]["pres_p"],
        frequency_penalty = req["sampling"]["freq_p"],
        repetition_penalty = req["sampling"]["rep_p"],
        seed = req["seed"],
        tools = len(req["tools"] or []),
        tool_choice = req["tool_choice"],
        stops = len(req["stop"] or []),
        enable_thinking = req["enable_thinking"],
        reasoning_effort = req["chat_template_kwargs"].get("reasoning_effort"),
        preserve_thinking = req["chat_template_kwargs"].get("preserve_thinking"),
    )

    def on_start(attempt, prompt_toks, queue_seconds):
        log_request(cid, "fulfilling", attempt = attempt,
                    prompt_tokens = prompt_toks,
                    queue_ms = round(queue_seconds * 1000, 1),
                    draft = _draft_method(generator))

    import asyncio
    if not req["stream"]:
        try:
            text, calls, finish, ptoks, otoks, reasoning, content, metrics = \
                await asyncio.to_thread(
                    generate_full, generator, tokenizer, req["messages"],
                    req["max_tokens"], req["sampling"], req["seed"], req["tools"],
                    req["tool_choice"], req["stop"], None, on_start,
                    req["chat_template_kwargs"])
        except AssertionError as e:
            log_request(cid, "failed", status = 400, reason = "context_or_cache",
                        error = str(e))
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status = 400)
        except Exception as e:
            log_request(cid, "failed", status = 500, error = type(e).__name__,
                        detail = str(e),
                        request_ms = round((time.perf_counter() - request_started) * 1000, 1))
            return web.json_response(
                {"error": {"message": "generation failed", "type": "server_error"}},
                status = 500)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        _log_completion(cid, request_started, False, finish, ptoks, otoks, metrics)
        return web.json_response({
            "id": cid,
            "object": "chat.completion", "created": int(time.time()),
            "model": req["model_id"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": otoks,
                      "total_tokens": ptoks + otoks},
        })

    # ---- streaming (SSE) ----
    resp = web.StreamResponse(headers = {
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    model_id = req["model_id"]
    req_schemas = build_tool_schemas(req["tools"])

    async def run():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        forced_choice = req["tool_choice"] not in (None, "auto", "none")

        def worker():
            try:
                result = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["sampling"], req["seed"], req["tools"],
                    req["tool_choice"], req["stop"],
                    on_text = None if forced_choice else on_text,
                    on_start = on_start,
                    chat_template_kwargs = req["chat_template_kwargs"])
                loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", e))
        loop.run_in_executor(None, worker)

        async def send(delta, finish = None):
            obj = {"id": cid, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        pending, finish, calls_emitted = "", None, False
        call_idx = [0]
        in_think = [req["enable_thinking"]]
        THINK_CLOSE = "</think>"

        async def send_call(c):
            nonlocal calls_emitted
            calls_emitted = True
            await send({"tool_calls": [dict(c, index = call_idx[0])]})
            call_idx[0] += 1

        async def flush_pending(final = False):
            """Emit everything parseable from pending; keep marker-safe tail."""
            nonlocal pending
            while True:
                if in_think[0]:
                    close = pending.find(THINK_CLOSE)
                    if close >= 0:
                        head, pending = pending[:close], pending[close + len(THINK_CLOSE):]
                        if head.strip():
                            await send({"reasoning_content": head.lstrip("\n")})
                        in_think[0] = False
                        continue
                    cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                    piece = pending[:cut]
                    if piece.strip():
                        await send({"reasoning_content": piece})
                    pending = pending[cut:]
                    return
                if TOOL_CALL_OPEN in pending:
                    head, rest = pending.split(TOOL_CALL_OPEN, 1)
                    if head.strip() or (final and head):
                        await send({"content": head})
                    if TOOL_CALL_CLOSE in rest:
                        block, pending = rest.split(TOOL_CALL_CLOSE, 1)
                        _, calls = parse_tool_calls(
                            TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE,
                            req_schemas)
                        for c in calls:
                            await send_call(c)
                        continue
                    # unterminated call: final -> implicit close, else hold
                    if final and "<function=" in rest:
                        _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest,
                                                    req_schemas)
                        for c in calls:
                            await send_call(c)
                        pending = ""
                    else:
                        pending = TOOL_CALL_OPEN + rest
                    return
                cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                await send({"content": pending[:cut]})
                pending = pending[cut:]
                return

        completed = None
        while True:
            kind, payload = await queue.get()
            if kind == "error":
                log_request(cid, "failed", status = 500,
                            error = type(payload).__name__, detail = str(payload),
                            request_ms = round(
                                (time.perf_counter() - request_started) * 1000, 1))
                await resp.write(
                    f'data: {json.dumps({"error": {"message": str(payload)}})}\n\n'.encode())
                break
            if kind == "delta":
                pending += payload
                await flush_pending()
            elif kind == "done":
                text, calls, finish, ptoks, otoks, reasoning, content, metrics = payload
                await flush_pending(final = True)
                if forced_choice:
                    # Buffered path (no deltas were streamed): emit the
                    # authoritative complete result as deltas.
                    if reasoning:
                        await send({"reasoning_content": reasoning})
                    if content:
                        await send({"content": content})
                if not calls_emitted and calls:
                    for c in calls:
                        await send_call(c)
                await send({}, finish = finish)
                await resp.write(b"data: [DONE]\n\n")
                completed = (finish, ptoks, otoks, metrics)
                break
        await resp.write_eof()
        if completed is not None:
            finish, ptoks, otoks, metrics = completed
            _log_completion(cid, request_started, True, finish, ptoks, otoks, metrics)
    try:
        await run()
    except ConnectionResetError:
        log_request(cid, "disconnected", stream = True,
                    request_ms = round((time.perf_counter() - request_started) * 1000, 1))
    return resp


def main():
    global MODEL_DIR, DRAFT_DIR, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default = MODEL_DIR)
    ap.add_argument("-dm", "--draft_model", default = DRAFT_DIR,
                    help = "Draft model path, 'mtp' for MTP drafting (head inside the "
                           "main checkpoint: no extra weights, much smaller KV footprint) "
                           "or 'none' to disable drafting")
    ap.add_argument("-gs", "--grid_size", type = int, default = 110)
    ap.add_argument("-cs", "--cache_size", type = int, default = 65536,
                    help = "KV cache size in tokens (default 65536; 8192 default "
                           "of model_init is too small for large tool sets)")
    ap.add_argument("-cq", "--cache_quant", type = str, default = None,
                    help = "Quantized KV cache bits, e.g. 8 or 8,4 (k_bits[,v_bits])")
    ap.add_argument("-p", "--port", type = int, default = PORT)
    ap.add_argument("--host", type = str, default = "0.0.0.0",
                    help = "Interface to bind (use 127.0.0.1 for local-only)")
    ap.add_argument("-ccs", "--cpu_cache_size", type = float, default = 0.0,
                    help = "CPU second-tier cache size in GB (pages spill from "
                           "GPU when the GPU cache is full)")
    ap.add_argument("--max_body_mb", type = int, default = 64,
                    help = "max request body size in MiB (aiohttp's built-in "
                           "default is 1 MiB, far too small for a full tool "
                           "set + a long transcript)")
    ap.add_argument("--default_reasoning_effort",
                    choices = ("none",) + REASONING_EFFORTS, default = "low",
                    help = "default Qwen reasoning mode: none, low, medium, or xhigh")
    ap.add_argument("--default_preserve_thinking",
                    choices = ("true", "false"), default = "false",
                    help = "retain reasoning from earlier assistant turns by default")
    args = ap.parse_args()
    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model,
            "-gs", str(args.grid_size), "-cs", str(args.cache_size)]
    if use_mtp:
        argv += ["-mtp"]
    elif use_draft:
        argv += ["-dm", args.draft_model]
    if args.cache_quant:
        argv += ["-cq", args.cache_quant]
    if args.cpu_cache_size:
        argv += ["-ccs", str(args.cpu_cache_size)]

    print(f" == loading {args.model}"
          + (" + MTP head" if use_mtp else
             (f" + draft {args.draft_model}" if use_draft else " (no draft)"))
          + " ...", flush = True)
    generator, tokenizer = build_model(argv, use_draft = use_draft)
    stats["context_length"] = int(args.cache_size)
    print(" == model ready; accepting requests", flush = True)

    app = web.Application(client_max_size = args.max_body_mb * 1024 * 1024)
    app["generator"] = generator
    app["tokenizer"] = tokenizer
    effort = args.default_reasoning_effort
    app["chat_template_defaults"] = {
        "enable_thinking": effort != "none",
        "preserve_thinking": args.default_preserve_thinking == "true",
    }
    if effort != "none":
        app["chat_template_defaults"]["reasoning_effort"] = effort
    app["max_body_mb"] = args.max_body_mb
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    web.run_app(app, host = args.host, port = args.port, print = None)


if __name__ == "__main__":
    main()
