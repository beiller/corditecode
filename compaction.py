import json
import logging
import os
from datetime import datetime

import aiohttp

from conversation import CONVERSATION_DIR, approximate_token_count, archive_conversation
from main_types import Message, SessionID

logger = logging.getLogger()

THINKING_SESSION_MIN_TOKENS = 2000
THINKING_NOISE_FLOOR_CHARS = 100
THINKING_INPUT_MAX_CHARS = 150_000
TOOL_OUTPUT_MIN_CHARS = 700
CHUNK_MIN_CHARS = 700
USER_CHUNK_MIN_CHARS = 700
SUMMARY_SYSTEM_PROMPT = (
    "Summarize this thinking/reasoning section of a coding assistant. "
    "Preserve key decisions, findings, file paths, and next steps. Terse bullets, no prose."
)
THINKING_SESSION_SYSTEM_PROMPT = (
    "You are given all thinking/reasoning sections from a coding assistant session, in order. "
    "Produce ONE condensed summary of the whole session's reasoning: key decisions, findings, "
    "file paths, open questions, next steps. Terse bullets, max 15 bullets."
)


def _pct(env_var: str, default: float) -> float:
    raw = os.environ.get(env_var, str(default))
    try:
        pct = float(raw)
    except ValueError:
        logger.warning(f"Invalid {env_var}={raw!r}, using default {default:g}")
        return default
    if not 1 <= pct <= 100:
        logger.warning(f"{env_var}={pct} out of range 1-100, clamping")
        pct = min(max(pct, 1.0), 100.0)
    return pct


def compaction_trigger(context_size: int) -> int:
    """Auto-trigger threshold: compaction starts when context exceeds this (default: 70%)."""
    return int(context_size * _pct("COMPACTION_TRIGGER_PCT", 70.0) / 100)


def compaction_target(context_size: int) -> int:
    """Shrink-to target: stages stop once context is at or below this (default: 70%)."""
    return int(context_size * _pct("COMPACTION_TARGET_PCT", 70.0) / 100)


def context_token_count(messages: list[Message]) -> int:
    return approximate_token_count(json.dumps(messages))


async def summarize_text(
    base_url: str, text: str, model: str, system_prompt: str = SUMMARY_SYSTEM_PROMPT, max_tokens: int | None = None
) -> str | None:
    """Send a single text to the model for summarization; return the summary or None on failure."""
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "stream": True,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    try:
        parts: list[str] = []
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(sock_read=None, sock_connect=5 * 60, connect=5 * 60),
            ) as resp:
                async for line in resp.content:
                    raw_line = line.decode("utf-8").strip()
                    if not raw_line or raw_line.startswith(":"):
                        continue
                    if raw_line == "data: [DONE]":
                        break
                    if raw_line.startswith("data: "):
                        chunk = json.loads(raw_line[6:])
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            parts.append(content)
        return "".join(parts).strip() or None
    except Exception:
        logger.exception("summarize_text failed")
        return None


def offload_tool_round_trip(
    session_id: SessionID, name: str, call_id: str, arguments: str, result: Message | None
) -> str:
    """Offload a tool call round-trip (args + result) to one file; returns the path."""
    call_id = (call_id or "unknown")[:8]
    path = CONVERSATION_DIR / f"{session_id}_{name}_{call_id}.txt"
    parts = []
    if arguments:
        parts.append(f"## arguments\n{arguments}")
    if result is not None and result.get("content"):
        parts.append(f"## result\n{result['content']}")
    with open(path, "w") as fh:
        fh.write("\n\n".join(parts))
    return str(path)


def replace_tool_call_with_pointer(tc: dict, name: str, arguments: str, path: str) -> None:
    tc["function"]["arguments"] = json.dumps(
        {"note": f"[{name} args ({len(arguments)} chars) saved to {path} - use read_file to retrieve]"}
    )


def replace_tool_result_with_pointer(result: Message, name: str, path: str) -> None:
    result["content"] = (
        f"[{name} output ({len(result.get('content') or '')} chars) saved to {path} - use read_file or conversation_search to retrieve]"
    )


def next_chunk_number(session_id: SessionID) -> int:
    prefix = f"{session_id}_chunk_"
    nums = []
    if CONVERSATION_DIR.exists():
        for f in CONVERSATION_DIR.iterdir():
            n = f.name
            if n.startswith(prefix) and n.endswith(".txt"):
                try:
                    nums.append(int(n[len(prefix) : -4]))
                except ValueError:
                    pass
    return max(nums, default=0) + 1


def offload_chunk(session_id: SessionID, msg: Message) -> None:
    path = CONVERSATION_DIR / f"{session_id}_chunk_{next_chunk_number(session_id)}.txt"
    with open(path, "w") as fh:
        fh.write(msg["content"])
    msg["content"] = f"[context chunk saved to {path} - use read_file or conversation_search to retrieve]"


async def compact_messages(
    base_url: str, messages: list[Message], session_id: SessionID, context_size: int, model: str
) -> None:
    """Staged compaction: 1) summarize thinking, 2) offload tool outputs, 3) chunk non-user, 4) chunk user."""
    target = compaction_target(context_size)

    # Stage 1 - single whole-session thinking summary, swapped into the latest thinking message
    thinking_msgs = [
        m
        for m in messages
        if len(m.get("reasoning_content") or "") >= THINKING_NOISE_FLOOR_CHARS and not m.get("compacted_thinking")
    ]
    if (
        sum(len(m["reasoning_content"]) for m in thinking_msgs) // 4 >= THINKING_SESSION_MIN_TOKENS
        and context_token_count(messages) > target
    ):
        section_texts = [f"--- section {i + 1} ---\n{m['reasoning_content']}" for i, m in enumerate(thinking_msgs)]
        total_chars = sum(len(s) for s in section_texts)
        while total_chars > THINKING_INPUT_MAX_CHARS and len(section_texts) > 1:
            total_chars -= len(section_texts.pop(0))
        sections = "\n\n".join(section_texts)
        max_tokens = max(2000, total_chars // 4 // 10)
        summary = await summarize_text(
            base_url, sections, model, system_prompt=THINKING_SESSION_SYSTEM_PROMPT, max_tokens=max_tokens
        )
        if summary is None:
            summary = await summarize_text(base_url, sections, model, system_prompt=THINKING_SESSION_SYSTEM_PROMPT)
        if summary:
            host = thinking_msgs[-1]
            for m in thinking_msgs:
                m["reasoning_content"] = ""
            host["reasoning_content"] = summary
            host["compacted_thinking"] = True

    # Stage 2 - offload tool call round-trips (args + result) to files, largest pairs first
    tool_results = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}
    known_ids: set[str] = set()
    pairs: list[tuple[int, dict, Message | None]] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            call_id = tc.get("id", "")
            known_ids.add(call_id)
            args = tc.get("function", {}).get("arguments") or ""
            result = tool_results.get(call_id)
            size = len(args) + (len(result.get("content") or "") if result is not None else 0)
            if size >= TOOL_OUTPUT_MIN_CHARS:
                pairs.append((size, tc, result))
    for m in messages:
        if m.get("role") == "tool" and m.get("tool_call_id") not in known_ids and len(m.get("content") or "") >= TOOL_OUTPUT_MIN_CHARS:
            pairs.append((len(m["content"]), {"id": m.get("tool_call_id"), "function": {"name": "tool"}}, m))
    pairs.sort(key=lambda p: p[0], reverse=True)
    for _size, tc, result in pairs:
        if context_token_count(messages) <= target:
            break
        name = tc.get("function", {}).get("name", "tool")
        args = tc.get("function", {}).get("arguments") or ""
        path = offload_tool_round_trip(session_id, name, tc.get("id"), args, result)
        if args:
            replace_tool_call_with_pointer(tc, name, args, path)
        if result is not None and result.get("content"):
            replace_tool_result_with_pointer(result, name, path)

    # Stage 3 - chunk offload non-user messages, oldest first
    for m in messages:
        if context_token_count(messages) <= target:
            break
        if m.get("role") in ("user", "system"):
            continue
        if len(m.get("content") or "") < CHUNK_MIN_CHARS:
            continue
        offload_chunk(session_id, m)

    # Stage 4 - chunk offload user messages over the minimum, oldest first
    for m in messages:
        if context_token_count(messages) <= target:
            break
        if m.get("role") != "user":
            continue
        if len(m.get("content") or "") <= USER_CHUNK_MIN_CHARS:
            continue
        offload_chunk(session_id, m)

    # Final fallback - archive everything if still over budget
    # if context_token_count(messages) > target:
    #     filename = archive_conversation(session_id, messages)
    #     logger.warning(f"Compaction stages insufficient, archived to {filename}")
    #     new_messages = [
    #         {
    #             "role": "assistant",
    #             "content": f"The previous conversation was archived to {filename}",
    #             "timestamp": datetime.now().isoformat(),
    #         },
    #         *messages[-4:],
    #     ]
    #     messages.clear()
    #     messages.extend(new_messages)
