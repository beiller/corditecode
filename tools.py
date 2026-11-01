#!/usr/bin/env python3
"""Tool implementations for the function-calling server."""

import difflib
import json
import os
import pathlib
import re
import subprocess
import urllib.request

from main_types import FunctionParameters, Tool, ToolHandler
from search import search as _search


def make_tool(
    name: str,
    description: str,
    parameters: FunctionParameters,
) -> Tool:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
BLOCKED = [
    r"\brm\b",
    r"\brmdir\b",
    r"\bdd\b",
    r"\bmkfs\b",
    r"\bshred\b",
    r"\btruncate\b",
    r"\bwipe\b",
    r"\bmv\b.*(/dev|/sys|/proc)",
    r">\s*/dev/(s|h|v|x)d",  # redirect to disk devices
    r":\(\)\{.*\}",  # fork bomb
    r"\bchmod\s+[0-7]*0\b",  # removing all perms
    r"\bchown\b.*root",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bcurl\b.*\|\s*(ba)?sh",  # curl | bash
    r"\bwget\b.*-O.*\|\s*sh",
]


def is_safe(cmd: str) -> tuple[bool, str | None]:
    for pattern in BLOCKED:
        if re.search(pattern, cmd):
            return False, pattern
    return True, None


def run_bash(command: str, allow_unsafe_commands: bool = False) -> str:
    """Run a shell command and return its output.
    - Do not allow unsafe commands unless specifically granted permission.
    - Only if you have got explicit consent from the user, use `allow_unsafe_commands=True`
    - This will run in bash, and the limit is ~10kb so try to limit verbose output
    """
    # MAX_RETURN_CHARS = 10_000
    safe, blocked_pattern = is_safe(command)
    if not allow_unsafe_commands and not safe:
        return json.dumps(
            {
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "error": f"Permission denied (blocked pattern: {blocked_pattern}). Ask the user's permission to run this command, and if granted, call this function again using allow_unsafe_commands=True",
            }
        )
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        return json.dumps(
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "error": f"Command execution failed: {str(e)}",
            }
        )


def change_working_directory(directory: str) -> str:
    """Change the process working directory. Relative paths are resolved against the current working directory.
    - The target must be an existing directory.
    - Affects relative path resolution for all subsequent file tools and bash commands.
    """
    target = directory if os.path.isabs(directory) else os.path.abspath(directory)
    try:
        if not os.path.isdir(target):
            return json.dumps({"success": False, "error": f"Not a directory: {target}"})
        os.chdir(target)
        return json.dumps({"success": True, "working_directory": os.getcwd()})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


DEFAULT_READ_LIMIT = 2000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 50 * 1024
SAMPLE_BYTES = 4096
BINARY_EXTENSIONS = {
    ".zip",
    ".tar",
    ".gz",
    ".exe",
    ".dll",
    ".so",
    ".class",
    ".jar",
    ".war",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".bin",
    ".dat",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".wasm",
    ".pyc",
    ".pyo",
}


def _is_binary_file(filepath: str, sample: bytes) -> bool:
    if os.path.splitext(filepath)[1].lower() in BINARY_EXTENSIONS:
        return True
    if not sample:
        return False
    non_printable = 0
    for b in sample:
        if b == 0:
            return True
        if b < 9 or (b > 13 and b < 32):
            non_printable += 1
    return non_printable / len(sample) > 0.3


def _suggest_files(filepath: str) -> list[str]:
    directory, base = os.path.split(filepath)
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    matches = [e for e in entries if base.lower() in e.lower() or e.lower() in base.lower()]
    return [os.path.join(directory, e) for e in sorted(matches)[:3]]


def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a file or directory from the local filesystem. If the path does not exist, an error is returned.

    Usage:
    - The file_path parameter should be an absolute path (relative paths are resolved against the working directory).
    - By default, this tool returns up to 2000 lines from the start of the file.
    - The offset parameter is the line number to start from (1-indexed).
    - To read later sections, call this tool again with a larger offset.
    - Contents are returned with each line prefixed by its line number as `<line>: <content>`. For directories, entries are returned one per line with a trailing `/` for subdirectories.
    - Any line longer than 2000 characters is truncated. Output is capped at 50 KB.
    - Call this tool in parallel when you know there are multiple files you want to read.
    - Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window.
    """
    filepath = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)

    if not os.path.exists(filepath):
        suggestions = _suggest_files(filepath)
        msg = f"File not found: {filepath}"
        if suggestions:
            msg += "\n\nDid you mean one of these?\n" + "\n".join(suggestions)
        return json.dumps({"status": "error", "description": msg})

    if offset is not None:
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            return json.dumps({"status": "error", "description": f"Invalid offset: {offset!r}"})
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return json.dumps({"status": "error", "description": f"Invalid limit: {limit!r}"})
    start = (offset or 1) - 1
    count_limit = limit or DEFAULT_READ_LIMIT

    if os.path.isdir(filepath):
        try:
            entries = sorted(os.listdir(filepath))
        except OSError as e:
            return json.dumps({"status": "error", "description": str(e)})
        listed = []
        for name in entries:
            full = os.path.join(filepath, name)
            if os.path.isdir(full):
                listed.append(name + "/")
            else:
                listed.append(name)
        sliced = listed[start : start + count_limit]
        truncated = start + len(sliced) < len(listed)
        notice = (
            f"\n(Showing {len(sliced)} of {len(listed)} entries. Use offset={start + len(sliced) + 1} to read beyond entry {(offset or 1) + len(sliced)})"
            if truncated
            else f"\n({len(listed)} entries)"
        )
        return json.dumps(
            {
                "status": "success",
                "result": f"<path>{filepath}</path>\n<type>directory</type>\n<entries>\n"
                + "\n".join(sliced)
                + notice
                + "\n</entries>",
            }
        )

    try:
        with open(filepath, "rb") as fh:
            sample = fh.read(SAMPLE_BYTES)
    except OSError as e:
        return json.dumps({"status": "error", "description": str(e)})

    if _is_binary_file(filepath, sample):
        return json.dumps({"status": "error", "description": f"Cannot read binary file: {filepath}"})

    raw = []
    total_lines = 0
    more = False
    cut = False
    done = False
    bytes_used = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for text in fh:
                if done:
                    break
                line = text.rstrip("\n")
                total_lines += 1
                if total_lines <= start:
                    continue
                if len(raw) >= count_limit:
                    more = True
                    continue
                if len(line) > MAX_LINE_LENGTH:
                    line = line[:MAX_LINE_LENGTH] + f"... (line truncated to {MAX_LINE_LENGTH} chars)"
                size = len(line.encode("utf-8")) + (1 if raw else 0)
                if bytes_used + size <= MAX_BYTES:
                    raw.append(line)
                    bytes_used += size
                else:
                    cut = True
                    more = True
                    done = True
    except OSError as e:
        return json.dumps({"status": "error", "description": str(e)})

    if total_lines < (offset or 1) and not (total_lines == 0 and (offset or 1) == 1):
        return json.dumps(
            {"status": "error", "description": f"Offset {offset} is out of range for this file ({total_lines} lines)"}
        )

    output = [f"<path>{filepath}</path>", "<type>file</type>", "<content>"]
    output += [f"{i + (offset or 1)}: {line}" for i, line in enumerate(raw)]

    last = (offset or 1) + len(raw) - 1
    next_offset = last + 1
    if cut:
        output.append(
            f"\n\n(Output capped at {MAX_BYTES // 1024} KB. Showing lines {(offset or 1)}-{last}. Use offset={next_offset} to continue.)"
        )
    elif more:
        output.append(
            f"\n\n(Showing lines {(offset or 1)}-{last} of {total_lines}. Use offset={next_offset} to continue.)"
        )
    else:
        output.append(f"\n\n(End of file - total {total_lines} lines)")
    output.append("</content>")

    return json.dumps({"status": "success", "result": "\n".join(output)})


def write_file(file_path: str, file_contents: str) -> str:
    """Writes a file to the local filesystem.

    Usage:
    - This tool will overwrite the existing file if there is one at the provided path.
    - If this is an existing file, you MUST use the Read tool first to read the file's contents.
    - ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
    - NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
    - Use absolute file paths (relative paths are resolved against the working directory).
    """
    filepath = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)
    try:
        exists = os.path.exists(filepath)
        source_bom = False
        content_old = ""
        if exists:
            with open(filepath, "rb") as fh:
                raw_bytes = fh.read()
            for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
                if raw_bytes.startswith(bom):
                    source_bom = True
                    content_old = raw_bytes[len(bom) :].decode("utf-8", errors="replace")
                    break
            if not source_bom:
                content_old = raw_bytes.decode("utf-8", errors="replace")

        desired_bom = source_bom or file_contents.startswith("\ufeff")
        content_new = file_contents[1:] if (desired_bom and file_contents.startswith("\ufeff")) else file_contents

        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as fh:
            if desired_bom:
                fh.write("\ufeff")
            fh.write(content_new)

        diff = "".join(
            difflib.unified_diff(
                content_old.splitlines(keepends=True),
                content_new.splitlines(keepends=True),
                fromfile=filepath,
                tofile=filepath,
                n=3,
            )
        )

        return json.dumps({"success": True, "diff": diff})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def _detect_line_ending(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n")


def _convert_to_line_ending(text: str, ending: str) -> str:
    return text if ending == "\n" else text.replace("\n", ending)


def _line_trimmed_matches(content: str, find: str):
    original_lines = content.split("\n")
    search_lines = find.split("\n")
    if search_lines and search_lines[-1] == "":
        search_lines.pop()
    if not search_lines:
        return
    for i in range(len(original_lines) - len(search_lines) + 1):
        if all(original_lines[i + j].strip() == search_lines[j].strip() for j in range(len(search_lines))):
            start = sum(len(line) + 1 for line in original_lines[:i])
            end = start + sum(
                len(original_lines[i + k]) + (1 if k < len(search_lines) - 1 else 0) for k in range(len(search_lines))
            )
            yield content[start:end]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def _whitespace_normalized_matches(content: str, find: str):
    normalized_find = _norm(find)
    if not normalized_find:
        return
    lines = content.split("\n")
    for line in lines:
        normalized_line = _norm(line)
        if normalized_line == normalized_find:
            yield line
        elif normalized_find in normalized_line:
            words = find.strip().split()
            if words:
                pattern = r"\s+".join(re.escape(w) for w in words)
                m = re.search(pattern, line)
                if m:
                    yield m.group(0)
    find_lines = find.split("\n")
    if len(find_lines) > 1:
        for i in range(len(lines) - len(find_lines) + 1):
            block = "\n".join(lines[i : i + len(find_lines)])
            if _norm(block) == normalized_find:
                yield block


def _indentation_flexible_matches(content: str, find: str):
    def deindent(text: str) -> str:
        lines = text.split("\n")
        nonempty = [line for line in lines if line.strip()]
        if not nonempty:
            return text
        min_indent = min(len(line) - len(line.lstrip()) for line in nonempty)
        return "\n".join(line[min_indent:] if line.strip() else line for line in lines)

    normalized_find = deindent(find)
    content_lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(content_lines) - len(find_lines) + 1):
        block = "\n".join(content_lines[i : i + len(find_lines)])
        if deindent(block) == normalized_find:
            yield block


def _trimmed_boundary_matches(content: str, find: str):
    trimmed_find = find.strip()
    if trimmed_find == find or not trimmed_find:
        return
    if trimmed_find in content:
        yield trimmed_find
    lines = content.split("\n")
    find_lines = find.split("\n")
    for i in range(len(lines) - len(find_lines) + 1):
        block = "\n".join(lines[i : i + len(find_lines)])
        if block.strip() == trimmed_find:
            yield block


def _replace_in_content(content: str, old_string: str, new_string: str, replace_all: bool) -> str:
    not_found = True
    replacers = (
        lambda c, f: [f],
        _line_trimmed_matches,
        _whitespace_normalized_matches,
        _indentation_flexible_matches,
        _trimmed_boundary_matches,
    )
    for replacer in replacers:
        for candidate in replacer(content, old_string):
            idx = content.find(candidate)
            if idx == -1:
                continue
            not_found = False
            if replace_all:
                return content.replace(candidate, new_string)
            if content.rfind(candidate) != idx:
                continue
            return content[:idx] + new_string + content[idx + len(candidate) :]
    if not_found:
        raise ValueError(
            "Could not find old_string in the file. It must match exactly, including whitespace, indentation, and line endings."
        )
    raise ValueError(
        "Found multiple matches for old_string. Provide more surrounding context to make the match unique, or use replace_all."
    )


def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Performs exact string replacements in files.

    Usage:
    - When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: line number + colon + space. Everything after that space is the actual file content to match. Never include any part of the line number prefix in old_string or new_string.
    - ALWAYS prefer editing existing files in the codebase. NEVER write new files unless explicitly required.
    - The edit will FAIL if old_string is not found in the file with an error "Could not find old_string in the file".
    - The edit will FAIL if old_string is found multiple times in the file with an error "Found multiple matches for old_string. Provide more surrounding context to make the match unique." Either provide a larger string with more surrounding context to make it unique or use replace_all to change every instance.
    - Use replace_all for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.
    """
    filepath = file_path if os.path.isabs(file_path) else os.path.abspath(file_path)
    try:
        if old_string == new_string:
            return json.dumps(
                {"success": False, "error": "No changes to apply: old_string and new_string are identical."}
            )
        if not os.path.exists(filepath):
            return json.dumps({"success": False, "error": f"File {filepath} not found"})
        if os.path.isdir(filepath):
            return json.dumps({"success": False, "error": f"Path is a directory, not a file: {filepath}"})

        with open(filepath, "rb") as fh:
            raw_bytes = fh.read()
        source_bom = False
        content_old = ""
        for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
            if raw_bytes.startswith(bom):
                source_bom = True
                content_old = raw_bytes[len(bom) :].decode("utf-8", errors="replace")
                break
        if not source_bom:
            content_old = raw_bytes.decode("utf-8", errors="replace")

        desired_bom = source_bom or new_string.startswith("\ufeff")
        if old_string == "":
            content_new = new_string[1:] if (desired_bom and new_string.startswith("\ufeff")) else new_string
        else:
            ending = _detect_line_ending(content_old)
            old = _convert_to_line_ending(_normalize_line_endings(old_string), ending)
            replacement = _convert_to_line_ending(_normalize_line_endings(new_string), ending)
            try:
                content_new = _replace_in_content(content_old, old, replacement, bool(replace_all))
            except ValueError as e:
                return json.dumps({"success": False, "error": str(e)})

        with open(filepath, "w", encoding="utf-8") as fh:
            if desired_bom:
                fh.write("\ufeff")
            fh.write(content_new)

        diff = "".join(
            difflib.unified_diff(
                _normalize_line_endings(content_old).splitlines(keepends=True),
                _normalize_line_endings(content_new).splitlines(keepends=True),
                fromfile=filepath,
                tofile=filepath,
                n=3,
            )
        )

        additions = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))

        return json.dumps({"success": True, "diff": diff, "additions": additions, "deletions": deletions})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def load_model(model_name: str) -> str:
    """Load a specific model into llama-server router mode via its API.

    Requires llama-server to be running in router mode (started without --model flag).
    The model must be discoverable via --models-dir or preset configuration.

    Args:
        model_name: Name or path of the model to load

    Returns:
        JSON string with success status and details
    """
    base_url = os.getenv("LLAMA_BASE_URL")

    try:
        url = f"{base_url}/models/load"
        data = json.dumps({"model": model_name}).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
            return json.dumps({"success": True, "model": model_name, "details": result})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else "Unknown error"
        return json.dumps({"success": False, "http_status": e.code, "error": error_body})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def unload_model(model_name: str) -> str:
    """Unload a specific model from llama-server router mode via its API.

    Requires llama-server to be running in router mode.

    Args:
        model_name: Name or path of the model to unload

    Returns:
        JSON string with success status and details
    """
    base_url = os.getenv("LLAMA_BASE_URL")
    try:
        url = f"{base_url}/models/unload"
        data = json.dumps({"model": model_name}).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return json.dumps({"success": True, "model": model_name, "details": result})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else "Unknown error"
        return json.dumps({"success": False, "http_status": e.code, "error": error_body})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def get_loaded_model() -> str | None:
    """Return the loaded model name, or None if no model is currently loaded."""
    base_url = os.getenv("LLAMA_BASE_URL")
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            for model in result.get("data", []):
                if model.get("status", {}).get("value") == "loaded":
                    return model["id"]
            return None
    except Exception:
        return None


def list_models() -> str:
    """List all available models in llama-server router mode.

    Requires llama-server to be running in router mode.
    Returns model names, statuses, and metadata.

    Returns:
        JSON string with list of models and their status
    """
    base_url = os.getenv("LLAMA_BASE_URL")
    try:
        url = f"{base_url}/models"

        req = urllib.request.Request(url, method="GET")

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def load_skills(skills_dir: str | pathlib.Path) -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Scan skills_dir for .md files and return (tools, registry) for each."""
    skills_dir = pathlib.Path(skills_dir)

    tools = []
    registry = {}
    if not skills_dir.is_dir():
        return tools, registry

    for md_file in sorted(skills_dir.glob("*.md")):
        text = md_file.read_text()
        lines = text.splitlines()
        description = lines[0].strip() if lines else md_file.stem
        body = "\n".join(lines[1:]).strip()
        skill_name = md_file.stem + "_skill"

        tool = make_tool(
            name=skill_name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                },
                "required": ["command"],
            },
        )

        def _make_handler(content: str):
            def handler(*, command: str) -> str:
                return content

            return handler

        tools.append(tool)
        registry[skill_name] = _make_handler(body)

    return tools, registry


def conversation_search(keyword: str) -> str:
    """Searches for a keyword and returns snippets and the relevant file that contians the conversation in question. For the full context, and only if needed, read the file"""
    return _search(keyword)
