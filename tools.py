#!/usr/bin/env python3
"""Tool implementations for the function-calling server."""

import json
import os
import subprocess
import urllib.request
from typing import List
import pathlib
from main_types import *
import re



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
    r"\brm\b", r"\brmdir\b", r"\bdd\b",
    r"\bmkfs\b", r"\bshred\b", r"\btruncate\b",
    r"\bwipe\b", r"\bmv\b.*(/dev|/sys|/proc)",
    r">\s*/dev/(s|h|v|x)d",   # redirect to disk devices
    r":\(\)\{.*\}",            # fork bomb
    r"\bchmod\s+[0-7]*0\b",    # removing all perms
    r"\bchown\b.*root",
    r"\bsudo\b", r"\bsu\b",
    r"\bcurl\b.*\|\s*(ba)?sh", # curl | bash
    r"\bwget\b.*-O.*\|\s*sh",
]
def is_safe(cmd: str) -> tuple[bool, str | None]:
    for pattern in BLOCKED:
        if re.search(pattern, cmd):
            return False, pattern
    return True, None

def run_bash(command: str, allow_unsafe_commands=False) -> str:
    """Run a shell command and return its output. 
- Do not allow unsafe commands unless specifically granted permission.
- Only if you have got explicit consent from the user, use `allow_unsafe_commands=True`
- This will run in bash, and the limit is ~10kb so try to limit verbose output
"""
    #MAX_RETURN_CHARS = 10_000
    if not allow_unsafe_commands and not is_safe(command):
        return {
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "error": f"Permission denied. Ask the user's permission to run this command, and if granted, call this function again using allow_unsafe_commands=True",
        }
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



def read_file(file_path: str, line_start: int|None = None, line_end: int|None = None) -> str:
    """Read a file. Use absolute file paths. Use this over bash. Use optional line_start and line_end to read a chunk of a file."""
    try:
        with open(file_path) as fh:
            return json.dumps({"status": "success", "result": "\n".join(fh.readlines()[line_start or 0:line_end or -1])})
    except Exception as e:
        return json.dumps({"status": "error", "description": str(e)})




def write_file(file_path: str, file_contents: str) -> str:
    """Write a file
- Use this instead of bash to avoid mangling text
- Use absolute file paths.
- BE CAUTIOUS WRITING FILES, check with the user first. 
"""
    try:
        with open(file_path, 'w') as fh:
            fh.write(file_contents)
        return '{"success": true}'
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
        data = json.dumps({"model": model_name}).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.dumps({"success": True, "model": model_name, "details": result})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else "Unknown error"
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
        data = json.dumps({"model": model_name}).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.dumps({"success": True, "model": model_name, "details": result})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else "Unknown error"
        return json.dumps({"success": False, "http_status": e.code, "error": error_body})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


def get_loaded_model() -> str | None:
    """Return the loaded model name, or None if no model is currently loaded."""
    base_url = os.getenv("LLAMA_BASE_URL")
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
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
        
        req = urllib.request.Request(
            url,
            method="GET"
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})



def load_skills(
    skills_dir: str | pathlib.Path
) -> tuple[list[Tool], dict[str, ToolHandler]]:
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


from search import search as _search

def conversation_search(
    keyword: str
) -> str:
    """Searches for a keyword and returns snippets and the relevant file that contians the conversation in question. For the full context, and only if needed, read the file"""
    return _search(keyword)

