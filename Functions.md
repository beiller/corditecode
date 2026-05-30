# Available Tool Functions

**How to use:** These tools are invoked through plain language prompts — just describe what you want in natural English and the system will call them for you. No need to reference function names or parameters explicitly.

Examples:
- "list models for me please" → calls `list_models`
- "run bash sleep 10" → calls `run_bash(command="sleep 10")`
- "search huggingface for a Llama model with GGUF files" → calls `search_model(search_query=...)`

---

## Bash & System

### `run_bash`
Run a bash command and return its output. Use for executing shell commands, file operations, running scripts, etc.

**Parameters:**
- `command` (string): The bash command to execute

---

### `mux_skill`
Start a long-running command in a background tmux session. Useful for parallelizing work or running processes that take time without blocking. Output can be written to files in `/tmp/`.

**Parameters:**
- `command` (string): The command to run in the background

---

## Model Management

### `list_models`
List all available models configured in llama-server router mode. Returns model names, statuses (loaded/unloaded), and metadata like context size, parameter count, etc. Requires llama-server to be running in router mode.

**Parameters:** None

---

### `load_model`
Load a specific model into llama-server via the API. The server must be running in router mode (started without --model flag). The model must already exist on disk or be discoverable via preset configuration.

**Parameters:**
- `model_name` (string): Name of the model to load (e.g., `"Qwen_Qwen3.6-27B-Q4_K_M"`)

---

### `download_model`
Download a GGUF quantized model file from Hugging Face. Provide both the repo ID and filename for the specific file you want. Use `search_model` first if unsure which files are available.

**Parameters:**
- `repo_id` (string): The HuggingFace repository ID (e.g., `"Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"`)
- `filename` (string): The GGUF filename to download (e.g., `"q4_k_m.gguf"`)

---

### `search_model`
Search HuggingFace for models matching a query string. Returns a list of repositories with their IDs, descriptions, and file listings. Useful before downloading to find the right model.

**Parameters:**
- `search_query` (string): The search term or keyword
- `limit` (integer, optional): Maximum number of results to return

---

## Web & Network

### `curl_skill`
Fetch a webpage URL and return its contents as text. Use for retrieving web pages, APIs, documentation, etc. Pass the full curl command including any flags needed.

**Parameters:**
- `command` (string): The curl command to run (e.g., `"https://example.com"`)

---

## Conversation & Session Management

### `conversation_search`
Search through conversation history for a keyword or phrase. Returns matching snippets and the file path containing the relevant conversation. Useful for looking up past context, decisions, or code that was discussed previously.

**Parameters:**
- `keyword` (string): The term to search for in conversations

---

### `reset_session`
Reset the current session and clear chat history. Requires both a user ID and session ID to properly identify which conversation state to reset.

**Parameters:**
- `user_id` (string): The current user's identifier
- `session_id` (string): The active session identifier
