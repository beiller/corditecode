import asyncio
import logging
from typing import Callable

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widgets import Input, Markdown, Static

logger = logging.getLogger()

queue_query = None
get_response = None
set_interrupt = None
usage_events = None
turn_events = None
context_size = 16384
app = None
_shutdown_event = asyncio.Event()


MOON_PHASES = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]


class ChatApp(App):
    STATE_STREAMING = "streaming"
    STATE_IDLE = "idle"
    CSS = """
    #stats_bar {
        border: solid $accent;
        background: $surface;
        padding: 0 1;
        color: $text-muted;
    }
    """

    async def on_mount(self):
        # This ensures the input is ready for typing immediately
        self.query_one(Input).focus()

        container = self.query_one("#chat_container")
        await container.mount(Static("[bold cyan]⟡  C H A T[/]"))
        await container.mount(Static("[grey]Local AI Assistant[/]"))

        self.state = ChatApp.STATE_IDLE
        self.busy = False
        self.done = False
        self._frame_idx = 0
        self._pending: set[str] = set()
        self._stats_text = f"tokens: - / {context_size}"

        self.set_interval(0.2, self.tick_spinner)
        self.run_worker(write_output())

    def compose(self) -> ComposeResult:
        # Use a scrollable container for the chat history
        yield VerticalScroll(id="chat_container")
        yield Static(f"tokens: - / {context_size}", id="stats_bar")
        yield Input(placeholder="Type here...")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            return
        container = self.query_one("#chat_container")
        await container.mount(Static(f"[bold cyan]You:[/][white] {escape(event.value)}[/]"))
        container.scroll_end(animate=False)
        msg_id = await queue_query("0", event.value)
        self._pending.add(msg_id)
        self.busy = True
        self.done = False
        self._frame_idx = 0
        self.render_bar()
        event.input.value = ""

    async def debug(self, text):
        container = self.query_one("#chat_container")
        await container.mount(Static(text))
        container.scroll_end(animate=False)

    async def create_container(self, markdown=False):
        container = self.query_one("#chat_container")
        widget = Markdown("") if markdown else Static("")
        await container.mount(widget)
        self.current_response = widget

    def key_control_c(self, event: Key) -> None:
        event.stop()
        event.prevent_default()
        on_ctrl_c()

    def render_bar(self) -> None:
        if self.busy:
            frame = MOON_PHASES[self._frame_idx % len(MOON_PHASES)]
            self._frame_idx += 1
            prefix = f"{frame} "
        elif self.done:
            prefix = "✅ "
        else:
            prefix = "  "
        self.query_one("#stats_bar").update(f"{prefix}{self._stats_text}")

    def tick_spinner(self) -> None:
        if self.busy:
            self.render_bar()

    async def turn_done(self, msg_id: str) -> None:
        self._pending.discard(msg_id)
        if self._pending:
            return
        self.busy = False
        self.done = True
        self.render_bar()

    async def update_stats(self, usage: dict) -> None:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt_tokens + completion_tokens)
        ctx = context_size or 16384
        pct = (total / ctx * 100) if ctx else 0.0
        self._stats_text = (
            f"tokens: {total} / {ctx} ({pct:.1f}%) | prompt: {prompt_tokens} completion: {completion_tokens}"
        )
        self.render_bar()

    async def on_llm_text(self, user_id, role, token, chunk):
        """Now you can append tokens without a newline!"""
        if self.state == ChatApp.STATE_IDLE:
            if chunk:
                self.state = ChatApp.STATE_STREAMING
                self.current_text = ""
                await self.create_container(markdown=True)
            else:
                await self.create_container(markdown=False)

        if not chunk and token == "":
            # Per-LLM-call end marker (fires between tool calls too) - reset streaming state only.
            # The checkmark is set by the turn event, which fires once per user turn.
            self.state = ChatApp.STATE_IDLE
            return

        if self.state == ChatApp.STATE_STREAMING:
            self.current_text += token
            new_content = self.current_text
        else:
            new_content = f"[bold magenta]{role} 🤖:[/][white] {escape(token)}[/]"

        self.current_response.update(new_content)
        container = self.query_one("#chat_container")
        container.scroll_end(animate=False)

    # async def on_paste(self, event: Paste) -> None:
    #     file_path = event.text.strip().replace("file://", "")
    #     await self.debug(file_path)
    #     return


async def write_output():
    """Write output responses. Handles shutdown gracefully."""
    global app
    logger.info("Starting output writer...")
    try:
        while not _shutdown_event.is_set():
            try:
                # Use a timeout to check shutdown event periodically
                user_id, role, token, chunk = await asyncio.wait_for(get_response(), timeout=0.5)
                asyncio.run_coroutine_threadsafe(app.on_llm_text(user_id, role, token, chunk), asyncio.get_event_loop())
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                logger.info("Output writer cancelled")
                break

            # Drain any usage events that arrived (non-blocking)
            while usage_events is not None:
                try:
                    user_id, usage = usage_events.get_nowait()
                    asyncio.run_coroutine_threadsafe(app.update_stats(usage), asyncio.get_event_loop())
                except asyncio.QueueEmpty:
                    break

            # Drain turn-end events (non-blocking) - one per user message, after the full tool loop
            while turn_events is not None:
                try:
                    msg_id = turn_events.get_nowait()
                    asyncio.run_coroutine_threadsafe(app.turn_done(msg_id), asyncio.get_event_loop())
                except asyncio.QueueEmpty:
                    break

            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"Error in write_output: {e}")
        raise


def on_ready():
    # print_prompt()
    pass


def stop():
    """Signal shutdown and clean up terminal state."""
    logger.info("Stopping terminal...")
    _shutdown_event.set()


def on_ctrl_c():
    global queue_query, get_response, set_interrupt
    # print("\nCtrl+C ignored, still running. Use kill or another method to stop.")
    asyncio.run_coroutine_threadsafe(set_interrupt("0"), asyncio.get_running_loop())


async def run_ui():
    global app
    app = ChatApp()
    await app.run_async()


async def init(_queue_query: Callable, _get_response: Callable, _set_interrupt: Callable):
    """Initialize the terminal client and start I/O tasks."""
    global queue_query, get_response, set_interrupt
    queue_query = _queue_query
    get_response = _get_response
    set_interrupt = _set_interrupt

    # Start both tasks with proper cancellation handling
    try:
        await run_ui()
    except asyncio.CancelledError:
        logger.info("Terminal init cancelled")
        stop()
        raise
