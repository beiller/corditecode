import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


def make_assistant_msg(name: str, args: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}],
    }


class TestExecuteToolCalls(unittest.TestCase):
    def test_sync_tool_handler(self):
        def mytool(x: str = "") -> str:
            return "sync-" + x

        results = asyncio.run(
            main.execute_tool_calls(make_assistant_msg("mytool", '{"x": "hi"}'), {"mytool": mytool}, "user1")
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["role"], "tool")
        self.assertEqual(results[1]["content"], "sync-hi")

    def test_async_tool_handler_is_awaited(self):
        async def mytool(x: str = "") -> str:
            await asyncio.sleep(0)
            return "async-" + x

        results = asyncio.run(
            main.execute_tool_calls(make_assistant_msg("mytool", '{"x": "hi"}'), {"mytool": mytool}, "user1")
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["role"], "tool")
        self.assertEqual(results[1]["content"], "async-hi")

    def test_compact_session_is_registered(self):
        tools, registry = main.build_tools()
        names = [t["function"]["name"] for t in tools]
        self.assertIn("compact_session", names)
        self.assertIs(registry["compact_session"], main.compact_session)


if __name__ == "__main__":
    unittest.main()
