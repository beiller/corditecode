import asyncio
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compaction
import conversation

REAL_CONVERSATION = pathlib.Path(__file__).parent.parent / "conversations" / "x9z2fvmh.json"


class CompactionBase(unittest.TestCase):
    def setUp(self):
        self.summarize_calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name)
        # Pin compaction env vars so tests are deterministic regardless of the ambient environment
        self.env_patcher = mock.patch.dict(
            os.environ, {"COMPACTION_TRIGGER_PCT": "70", "COMPACTION_TARGET_PCT": "70"}
        )
        self.env_patcher.start()
        self.patchers = [
            mock.patch.object(compaction, "CONVERSATION_DIR", self.path),
            mock.patch.object(conversation, "CONVERSATION_DIR", self.path),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        self.env_patcher.stop()
        self.tmp.cleanup()

    def run_compact(self, messages, context_size=8000, summarize=None):
        async def stub_summarize(base_url, text, model, **kwargs):
            self.summarize_calls.append(text)
            return "SUMMARY"

        with mock.patch.object(compaction, "summarize_text", summarize or stub_summarize):
            asyncio.run(compaction.compact_messages("http://localhost:1", messages, "sess", context_size, "test-model"))


class TestStages(CompactionBase):
    def test_stage1_single_call_summarizes_whole_session(self):
        big = {"role": "assistant", "content": None, "reasoning_content": "A" * 30000}
        small = {"role": "assistant", "content": None, "reasoning_content": "B" * 10000}
        messages = [big, small]
        self.run_compact(messages)
        # exactly one call, sections in chronological order
        self.assertEqual(len(self.summarize_calls), 1)
        self.assertIn("--- section 1 ---\n" + "A" * 30000, self.summarize_calls[0])
        self.assertIn("--- section 2 ---\n" + "B" * 10000, self.summarize_calls[0])
        # summary swapped into the latest thinking message, others cleared, host flagged
        self.assertEqual(big["reasoning_content"], "")
        self.assertEqual(small["reasoning_content"], "SUMMARY")
        self.assertTrue(small.get("compacted_thinking"))

    def test_stage1_no_call_when_total_thinking_under_2000_tokens(self):
        boundary = {"role": "assistant", "content": None, "reasoning_content": "B" * 2000}
        small = {"role": "assistant", "content": None, "reasoning_content": "S" * 1900}
        messages = [boundary, small]
        self.run_compact(messages, context_size=600)
        self.assertEqual(len(self.summarize_calls), 0)
        self.assertEqual(boundary["reasoning_content"], "B" * 2000)
        self.assertEqual(small["reasoning_content"], "S" * 1900)

    def test_stage1_flagged_summary_not_recompacted(self):
        host = {"role": "assistant", "content": None, "reasoning_content": "S" * 5000, "compacted_thinking": True}
        filler = {"role": "assistant", "content": "F" * 30000}
        messages = [host, filler]
        self.run_compact(messages)
        self.assertEqual(len(self.summarize_calls), 0)
        self.assertEqual(host["reasoning_content"], "S" * 5000)

    def test_stage1_summarize_failure_keeps_original_thinking(self):
        async def failing(base_url, text, model, **kwargs):
            return None

        msg = {"role": "assistant", "content": None, "reasoning_content": "A" * 30000}
        messages = [msg]
        self.run_compact(messages, summarize=failing)
        self.assertEqual(msg["reasoning_content"], "A" * 30000)

    def test_stage2_offloads_tool_outputs_largest_first(self):
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_abc12345xyz", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}}
            ],
        }
        big_tool = {"role": "tool", "tool_call_id": "call_abc12345xyz", "content": "C" * 60000}
        small_tool = {"role": "tool", "tool_call_id": "call_other123", "content": "D" * 3000}
        messages = [assistant, big_tool, small_tool]
        self.run_compact(messages)
        path = self.path / "sess_run_bash_call_abc.txt"
        self.assertTrue(path.exists())
        text = path.read_text()
        self.assertIn("## arguments\n{}", text)
        self.assertIn("## result\n" + "C" * 60000, text)
        # args side replaced with a valid JSON pointer to the same file
        new_args = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
        self.assertIn("saved to", new_args["note"])
        self.assertIn(str(path), new_args["note"])
        self.assertIn("saved to", big_tool["content"])
        self.assertIn(str(path), big_tool["content"])
        self.assertEqual(small_tool["content"], "D" * 3000)

    def test_stage2_offloads_args_and_result_to_one_file(self):
        args = json.dumps({"command": "echo " + "x" * 5000})
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_pair1234", "type": "function", "function": {"name": "run_bash", "arguments": args}}
            ],
        }
        result = {"role": "tool", "tool_call_id": "call_pair1234", "content": "R" * 500}
        messages = [assistant, result]
        self.run_compact(messages, context_size=1000)
        path = self.path / "sess_run_bash_call_pai.txt"
        self.assertTrue(path.exists())
        text = path.read_text()
        self.assertIn("## arguments\n" + args, text)
        self.assertIn("## result\n" + "R" * 500, text)
        new_args = json.loads(assistant["tool_calls"][0]["function"]["arguments"])
        self.assertIn("saved to", new_args["note"])
        self.assertIn(str(path), new_args["note"])
        self.assertIn(str(path), result["content"])

    def test_stage2_pair_under_threshold_untouched(self):
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_small123", "type": "function", "function": {"name": "run_bash", "arguments": json.dumps({"a": "b" * 300})}}
            ],
        }
        result = {"role": "tool", "tool_call_id": "call_small123", "content": "R" * 300}
        filler = {"role": "assistant", "content": "A" * 30000}
        messages = [assistant, result, filler]
        self.run_compact(messages)
        self.assertEqual(assistant["tool_calls"][0]["function"]["arguments"], json.dumps({"a": "b" * 300}))
        self.assertEqual(result["content"], "R" * 300)
        self.assertFalse([f for f in self.path.iterdir() if "run_bash" in f.name])

    def test_stage2_unknown_tool_call_id_uses_fallback_name(self):
        tool = {"role": "tool", "tool_call_id": "call_other123", "content": "D" * 30000}
        messages = [tool]
        self.run_compact(messages)
        path = self.path / "sess_tool_call_oth.txt"
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "## result\n" + "D" * 30000)

    def test_stage3_chunk_offload_oldest_first_skips_user(self):
        user = {"role": "user", "content": "U" * 100}
        old = {"role": "assistant", "content": "A" * 20000}
        recent = {"role": "assistant", "content": "B" * 20000}
        messages = [user, old, recent]
        self.run_compact(messages)
        path = self.path / "sess_chunk_1.txt"
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "A" * 20000)
        self.assertIn("saved to", old["content"])
        self.assertEqual(recent["content"], "B" * 20000)
        self.assertEqual(user["content"], "U" * 100)

    def test_stage4_offloads_large_user_messages(self):
        user = {"role": "user", "content": "U" * 30000}
        assistant = {"role": "assistant", "content": "a" * 800}
        messages = [user, assistant]
        self.run_compact(messages)
        chunk1 = self.path / "sess_chunk_1.txt"
        chunk2 = self.path / "sess_chunk_2.txt"
        self.assertTrue(chunk1.exists())
        self.assertTrue(chunk2.exists())
        self.assertEqual(chunk1.read_text(), "a" * 800)
        self.assertEqual(chunk2.read_text(), "U" * 30000)
        self.assertIn("saved to", user["content"])

    def test_no_compaction_when_under_target(self):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        self.run_compact(messages)
        self.assertEqual(messages[0]["content"], "hello")
        self.assertEqual(len(list(self.path.iterdir())), 0)

    @unittest.skip("fallback archive is commented out in compaction.py (triggers too often)")
    def test_fallback_archive_when_still_over_budget(self):
        messages = [{"role": "assistant", "content": "x" * 450} for _ in range(20)]
        self.run_compact(messages, context_size=2000)
        archived = [f for f in self.path.iterdir() if f.name.endswith("_sess.txt")]
        self.assertTrue(archived)
        self.assertEqual(len(messages), 5)
        self.assertTrue(messages[0]["content"].startswith("The previous conversation was archived to"))

    def test_none_content_messages_are_safe(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call_x", "content": None},
            {"role": "system", "content": None},
        ]
        self.run_compact(messages)
        for m in messages:
            self.assertIsNone(m["content"])


class TestThresholds(CompactionBase):
    def test_trigger_and_target_default_pct(self):
        self.assertEqual(compaction.compaction_trigger(1000), 700)
        self.assertEqual(compaction.compaction_target(1000), 700)

    def test_trigger_and_target_respect_env(self):
        with mock.patch.dict(os.environ, {"COMPACTION_TRIGGER_PCT": "80", "COMPACTION_TARGET_PCT": "20"}):
            self.assertEqual(compaction.compaction_trigger(1000), 800)
            self.assertEqual(compaction.compaction_target(1000), 200)

    def test_pct_invalid_falls_back_to_default(self):
        with mock.patch.dict(os.environ, {"COMPACTION_TRIGGER_PCT": "abc"}):
            self.assertEqual(compaction.compaction_trigger(1000), 700)

    def test_pct_out_of_range_is_clamped(self):
        with mock.patch.dict(os.environ, {"COMPACTION_TRIGGER_PCT": "0", "COMPACTION_TARGET_PCT": "150"}):
            self.assertEqual(compaction.compaction_trigger(1000), 10)
            self.assertEqual(compaction.compaction_target(1000), 1000)


class TestRealConversation(CompactionBase):
    def test_compact_real_4mb_conversation(self):
        with open(REAL_CONVERSATION) as f:
            messages = json.load(f)
        # 5 messages, one 3.28MB tool output; ~1M tokens at context_size=1M, target 700K
        self.assertGreater(compaction.context_token_count(messages), compaction.compaction_target(1_000_000))
        self.run_compact(messages, context_size=1_000_000)
        # No reasoning content in this file, so stage 1 must not call the model
        self.assertEqual(self.summarize_calls, [])
        # Stage 2 offloads the 3.28MB run_bash output (largest first), then stops under target
        offloaded = [f for f in self.path.iterdir() if f.name.endswith(".txt")]
        self.assertEqual(len(offloaded), 1)
        big_file = offloaded[0]
        self.assertIn("run_bash", big_file.name)
        self.assertGreater(big_file.stat().st_size, 3_000_000)
        # The big tool message is replaced with a pointer; the small one (3991 chars) is untouched
        big_tool = next(m for m in messages if "saved to" in (m.get("content") or ""))
        self.assertIn(str(big_file), big_tool["content"])
        small_tool = next(m for m in messages if m.get("role") == "tool" and "saved to" not in (m.get("content") or ""))
        self.assertEqual(len(small_tool["content"]), 3991)
        # No chunk offloads needed after stage 2
        self.assertFalse([f for f in self.path.iterdir() if "_chunk_" in f.name])
        self.assertLessEqual(compaction.context_token_count(messages), compaction.compaction_target(1_000_000))


class TestChunkNumbering(CompactionBase):
    def test_chunk_numbers_continue_across_compactions(self):
        def big_messages(tag):
            return [{"role": "assistant", "content": tag * 20000} for _ in range(3)]

        first = big_messages("A")
        self.run_compact(first)
        self.assertTrue((self.path / "sess_chunk_1.txt").exists())
        self.assertTrue((self.path / "sess_chunk_2.txt").exists())
        chunk1_content = (self.path / "sess_chunk_1.txt").read_text()

        second = big_messages("B")
        self.run_compact(second)
        new_chunks = sorted(f.name for f in self.path.iterdir() if "_chunk_" in f.name)
        self.assertIn("sess_chunk_3.txt", new_chunks)
        # Earlier chunks are not overwritten
        self.assertEqual((self.path / "sess_chunk_1.txt").read_text(), chunk1_content)


if __name__ == "__main__":
    unittest.main()
