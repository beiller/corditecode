import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import edit_file, read_file, write_file


def ok(result: str) -> dict:
    return json.loads(result)


class FileToolsBase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "sample.txt")

    def write_raw(self, content: str, bom: bytes = b""):
        with open(self.path, "wb") as f:
            f.write(bom + content.encode("utf-8"))

    def read_raw(self) -> str:
        with open(self.path, "rb") as f:
            data = f.read()
        for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
            if data.startswith(bom):
                return data[len(bom) :].decode("utf-8")
        return data.decode("utf-8")


class TestReadFile(FileToolsBase):
    def test_basic_read(self):
        self.write_raw("hello\nworld\n")
        r = ok(read_file(self.path))
        self.assertEqual(r["status"], "success")
        self.assertIn("1: hello", r["result"])
        self.assertIn("2: world", r["result"])

    def test_offset_and_limit(self):
        self.write_raw("one\ntwo\nthree\nfour\n")
        r = ok(read_file(self.path, offset=2, limit=2))
        self.assertIn("2: two", r["result"])
        self.assertIn("3: three", r["result"])
        self.assertNotIn("4: four", r["result"])
        self.assertIn("Use offset=4 to continue", r["result"])

    def test_offset_out_of_range(self):
        self.write_raw("one\n")
        r = ok(read_file(self.path, offset=99))
        self.assertEqual(r["status"], "error")
        self.assertIn("out of range", r["description"])

    def test_missing_file_with_suggestion(self):
        self.write_raw("x\n")
        r = ok(read_file(self.path + ".extra"))
        self.assertEqual(r["status"], "error")
        self.assertIn("Did you mean", r["description"])
        self.assertIn(self.path, r["description"])

    def test_directory_listing(self):
        os.makedirs(os.path.join(self.dir.name, "subdir"))
        with open(os.path.join(self.dir.name, "a.txt"), "w") as f:
            f.write("x\n")
        r = ok(read_file(self.dir.name))
        self.assertIn("subdir/", r["result"])
        self.assertIn("a.txt", r["result"])

    def test_binary_rejected(self):
        with open(os.path.join(self.dir.name, "blob.bin"), "wb") as f:
            f.write(b"\x00\x01\x02data")
        r = ok(read_file(os.path.join(self.dir.name, "blob.bin")))
        self.assertEqual(r["status"], "error")
        self.assertIn("binary", r["description"].lower())

    def test_long_line_truncated(self):
        self.write_raw("start\n" + "x" * 2500 + "\nend\n")
        r = ok(read_file(self.path))
        self.assertIn("truncated to 2000 chars", r["result"])

    def test_emoji_and_special_chars_roundtrip(self):
        content = "héllo wörld\n日本語のテキスト\n🎉🚀💯 emoji line\na\u0301combining\ufeffbom \u200bzwb\n"
        self.write_raw(content)
        r = ok(read_file(self.path))
        self.assertEqual(r["status"], "success")
        for line in ("héllo wörld", "日本語のテキスト", "🎉🚀💯 emoji line"):
            self.assertIn(line, r["result"])

    def test_crlf_content_read(self):
        with open(self.path, "wb") as f:
            f.write(b"line one\r\nline two\r\n")
        r = ok(read_file(self.path))
        self.assertEqual(r["status"], "success")
        self.assertIn("1: line one", r["result"])


class TestWriteFile(FileToolsBase):
    def test_write_new_file(self):
        r = ok(write_file(self.path, "hello\n"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "hello\n")

    def test_creates_parent_dirs(self):
        nested = os.path.join(self.dir.name, "a", "b", "c.txt")
        r = ok(write_file(nested, "deep\n"))
        self.assertTrue(r["success"])
        with open(nested) as f:
            self.assertEqual(f.read(), "deep\n")

    def test_overwrite_produces_diff(self):
        self.write_raw("old line\n")
        r = ok(write_file(self.path, "new line\n"))
        self.assertTrue(r["success"])
        self.assertIn("-old line", r["diff"])
        self.assertIn("+new line", r["diff"])

    def test_emoji_and_special_chars_roundtrip(self):
        content = "🎉 emoji start\ncafé crème naïve\n日本語 + 中文 + 한국어\n"
        ok(write_file(self.path, content))
        self.assertEqual(self.read_raw(), content)

    def test_special_control_and_unicode_edge_chars(self):
        content = "tab\there\nnull-ish \x0bvertical tab\nzero-width\u200bjoiner\nsurrogate-adjacent \U0001f90a\n"
        ok(write_file(self.path, content))
        self.assertEqual(self.read_raw(), content)

    def test_bom_preserved_on_overwrite(self):
        self.write_raw("old\n", bom=b"\xef\xbb\xbf")
        ok(write_file(self.path, "new\n"))
        with open(self.path, "rb") as f:
            self.assertTrue(f.read().startswith(b"\xef\xbb\xbf"))

    def test_bom_added_when_content_has_bom(self):
        ok(write_file(self.path, "\ufeffcontent\n"))
        with open(self.path, "rb") as f:
            self.assertTrue(f.read().startswith(b"\xef\xbb\xbf"))


class TestEditFile(FileToolsBase):
    def test_exact_match(self):
        self.write_raw("alpha\nbeta\ngamma\n")
        r = ok(edit_file(self.path, "beta", "BETA"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "alpha\nBETA\ngamma\n")
        self.assertIn("-beta", r["diff"])
        self.assertIn("+BETA", r["diff"])

    def test_multiple_matches_fails(self):
        self.write_raw("dup\ndup\n")
        r = ok(edit_file(self.path, "dup", "x"))
        self.assertFalse(r["success"])
        self.assertIn("multiple", r["error"].lower())

    def test_replace_all(self):
        self.write_raw("foo\nfoo\nfoo\n")
        r = ok(edit_file(self.path, "foo", "bar", True))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "bar\nbar\nbar\n")

    def test_not_found(self):
        self.write_raw("present\n")
        r = ok(edit_file(self.path, "absent", "x"))
        self.assertFalse(r["success"])
        self.assertIn("could not find", r["error"].lower())

    def test_identical_strings_rejected(self):
        self.write_raw("same\n")
        r = ok(edit_file(self.path, "same", "same"))
        self.assertFalse(r["success"])
        self.assertIn("identical", r["error"].lower())

    def test_missing_file(self):
        r = ok(edit_file(os.path.join(self.dir.name, "nope.txt"), "a", "b"))
        self.assertFalse(r["success"])
        self.assertIn("not found", r["error"].lower())

    def test_directory_rejected(self):
        r = ok(edit_file(self.dir.name, "a", "b"))
        self.assertFalse(r["success"])
        self.assertIn("directory", r["error"].lower())

    def test_empty_old_string_replaces_whole_file(self):
        self.write_raw("old content\nmore\n")
        r = ok(edit_file(self.path, "", "brand new\n"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "brand new\n")

    def test_multiline_match_with_context(self):
        self.write_raw("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        r = ok(edit_file(self.path, "def bar():\n    return 2", "def bar():\n    return 20"))
        self.assertTrue(r["success"])
        self.assertIn("return 20", self.read_raw())
        self.assertNotIn("return 2\n", self.read_raw().replace("return 20\n", ""))

    def test_indentation_flexible_match(self):
        self.write_raw("class A:\n    def m(self):\n        return 1\n")
        r = ok(edit_file(self.path, "def m(self):\n      return 1", "def m(self):\n        return 99"))
        self.assertTrue(r["success"])
        self.assertIn("return 99", self.read_raw())

    def test_whitespace_normalized_match(self):
        self.write_raw("int  x   =  5\n")
        r = ok(edit_file(self.path, "int x = 5", "int y = 6"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "int y = 6\n")

    def test_line_trimmed_match(self):
        self.write_raw("hello world\nnext line\n")
        r = ok(edit_file(self.path, "   hello world", "HELLO WORLD"))
        self.assertTrue(r["success"])
        self.assertIn("HELLO WORLD", self.read_raw())

    def test_crlf_preserved(self):
        with open(self.path, "wb") as f:
            f.write(b"line one\r\nline two\r\n")
        r = ok(edit_file(self.path, "line one", "LINE ONE"))
        self.assertTrue(r["success"])
        with open(self.path, "rb") as f:
            data = f.read()
        self.assertIn(b"LINE ONE\r\n", data)

    def test_emoji_and_special_chars_in_match_and_replacement(self):
        self.write_raw("greeting: héllo 🎉\nsecond: 日本語\n")
        r = ok(edit_file(self.path, "héllo 🎉", "wörld 💯"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "greeting: wörld 💯\nsecond: 日本語\n")

    def test_emoji_match_preserves_rest_of_file_bytes(self):
        original = "🚀 launch\ncafé\n한국어 line\n"
        self.write_raw(original)
        r = ok(edit_file(self.path, "🚀 launch", "🎉 party"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "🎉 party\ncafé\n한국어 line\n")

    def test_combining_and_zero_width_chars(self):
        self.write_raw("a\u0301ccent\ufeffbom \u200bzwb\n")
        r = ok(edit_file(self.path, "a\u0301ccent", "changed"))
        self.assertTrue(r["success"])
        self.assertEqual(self.read_raw(), "changed\ufeffbom \u200bzwb\n")

    def test_additions_deletions_counts(self):
        self.write_raw("one\ntwo\nthree\n")
        r = ok(edit_file(self.path, "two", "TWO"))
        self.assertTrue(r["success"])
        self.assertEqual(r["additions"], 1)
        self.assertEqual(r["deletions"], 1)


if __name__ == "__main__":
    unittest.main()
