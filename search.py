import sqlite3, json, sys, re
from pathlib import Path

CONVERSATIONS_DIR = "conversations"
DB = CONVERSATIONS_DIR + "/conversations.db"
DIR = Path(CONVERSATIONS_DIR)
CONTEXT_CHARS = 1000

def fts_snippet(text, terms_str):
    if not text or not terms_str: return ""
    ctx = CONTEXT_CHARS // 2
    results = []
    for term in [t.strip() for t in terms_str.split('|')]:
        pat = re.compile(re.escape(term), re.IGNORECASE)
        seen = set()
        for m in pat.finditer(text):
            k = (m.start(), len(m.group()))
            if k not in seen:
                seen.add(k)
                s, e = max(0, m.start()-ctx), min(len(text), m.end()+ctx)
                snip = text[s:e].replace('\n', ' ')[:CONTEXT_CHARS]
                results.append(f"{'...' if s else ''}{snip}{'...' if e < len(text) else ''}")
    return '\n'.join(results)[:500]


def fts_highlight(text, terms_str):
    if not text or not terms_str: return ""
    for term in [t.strip() for t in terms_str.split('|')]:
        pat = re.compile(re.escape(term), re.IGNORECASE)
        text = pat.sub(lambda m: f"<b>{m.group()}</b>", text)
    prefix = ".." 
    suffix = ""
    return (prefix + " " + text).replace("\n", " ")[:1024]


def search(term: str) -> str:
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.executescript("""DROP TABLE IF EXISTS fts_docs; DROP TABLE IF EXISTS docs""")
    c.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, filename TEXT, content TEXT)")
    c.execute("CREATE VIRTUAL TABLE fts_docs USING FTS5(content, content=docs)")

    for p in DIR.iterdir():
        if p.name == "conversations.db": continue
        text = " ".join(str(m.get("content") or "") for m in json.loads(p.read_text())) if p.suffix == ".json" else p.read_text()
        c.execute("INSERT INTO docs (filename, content) VALUES (?, ?)", (p.name, text))

    conn.commit()
    c.execute("INSERT OR IGNORE INTO fts_docs(rowid, content) SELECT id, content FROM docs")
    conn.commit()

    # Register custom FTS5 auxiliary functions as UDFs to replace missing built-ins
    conn.create_function("snippet", 2, lambda t,tr: fts_snippet(t,tr), deterministic=True)
    conn.create_function("highlight", 2, lambda t,tr: fts_highlight(t,tr), deterministic=True)

    rows = c.execute("""SELECT d.filename, 
        highlight(d.content, ?) as highlighted,
        snippet(d.content, ?) as snippets_text
        FROM fts_docs JOIN docs d ON d.id = fts_docs.rowid 
        WHERE fts_docs MATCH ? ORDER BY rank""", (term, term, term))

    matches = []
    for filename, highlighted, snips in rows:
        matches.append({
            "filename": CONVERSATIONS_DIR+"/"+filename,
            "highlighted": highlighted if highlighted else "",
            "snippets_text": snips if snips else ""
        })
    conn.close()
    return json.dumps({"query": term, "count": len(matches), "matches": matches}, indent=2)


if __name__ == "__main__":
    term = sys.argv[1] if len(sys.argv) > 1 else "linux"
    print(search(term))
