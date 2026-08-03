#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["textstat"]
# ///
"""Print Flesch-Kincaid readability scores per block for a .py or .md file.

A "block" is a docstring, comment run (for .py) or a paragraph (for .md). Blocks
under MIN_WORDS are still scored, but flagged low_confidence, since Flesch-Kincaid
grows unreliable on short text.

Usage:
    python readability.py <path>
    uv run readability.py <path>
"""

import ast
import re
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

import textstat

MIN_WORDS = 30


def score_block(label: str, text: str) -> str:
    """Return a formatted readability score line for a single block.

    Blocks under MIN_WORDS are still scored but flagged low_confidence,
    since Flesch-Kincaid grows unreliable on short text rather than unusable.
    """
    word_count = len(text.split())
    if word_count == 0:
        return f"{label} words=0 status=empty"
    grade = textstat.flesch_kincaid_grade(text)
    ease = textstat.flesch_reading_ease(text)
    status = " status=low_confidence" if word_count < MIN_WORDS else ""
    return (
        f"{label} words={word_count}{status} "
        f"flesch_kincaid_grade={grade} flesch_reading_ease={ease}"
    )


def blocks_from_comments(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (label, text) pairs for each run of consecutive whole-line comments."""
    with open(path, "rb") as f:
        run: list = []
        start_line = None
        prev_line = None
        for tok in tokenize.tokenize(f.readline):
            if tok.type != tokenize.COMMENT:
                continue
            text = tok.string.lstrip("#").strip()
            if run and prev_line == tok.start[0] - 1:
                run.append(text)
            else:
                if run:
                    yield f"line={start_line} block=comment", " ".join(run)
                run = [text]
                start_line = tok.start[0]
            prev_line = tok.start[0]
        if run:
            yield f"line={start_line} block=comment", " ".join(run)


def blocks_from_py(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (label, text) pairs for each docstring and comment block in a .py file."""
    tree = ast.parse(path.read_text())
    doc_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
    for node in ast.walk(tree):
        if not isinstance(node, doc_nodes):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        name = getattr(node, "name", "module")
        line = getattr(node, "lineno", 1)
        yield f"line={line} block={name}", doc
    yield from blocks_from_comments(path)


def blocks_from_md(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (label, paragraph) pairs for each paragraph, with markup stripped."""
    text = path.read_text()
    # Blank out code blocks but keep their newlines, so line numbers stay accurate.
    text = re.sub(
        r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL
    )
    text = re.sub(r"`([^`]*)`", r"\1", text)  # inline code — keep the token
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # italic
    # admonition marker lines (mkdocs-style), e.g. !!! warning "title"
    text = re.sub(r'^!!!\s+\S+(\s+".*?")?\s*$', "", text, flags=re.MULTILINE)

    table_row = re.compile(r"^\s*\|.*\|\s*$")
    para_lines: list = []
    start_line = None
    for i, raw_line in enumerate(text.split("\n"), start=1):
        if table_row.match(raw_line):
            if para_lines:
                yield f"line={start_line} block=paragraph", "\n".join(para_lines)
            para_lines, start_line = [], None
            continue
        if raw_line.strip():
            if start_line is None:
                start_line = i
            para_lines.append(raw_line)
        else:
            if para_lines:
                yield f"line={start_line} block=paragraph", "\n".join(para_lines)
            para_lines, start_line = [], None
    if para_lines:
        yield f"line={start_line} block=paragraph", "\n".join(para_lines)


def score_file(path: Path) -> None:
    """Extract blocks from a single .py or .md file and print a score per block."""
    if path.suffix == ".py":
        blocks = list(blocks_from_py(path))
    elif path.suffix == ".md":
        blocks = list(blocks_from_md(path))
    else:
        print(f"Unsupported file type: {path.suffix}", file=sys.stderr)
        return

    if not blocks:
        print(f"file={path} status=no_prose_found")
        return

    print(f"file={path} block_count={len(blocks)} min_words={MIN_WORDS}")
    for label, text in blocks:
        print(score_block(label, text))


def main() -> None:
    """Score a single file, or every .py/.md file under a directory, from argv."""
    if len(sys.argv) != 2:
        print("Usage: readability.py <path-to-file-or-directory>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix in (".py", ".md"))
        if not files:
            print(f"No .py or .md files found under {path}", file=sys.stderr)
            sys.exit(1)
        for file in files:
            score_file(file)
    else:
        score_file(path)


if __name__ == "__main__":
    main()
