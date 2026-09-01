#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["textstat", "typer"]
# ///
"""Run plain-English rules over .py or .md files, one line per finding.

Rules (run all by default, or pick with --rules):

    pleng001  Flesch-Kincaid readability score per block
    pleng002  sentences containing a colon, semicolon, or dash
    pleng003  sentences opening with a FANBOYS conjunction

A "block" is a docstring, comment run (for .py) or a paragraph (for .md).
Blocks below TRIVIAL_WORDS are skipped from scoring entirely (the maths is
meaningless on 1-3 word fragments). Blocks below MIN_WORDS are still scored
but flagged low_confidence.

Usage:
    python readability.py <path> [<path> ...]
    uv run readability.py <path> [<path> ...] [--top N] [--rules pleng002]
"""

import ast
import json
import re
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import textstat
import typer

MIN_WORDS = 30
TRIVIAL_WORDS = 5

FANBOYS = {"for", "and", "nor", "but", "or", "yet", "so"}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PUNCT_MARKS = re.compile(r"[;:]|—|–|(?<=\s)-(?=\s)")
LEADING_WORD = re.compile(r"^[\"'“‘(]*([A-Za-z]+)")
INLINE_CODE = re.compile(r"`([^`]*)`")


class Rule(StrEnum):
    """Plain-English rule ids, in the order they run and print."""

    READABILITY = "pleng001"
    PUNCTUATION = "pleng002"
    FANBOYS = "pleng003"


RULE_DESCRIPTIONS = {
    Rule.READABILITY: "Flesch-Kincaid readability score per block",
    Rule.PUNCTUATION: "sentences containing a colon, semicolon, or dash",
    Rule.FANBOYS: "sentences opening with a FANBOYS conjunction",
}


def strip_inline_code(text: str) -> str:
    """Drop backticks but keep the token inside, so identifiers stay readable."""
    return INLINE_CODE.sub(r"\1", text)


def sentences_of(text: str) -> list[str]:
    """Split a block into sentences (naive, on . ! ?), dropping empty fragments."""
    parts = SENTENCE_SPLIT.split(text.replace("\n", " "))
    return [s.strip() for s in parts if s.strip()]


@dataclass
class BlockScore:
    """The readability result for one block, or the reason it wasn't scored."""

    label: str
    words: int
    status: str  # "", "low_confidence", "trivial"
    grade: float | None
    ease: float | None


def score_block(label: str, text: str) -> BlockScore:
    """Score a block, or mark it trivial instead of scoring meaningless text."""
    word_count = len(text.split())
    if word_count < TRIVIAL_WORDS:
        return BlockScore(label, word_count, "trivial", None, None)
    grade = textstat.flesch_kincaid_grade(text)
    ease = textstat.flesch_reading_ease(text)
    status = "low_confidence" if word_count < MIN_WORDS else ""
    return BlockScore(label, word_count, status, grade, ease)


def format_score(score: BlockScore) -> str:
    """Format a scored block as a single key=value line."""
    status = f" status={score.status}" if score.status else ""
    return (
        f"{score.label} words={score.words}{status} "
        f"flesch_kincaid_grade={score.grade} flesch_reading_ease={score.ease}"
    )


def punctuation_sentences(text: str) -> list[str]:
    """Return each sentence in text that contains a colon, semicolon, or dash."""
    return [s for s in sentences_of(text) if PUNCT_MARKS.search(s)]


def fanboys_sentences(text: str) -> list[tuple[str, str]]:
    """Return (word, sentence) pairs for sentences opening with a FANBOYS word."""
    found = []
    for s in sentences_of(text):
        m = LEADING_WORD.match(s)
        if m and m.group(1).lower() in FANBOYS:
            found.append((m.group(1).lower(), s))
    return found


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
        yield f"line={line} block={name}", strip_inline_code(doc)
    for label, text in blocks_from_comments(path):
        yield label, strip_inline_code(text)


def blocks_from_md(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (label, paragraph) pairs for each paragraph, with markup stripped."""
    text = path.read_text()
    # Blank out code blocks but keep their newlines, so line numbers stay accurate.
    text = re.sub(
        r"```.*?```", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL
    )
    text = strip_inline_code(text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # italic
    # admonition marker lines (zensical-style), e.g. !!! warning "title"
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


def score_file(path: Path, top: int | None, rules: list[Rule]) -> None:
    """Extract blocks from a file, print a summary, then each active rule's findings."""
    if path.suffix == ".py":
        blocks = list(blocks_from_py(path))
    elif path.suffix == ".md":
        blocks = list(blocks_from_md(path))
    else:
        typer.echo(f"Unsupported file type: {path.suffix}", err=True)
        return

    if not blocks:
        print(f"file={path} status=no_prose_found")
        return

    summary = [f"file={path}"]

    scored: list[BlockScore] = []
    if Rule.READABILITY in rules:
        scores = [score_block(label, text) for label, text in blocks]
        scored = [s for s in scores if s.grade is not None]
        trivial = sum(1 for s in scores if s.status == "trivial")
        low_conf = sum(1 for s in scores if s.status == "low_confidence")
        worst = max((s.grade for s in scored), default=None)
        worst_str = f"{worst:.1f}" if worst is not None else "none"
        summary.append(f"blocks={len(blocks)}")
        summary.append(f"trivial={trivial}")
        summary.append(f"low_confidence={low_conf}")
        summary.append(f"worst_grade={worst_str}")

    punct_flags: list[tuple[str, str]] = []
    if Rule.PUNCTUATION in rules:
        punct_flags = [
            (label, s) for label, text in blocks for s in punctuation_sentences(text)
        ]
        summary.append(f"punctuation_flags={len(punct_flags)}")

    fanboys_flags: list[tuple[str, str, str]] = []
    if Rule.FANBOYS in rules:
        fanboys_flags = [
            (label, word, s)
            for label, text in blocks
            for word, s in fanboys_sentences(text)
        ]
        summary.append(f"fanboys_flags={len(fanboys_flags)}")

    print(" ".join(summary))

    if Rule.READABILITY in rules:
        shown = scored
        if top is not None:
            shown = sorted(scored, key=lambda s: s.grade, reverse=True)[:top]
        for score in shown:
            print(format_score(score))

    for label, sentence in punct_flags:
        print(f"{label} sentence={json.dumps(sentence)}")

    for label, word, sentence in fanboys_flags:
        print(f"{label} word={word} sentence={json.dumps(sentence)}")


def parse_rules(value: str | None) -> list[Rule]:
    """Parse a comma-separated --rules value, defaulting to every rule."""
    if not value:
        return list(Rule)
    selected: list[Rule] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            selected.append(Rule(token))
        except ValueError:
            valid = ", ".join(r.value for r in Rule)
            typer.echo(f"Unknown rule: {token}. Valid: {valid}", err=True)
            raise typer.Exit(1) from None
    return selected


def main(
    paths: list[Path] = typer.Argument(  # noqa: B008
        ..., help="One or more .py/.md files or directories to score."
    ),
    top: int | None = typer.Option(
        None, "--top", help="Show only the N worst-scoring blocks per file."
    ),
    rules: str | None = typer.Option(
        None,
        "--rules",
        help="Comma-separated rule ids to run (default: all). "
        + "; ".join(f"{r.value}={d}" for r, d in RULE_DESCRIPTIONS.items()),
    ),
) -> None:
    """Score .py/.md files or directories against one or more plain-English rules."""
    selected_rules = parse_rules(rules)

    files: list = []
    for path in paths:
        if not path.exists():
            typer.echo(f"Path not found: {path}", err=True)
            raise typer.Exit(1)
        if path.is_dir():
            found = sorted(p for p in path.rglob("*") if p.suffix in (".py", ".md"))
            if not found:
                typer.echo(f"No .py or .md files found under {path}", err=True)
                raise typer.Exit(1)
            files.extend(found)
        else:
            files.append(path)

    for file in files:
        score_file(file, top=top, rules=selected_rules)


if __name__ == "__main__":
    typer.run(main)
