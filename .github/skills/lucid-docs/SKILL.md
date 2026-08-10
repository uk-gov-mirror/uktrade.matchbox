---
name: lucid-docs
description: Edit Python documentation, docstrings, comments, and repository terminology in plain English for technical readers. Reduce reading effort while preserving precise terms, contracts, design reasons, warnings, and non-obvious behaviour.
license: CC0-1.0
compatibility: Designed for coding agents that can inspect a repository and edit text files.
---

# Lucid docs

## Goal

Rewrite the specified documentation to reduce reading effort. Do not cut technical depth.

Write in plain English for technical readers:

> Plain syntax. Precise terms. Necessary detail.

Fluent, sophisticated prose is NOT plain English for technical readers. Rewrite sentence structure ruthlessly. Use short sentences and direct verbs. Reduce ideas per sentence, but keep precise technical terms.

Pay particular attention to the glossary.

Ensure you run all three final checks.

## Reader

The reader is a mid-level Python developer, with a B2 CEFR reading level (Flesch-Kincaid Grade 8-9).

Assume the reader can read idiomatic Python and understands common language features, standard library conventions, type hints, testing, and ordinary object-oriented design.

Do not explain standard Python or established dependencies unless their use here differs from normal expectations. Explain repository-specific concepts, hidden constraints, surprising behaviour, and non-obvious interactions.

State verified behaviour directly. Surface genuine uncertainty instead of weakening every claim.

## Repository language

Treat `docs/glossary.md` as the source of truth for repository-specific terms. Read it before editing documentation.

Add or update a term when the reader must understand it as a concept. Include domain concepts, named workflows, architectural concepts, and important local distinctions. Exclude ordinary Python terms, obvious identifiers, and incidental implementation details.

Write the glossary for developers as well as agents. Use the same plain, precise style as the rest of the documentation. Let the glossary own the canonical definition, while each document supplies enough local context to remain readable.

Do not invent a definition or silently resolve conflicting uses. Report ambiguity when the repository does not support one clear meaning.

## Editing

Read enough code, tests, types, nearby documentation, and glossary entries to preserve the meaning.

Keep information that helps a developer:

- use the code correctly
- understand a design choice
- preserve an invariant
- predict a side effect or failure
- follow a non-obvious interaction
- avoid a footgun

Then tighten the prose:

1. Put the main point first.
2. Give each sentence one job, but clarity always outranks brevity.
3. Put the subject and verb early.
4. Prefer direct verbs to noun-heavy phrases.
5. Remove words that add no meaning.
6. Keep exact technical terms.
7. Check that no condition, warning, or consequence was lost.

Do not mistake prose that reads fluently and is technically sophisticated for plain English for technical readers. 

Do not minimise the diff on the documentation you've been asked to review. Assume it has been written maximally, and your job is to rewrite minimally. Plain syntax. Precise terms. Necessary detail.

## House style

- Use British English.
- Use sentence case headings.
- Do not use semicolons.
- Remove colons and dashes unless they truly simplify.
- Do not start sentences with the FANBOYS set (for, and, nor, but, or, yet, so) unless the sentence stands alone, with its own subject, verb, and stated reason.
- Use present tense for current behaviour.
- Use one term for one concept.
- Describe earlier or planned code only in migration guides.

## Python documentation

Put each explanation at the narrowest scope that contains everyone who needs it. State shared reasoning once, then refer to it by name instead of repeating it.

Use module docstrings for cross-cutting design decisions, boundaries, and a short map of what the module contains. Explain why the module has its current shape, without reference to past or future code states outside of migration guides. Do not restate its implementation.

Use class and function docstrings for the unit's contract and local why. Record the invariant it relies on, the footgun it avoids, or the important alternative it rejects. Let names, signatures, and type hints carry what they already express.

Use inline comments for the smallest-scope fact or reason. Place them beside the lines they explain. A comment earns its place when it names a reasonable wrong turn, hidden constraint, or non-obvious consequence. Let the code express routine mechanics.

Match documentation to conceptual weight. Keep simple units brief. Do not add `Args` or `Returns` sections merely because a function has parameters or returns a value. Give fuller treatment to contracts, decisions, and risks that need it.

Document a shared contract at the highest abstraction that owns it. Overrides and implementations document only their differences, added constraints, and surprising behaviour.

Mention another part of the codebase only when the interaction affects correct use or safe maintenance.

## Final check

The result should be easier to read and no less exact.

### Phase 1

Confirm that:

- the first sentence carries the main point
- each sentence is easy to parse
- precise technical vocabulary remains
- repository-specific terms match `docs/glossary.md`
- new or changed concepts are reflected in the glossary
- each explanation sits at the narrowest useful scope
- documentation depth matches conceptual weight
- shared contracts are stated once at the abstraction that owns them
- comments add reasoning the code cannot show
- design reasons and warnings remain
- every claim is supported by the repository
- the prose describes the code that exists now, not past or future states

### Phase 2

Perform a final check of the readability score and house style:

```bash
uv run .github/skills/lucid-docs/readability.py <path>
```

This takes the following options:

* `--top <int>`: only show the N hardest-to-read blocks per file
* `--rules`: comma-separated rule ids to run (default: all)
  * `pleng001` readability score
  * `pleng002` colon/semicolon/dash
  * `pleng003` FANBOYS sentence openers

Ensure an acceptable reading level for the reader. You may make allowances for the requirements of technical writing.

Colons and dashes are permissable only if they truly simplify.

This script is a spot-check, not the target. It cannot detect banned words or fluent-but-wordy prose. 

### Phase 3

Reread each file in its entirety, assuming it does not yet read as plain English.

For each paragraph, find the sentence a plain English reader would stumble on, state why to yourself, then change it. Only move on once you cannot find one.

Do not attempt to minimise churn.

## Output

Edit the requested documentation and `docs/glossary.md` in place when tools allow it.

Report the files changed, unresolved terminology, and any claim that could not be verified. Keep the report brief.
