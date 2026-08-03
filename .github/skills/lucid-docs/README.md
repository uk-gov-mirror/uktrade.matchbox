# Lucid docs v0.1

Plain syntax. Precise terms. Necessary detail.

Reduce reading effort without reducing technical depth.

## Reader

The skill defines a mid-level Python reader by assumed knowledge.

It assumes comfort with idiomatic Python, common language features, standard library conventions, type hints, testing, and ordinary object-oriented design. It does not explain standard Python or established dependencies unless their local use is surprising. It explains repository-specific concepts, hidden constraints, surprising behaviour, and non-obvious interactions.

The voice is direct when the repository supports a clear claim. Uncertainty is surfaced.

## Repository glossary

`docs/glossary.md` is the canonical source for repository-specific terminology. It is user-facing documentation and shared context for coding agents.

The skill reads the glossary before editing documentation. It adds or updates terms that developers must understand as concepts, while excluding ordinary Python terms, obvious identifiers, and incidental implementation details.

The glossary owns each canonical definition. Individual documents still provide enough local context to remain readable. Ambiguous or conflicting uses are reported rather than silently resolved.

## Documentation locality

Each explanation belongs at the narrowest scope that contains everyone who needs it:

- module docstrings hold cross-cutting decisions, boundaries, and a short map
- class and function docstrings hold the unit's contract and local why
- inline comments hold the smallest-scope reason or constraint

Shared reasoning is stated once and referred to by name. Shared contracts live at the abstraction that owns them. Implementations and overrides document only their differences.

## Proportionality and comments

Documentation depth follows conceptual weight, not line count. Simple units stay brief. `Args` and `Returns` sections appear only when they add information beyond the signature and type hints.

A comment earns its place when it names a reasonable wrong turn, hidden constraint, or non-obvious consequence. The code remains responsible for routine mechanics.

## Our style

We want to establish a simple term to encapsulate our style as succinctly as possible.

The established umbrella term is **plain language**, or **plain English** in a UK house style.

Plain language is defined for an intended audience. It does not require replacing terms that the audience needs and understands. For developer documentation, this means simple sentence structure around exact technical vocabulary.

The narrower linguistic operation is **syntactic simplification**: making sentence structure easier to parse while preserving meaning.

The closest formal engineering analogue is **ASD-STE100 Simplified Technical English**. It constrains grammar and general vocabulary while permitting project-specific technical nouns and verbs. This skill borrows that distinction, but it does not claim ASD-STE100 compliance. The full standard is more restrictive and uses American English.

For this project, the practical label is:

> Plain English for technical readers.

## Research links

- ISO 24495-1 plain language standard: https://www.iso.org/obp/ui/#iso:std:iso:24495:-1:ed-1:v1:en
- GOV.UK guidance on clear language for specialists: https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/
- IEEE guidance on plain language for engineers: https://procomm.ieee.org/communication-resources-for-engineers/other-topics/plain-language/
- ASD-STE100 official overview: https://www.asd-ste100.org/about_STE.html
- Google guidance on short technical sentences: https://developers.google.com/tech-writing/one/short-sentences
- Matt Pocock's domain-modelling skill: https://github.com/mattpocock/skills/tree/main/skills/engineering/domain-modeling
- Martin Fowler on ubiquitous language: https://martinfowler.com/bliki/UbiquitousLanguage.html
