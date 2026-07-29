"""Resolve four company sources with matchlab.

Dedupe each source, then link the deduplicated entities together. Run `warehouse.py`
first.

Compare with `by_hand.py`, which does the same thing with Polars and a disjoint set.
Both print the same answer.
"""

from itertools import combinations
from pathlib import Path

from sqlalchemy import create_engine

from matchlab import Source
from matchlab.adapters import DuckDBAdapter
from matchlab.locations import RelationalDBLocation
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker

DB = Path(__file__).parent / "warehouse.sqlite"

# One cleaning rule, applied to whichever column each source happens to use.
NAME = (
    "regexp_replace(upper({0}), '\\s+|\\.|\\bLIMITED\\b|\\bLTD\\b|\\bPLC\\b', '', 'g')"
)
POSTCODE = "regexp_replace(upper({0}), '\\s+', '', 'g')"


def build() -> "Source":
    """Build the plan. Nothing runs until it's collected."""
    location = RelationalDBLocation(
        name="warehouse", client=create_engine(f"sqlite:///{DB}")
    )

    def source(name: str, key: str, name_col: str, postcode_col: str) -> Source:
        return Source(
            location=location,
            name=name,
            extract_transform=(f"select {key}, {name_col}, {postcode_col} from {name}"),
            key_field=key,
        )

    crn = source("crn", "company_number", "company_name", "registered_postcode")
    dh = source("dh", "dh_id", "name", "postcode")
    cdms = source("cdms", "account_ref", "account_name", "post_code")
    hmrc = source("hmrc", "trader_id", "trader_name", "postcode")

    def cleaned(src: Source, name_col: str, postcode_col: str):
        return src.view(
            cleaning={
                "name": NAME.format(src.f(name_col)),
                "postcode": POSTCODE.format(src.f(postcode_col)),
            }
        )

    views = {
        "crn": cleaned(crn, "company_name", "registered_postcode"),
        "dh": cleaned(dh, "name", "postcode"),
        "cdms": cleaned(cdms, "account_name", "post_code"),
        "hmrc": cleaned(hmrc, "trader_name", "postcode"),
    }

    # Dedupe within each source: same cleaned name and postcode is one company.
    dedupes = [
        view.dedupe(
            model_class=NaiveDeduper,
            model_settings={"unique_fields": ["name", "postcode"]},
        )
        for view in views.values()
    ]
    deduped = dedupes[0].resolve(*dedupes[1:])

    # Read each source *through* the dedupe, so `id` is an entity rather than a
    # record. Build each view once — one node, computed once, read by every link
    # that shares it.
    entity_views = {
        src.name: deduped.view(src, cleaning=views[src.name].cleaning)
        for src in (crn, dh, cdms, hmrc)
    }

    # Link the deduplicated entities to each other, not the raw rows. Every pair,
    # because two sources can share a company that a third has never heard of.
    comparison = "l.name = r.name and l.postcode = r.postcode"
    links = [
        entity_views[left].link(
            entity_views[right],
            model_class=DeterministicLinker,
            model_settings={"comparisons": comparison},
        )
        for left, right in combinations(entity_views, 2)
    ]

    return links[0].resolve(*links[1:])


if __name__ == "__main__":
    from warehouse import truth

    # Publishing is what makes a resolution findable later, without this script.
    companies = build().collect(DuckDBAdapter(":memory:")).publish("companies")

    resolution = companies.entities()
    groups: dict[int, set[str]] = {}
    for row in resolution.iter_rows(named=True):
        groups.setdefault(row["root"], set()).add(row["key"])

    print(f"{len(groups)} entities from {resolution.height} records\n")
    expected = truth()
    for keys in sorted(groups.values(), key=lambda k: sorted(k)[0]):
        entities = {expected[key] for key in keys}
        mark = "✓" if len(entities) == 1 else "✗"
        print(f"  {mark} {sorted(keys)}")
