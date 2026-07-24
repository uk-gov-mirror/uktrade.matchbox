"""Build the warehouse both pipelines read from.

Four sources describing the same companies, with the messiness you actually get:
different column names, different casing and punctuation, duplicates within a source,
and only partial overlap between sources.

The ground truth is known, so both pipelines can be checked rather than eyeballed.
Run this first: `python warehouse.py`.
"""

import sqlite3
from pathlib import Path

DB = Path(__file__).parent / "warehouse.sqlite"

# (entity, name, postcode) — entity is the truth, never given to either pipeline.
COMPANIES = [
    (1, "Acme Trading Ltd", "SW1A 1AA"),
    (2, "Beta Manufacturing PLC", "M1 4BT"),
    (3, "Gamma Logistics Limited", "LS1 4AP"),
    (4, "Delta Foods Ltd", "BS1 6EA"),
    (5, "Epsilon Consulting", "EH2 2BY"),
    (6, "Zeta Pharma Ltd", "CB2 1TN"),
    (7, "Eta Engineering Ltd", "B1 1AA"),
    (8, "Theta Retail Group", "G2 8DL"),
]

# Companies House: has near-duplicates from re-registrations.
CRN = [
    ("crn-01", "ACME TRADING LTD", "SW1A 1AA", 1),
    ("crn-02", "Acme Trading Ltd.", "SW1A1AA", 1),  # dup: punctuation + spacing
    ("crn-03", "BETA MANUFACTURING PLC", "M1 4BT", 2),
    ("crn-04", "GAMMA LOGISTICS LIMITED", "LS1 4AP", 3),
    ("crn-05", "Gamma Logistics Ltd", "LS1 4AP", 3),  # dup: Limited vs Ltd
    ("crn-06", "DELTA FOODS LTD", "BS1 6EA", 4),
    ("crn-07", "EPSILON CONSULTING", "EH2 2BY", 5),
    ("crn-08", "ZETA PHARMA LTD", "CB2 1TN", 6),
]

# Data Hub: free-text entry, so more variation.
DH = [
    ("dh-100", "Acme Trading", "sw1a 1aa", 1),
    ("dh-101", "Beta Manufacturing", "m1 4bt", 2),
    ("dh-102", "Beta Manufacturing plc", "M1  4BT", 2),  # dup: double space
    ("dh-103", "Delta Foods", "bs1 6ea", 4),
    ("dh-104", "Eta Engineering", "b1 1aa", 7),
    ("dh-105", "Theta Retail", "g2 8dl", 8),
]

# CRM: different column names entirely, and a company CH never saw.
CDMS = [
    ("cdms-A", "ACME TRADING LIMITED", "SW1A 1AA", 1),
    ("cdms-B", "Gamma Logistics", "LS1 4AP", 3),
    ("cdms-C", "Epsilon Consulting Ltd", "EH2 2BY", 5),
    ("cdms-D", "Eta Engineering Limited", "B1 1AA", 7),
]

# Exporter register: sparse, names only lightly normalised.
HMRC = [
    ("hmrc-9001", "Acme Trading Ltd", "SW1A 1AA", 1),
    ("hmrc-9002", "Zeta Pharma", "CB2 1TN", 6),
    ("hmrc-9003", "Theta Retail Group Ltd", "G2 8DL", 8),
]

TABLES = {
    # name: (ddl, rows, columns in the row tuples)
    "crn": (
        "CREATE TABLE crn (company_number TEXT, company_name TEXT, "
        "registered_postcode TEXT, _truth INTEGER)",
        CRN,
    ),
    "dh": (
        "CREATE TABLE dh (dh_id TEXT, name TEXT, postcode TEXT, _truth INTEGER)",
        DH,
    ),
    "cdms": (
        "CREATE TABLE cdms (account_ref TEXT, account_name TEXT, "
        "post_code TEXT, _truth INTEGER)",
        CDMS,
    ),
    "hmrc": (
        "CREATE TABLE hmrc (trader_id TEXT, trader_name TEXT, "
        "postcode TEXT, _truth INTEGER)",
        HMRC,
    ),
}


def build() -> Path:
    """Create the warehouse, replacing any existing one."""
    DB.unlink(missing_ok=True)
    with sqlite3.connect(DB) as conn:
        for table, (ddl, rows) in TABLES.items():
            conn.execute(ddl)
            placeholders = ",".join("?" * len(rows[0]))
            conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
    return DB


def truth() -> dict[str, int]:
    """Every source key mapped to its true entity — for checking, never for matching."""
    return {row[0]: row[-1] for _, rows in TABLES.values() for row in rows}


if __name__ == "__main__":
    print(f"built {build()}")  # noqa: T201
    print(f"{len(truth())} records across {len(TABLES)} sources")  # noqa: T201
