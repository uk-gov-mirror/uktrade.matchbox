"""What gets measured: the cases, and the sweeps that arrange them into questions.

The sweeps here vary **one axis at a time** from a shared baseline rather than crossing
every axis with every other, to save time and to isolate the impact of each axis.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.data import Warehouse
from benchmarks.topologies import Topology

# The middle of every sweep. Each sweep replaces exactly one of these.
BASELINE = Warehouse(n_entities=50_000, records_per_entity=3, n_sources=4)
BASELINE_TOPOLOGY = Topology.MESH


class Store(StrEnum):
    """Where a case's artifacts go while it runs.

    `FILE` is what a user gets by default and makes store size a real number on a real
    disk. `MEMORY` keeps everything in the process, which moves the same bytes into
    peak RSS — the same work, attributed differently, which is the only reason to
    measure both.
    """

    FILE = "file"
    MEMORY = "memory"


class Case(BaseModel):
    """One plan shape over one warehouse: the unit of measurement.

    Frozen and hashable so the runner can group cases by warehouse and build each one
    only once, however many topologies read it.
    """

    model_config = ConfigDict(frozen=True)

    topology: Topology
    warehouse: Warehouse
    store: Store = Store.FILE

    @property
    def label(self) -> str:
        """This case in one line, for a progress record to name."""
        return (
            f"{self.topology} / {self.warehouse.n_entities:,} entities / "
            f"{self.warehouse.records_per_entity} per entity / "
            f"{self.warehouse.n_sources} sources"
        )

    @property
    def expected_entities(self) -> int | None:
        """How many entities this shape must resolve to, where the shape fixes it.

        Every entity's records clean to one value, so any shape that can reach every
        pair of sources recovers the planted entities exactly. `DEDUPE` sees one
        source, so it recovers only what that source holds.

        `HUB` is the exception and returns `None`: it cannot merge two spokes that
        share an entity the hub has never heard of, so how many entities it resolves
        depends on the overlap rather than on the shape. That is the property the
        topology sweep exists to show, so it is reported, never asserted.
        """
        match self.topology:
            case Topology.DEDUPE:
                return self.warehouse.covered
            case Topology.CHAIN | Topology.MESH:
                return self.warehouse.n_entities
            case Topology.HUB:
                return None


class Sweep(BaseModel):
    """One question, and the cases that answer it.

    Attributes:
        title: The axis being varied.
        holding: What is held fixed, so a reader can see the table is honest.
        column: Which per-case value the first table column shows.
        cases: The cases, in the order they should be read.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    holding: str
    column: str
    cases: tuple[Case, ...] = Field(min_length=1)


def _fixed(warehouse: Warehouse, topology: Topology, *, but: str) -> str:
    """Describe everything about a case except the axis being swept."""
    parts = {
        "topology": str(topology),
        "entities": f"{warehouse.n_entities:,} entities",
        "records": f"{warehouse.records_per_entity} records/entity",
        "sources": f"{warehouse.n_sources} sources",
    }
    del parts[but]
    return ", ".join(parts.values())


def topology_sweep(warehouse: Warehouse, store: Store) -> Sweep:
    """Every shape over one warehouse. The headline table."""
    return Sweep(
        title="Plan topology",
        holding=_fixed(warehouse, BASELINE_TOPOLOGY, but="topology"),
        column="topology",
        cases=tuple(
            Case(topology=topology, warehouse=warehouse, store=store)
            for topology in Topology
        ),
    )


def entity_sweep(counts: Sequence[int], store: Store) -> Sweep:
    """More entities, same records each. Grows the resolver output, not duplication."""
    warehouses = [BASELINE.model_copy(update={"n_entities": count}) for count in counts]
    return Sweep(
        title="Number of entities",
        holding=_fixed(warehouses[0], BASELINE_TOPOLOGY, but="entities"),
        column="entities",
        cases=tuple(
            Case(topology=BASELINE_TOPOLOGY, warehouse=warehouse, store=store)
            for warehouse in warehouses
        ),
    )


def record_sweep(counts: Sequence[int], store: Store) -> Sweep:
    """More records per entity, same entities. Grows duplication, not the answer."""
    warehouses = [
        BASELINE.model_copy(update={"records_per_entity": count}) for count in counts
    ]
    return Sweep(
        title="Records per entity",
        holding=_fixed(warehouses[0], BASELINE_TOPOLOGY, but="records"),
        column="records",
        cases=tuple(
            Case(topology=BASELINE_TOPOLOGY, warehouse=warehouse, store=store)
            for warehouse in warehouses
        ),
    )


def source_sweep(counts: Sequence[int], store: Store) -> Sweep:
    """More sources, for both shapes whose link count depends on how many there are.

    The pair is the point: a mesh links every pair and a hub links `n-1`, so this is
    where the quadratic and the linear shape visibly part company.
    """
    cases = tuple(
        Case(
            topology=topology,
            warehouse=BASELINE.model_copy(update={"n_sources": count}),
            store=store,
        )
        for count in counts
        for topology in (Topology.HUB, Topology.MESH)
    )
    return Sweep(
        title="Number of sources",
        holding=f"{BASELINE.n_entities:,} entities, "
        f"{BASELINE.records_per_entity} records/entity, hub and mesh",
        column="sources",
        cases=cases,
    )


def profile(name: str, store: Store = Store.FILE) -> list[Sweep]:
    """The sweeps a named profile runs.

    `quick` is a smoke test — every shape, small enough to finish in under a minute,
    and the one to run before believing anything else here. `standard` is the run
    worth quoting. `full` pushes each axis to where the curves either stay straight or
    stop being straight, and takes long enough that you should start it and go away.

    Raises:
        KeyError: If the profile is not one of the three.
    """
    small = BASELINE.model_copy(update={"n_entities": 5_000, "records_per_entity": 2})

    profiles: dict[str, list[Sweep]] = {
        "quick": [topology_sweep(small, store)],
        "standard": [
            topology_sweep(BASELINE, store),
            entity_sweep([10_000, 50_000, 200_000], store),
            record_sweep([1, 3, 6], store),
            source_sweep([2, 4, 6], store),
        ],
        "full": [
            topology_sweep(BASELINE, store),
            entity_sweep([10_000, 50_000, 200_000, 1_000_000], store),
            record_sweep([1, 3, 6, 12], store),
            source_sweep([2, 4, 6, 8], store),
        ],
    }
    if name not in profiles:
        raise KeyError(f"Unknown profile '{name}'. Choose from {sorted(profiles)}.")
    return profiles[name]


PROFILES = ("quick", "standard", "full")
