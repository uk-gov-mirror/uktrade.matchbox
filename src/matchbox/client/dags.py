"""Objects to define a DAG which indexes, deduplicates and links data."""

import tempfile
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias

import polars as pl
from platformdirs import user_cache_path
from pydantic import validate_call

from matchbox.adapters import Adapter, DuckDBAdapter
from matchbox.client._settings import settings
from matchbox.client.locations import Location
from matchbox.client.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers import Resolver
from matchbox.client.results import ResolverMatches
from matchbox.client.sources import Source
from matchbox.common.dtos import (
    CollectionName,
    DefaultGroup,
    GroupName,
    ModelStepName,
    ResolverStepName,
    RunID,
    SourceStepName,
    Step,
    StepName,
    StepType,
)
from matchbox.common.exceptions import (
    MatchboxStepNotFoundError,
)
from matchbox.common.logging import log_mem_usage, logger, profile_time


class DAGNodeExecutionStatus(StrEnum):
    """Enumeration of node execution statuses."""

    SKIPPED = "skipped"
    DONE = "done"
    DOING = "doing"


DAGExecutionStatus: TypeAlias = dict[str, DAGNodeExecutionStatus]

CACHE_DIR = user_cache_path("matchbox")


class DAG:
    """Self-sufficient pipeline of indexing, deduping and linking steps."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        name: CollectionName,
        admin_group: GroupName = DefaultGroup.PUBLIC,
        adapter: Adapter | None = None,
    ) -> None:
        """Initialises empty DAG.

        Args:
            name: The name of the DAG.
            admin_group: Retained for API compatibility; unused in local mode.
            adapter: The storage adapter that holds computed step artifacts. Defaults
                to a DuckDB store in the DAG's temporary cache directory.
        """
        self.name: CollectionName = CollectionName(name)
        self.admin_group: GroupName = GroupName(admin_group)
        self._run: RunID | None = None
        self.nodes: dict[StepName, Source | Model | Resolver] = {}
        self.graph: dict[StepName, list[StepName]] = {}

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_dir = tempfile.TemporaryDirectory(dir=str(CACHE_DIR))
        self.cache_path = Path(self._cache_dir.name)

        self.adapter: Adapter = adapter or DuckDBAdapter(
            self.cache_path / "store.duckdb"
        )

    def _check_dag(self, dag: Self) -> None:
        """Check that the given DAG is the same as this one."""
        if self != dag:
            raise ValueError("Cannot mix DAGs")

    def _check_step(
        self,
        step: Source | Model | Resolver,
        check_parents: bool,
        check_dependencies: bool,
    ) -> None:
        """Validate that the step references existing nodes in this DAG."""
        if check_parents:
            for parent_name in step.config.parents:
                if parent_name not in self.nodes:
                    raise ValueError(f"Parent step {parent_name} not added to DAG")

        if check_dependencies:
            for dependency_name in step.config.dependencies:
                if dependency_name not in self.nodes:
                    raise ValueError(
                        f"Dependency step {dependency_name} not added to DAG"
                    )

    def _add_step(self, step: Source | Model | Resolver) -> None:
        """Validate and add sources and models to DAG."""
        self._check_dag(step.dag)

        # We allow overwriting nodes, because:
        # - One might want to iterate on a local DAG without starting over
        # - One might need to load a pending run in a script which defines the same
        #   DAG, to re-run a specific node.
        #
        # We only mandate that a model's direct inputs not be changed. This ensures
        # that `self.graph` is not altered. It does not guarantee that the new node
        # works well with the rest of the DAG, which must be verified by the user.
        # For example, you could remove source fields that are needed downstream, or
        # you could query sources that are not available to a resolver.
        # These issues are not checked when adding a node for the first time either.
        if step.name in self.nodes:
            if step.config.parents != self.nodes[step.name].config.parents:
                raise ValueError("Cannot re-assign name to model with different inputs")
            logger.info(f"Overwriting node '{step.name}'.")

        if isinstance(step, Model):
            self._check_dag(step.left_query.dag)
            if step.right_query:
                self._check_dag(step.right_query.dag)
        elif isinstance(step, Resolver):
            for input_node in step.inputs:
                self._check_dag(input_node.dag)

        self._check_step(step, check_parents=False, check_dependencies=True)

        self.graph[step.name] = [parent for parent in step.config.parents]

        self.nodes[step.name] = step

    def _topological_sort(self, deps: dict[StepName, set[StepName]]) -> list[StepName]:
        """Topologically sort prerequisite constraints using Kahn's algorithm.

        Args:
            deps: A graph of precedence constraints where keys are node names and
                values are names that must come before each key.

        Returns:
            Deterministic topological order preserving insertion-order tie-breaks.

        Raises:
            ValueError: If dependencies reference unknown nodes or if a cycle exists.
        """
        missing_dependencies = {
            dep for node_deps in deps.values() for dep in node_deps if dep not in deps
        }
        if missing_dependencies:
            raise ValueError(
                "Cannot sort graph with missing dependencies: "
                f"{sorted(missing_dependencies)}"
            )

        in_degree: dict[StepName, int] = {
            node: len(node_deps) for node, node_deps in deps.items()
        }
        children: dict[StepName, list[StepName]] = {node: [] for node in deps}

        for node, node_deps in deps.items():
            for dep in node_deps:
                children[dep].append(node)

        ready = deque([node for node, degree in in_degree.items() if degree == 0])
        ordered: list[StepName] = []

        while ready:
            node = ready.popleft()
            ordered.append(node)

            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

        if len(ordered) != len(deps):
            remaining = [node for node, degree in in_degree.items() if degree > 0]
            raise ValueError(
                f"Cannot sort graph with unresolved cycles involving: {remaining}"
            )

        return ordered

    @property
    def run(self) -> RunID:
        """Return the local run ID.

        Retained for compatibility with step paths. Local DAGs use a single run.
        """
        return self._run or RunID(1)

    @run.setter
    def run(self, run_id: RunID) -> None:
        """Set run ID manually."""
        self._run = run_id

    @property
    def sequence(self) -> list[StepName]:
        """Return nodes in topological execution order.

        Returns:
            List of node names in the order they would be executed by run_and_sync.
            Use as start/finish values to control partial execution.
        """
        deps_graph = {name: set(parents) for name, parents in self.graph.items()}
        return self._topological_sort(deps=deps_graph)

    @property
    def final_steps(self) -> list[Source | Model | Resolver]:
        """Returns all apex nodes in the DAG.

        Returns:
            List of all apex nodes (nodes with no incoming edges).
            Returns empty list if DAG is empty.
        """
        if not self.nodes:
            return []

        # Find all nodes that appear as dependencies
        all_dependencies = set()
        for deps in self.graph.values():
            all_dependencies.update(deps)

        # Apex nodes are those that don't appear as anyone's dependency
        apex_node_names = [node for node in self.graph if node not in all_dependencies]
        return [self.nodes[name] for name in apex_node_names]

    @property
    def default_resolver(self) -> Resolver:
        """Return the default resolver for this DAG."""
        steps = self.final_steps

        if not steps:
            raise ValueError("No final step found, DAG might contain cycles")

        resolvers: list[Resolver] = [s for s in steps if isinstance(s, Resolver)]

        if not resolvers:
            raise ValueError("The only final step is not a resolver")
        if len(resolvers) > 1:
            raise ValueError("Default resolver is ambiguous.")

        return resolvers[0]

    def source(self, *args: Any, **kwargs: Any) -> Source:
        """Create Source and add it to the DAG."""
        source = Source(*args, **kwargs, dag=self)
        self._add_step(source)

        return source

    def model(self, *args: Any, **kwargs: Any) -> Model:
        """Create Model and add it to the DAG."""
        model = Model(*args, **kwargs, dag=self)
        self._add_step(model)

        return model

    def resolver(self, *args: Any, **kwargs: Any) -> Resolver:
        """Create a resolver and add it to the DAG."""
        resolver = Resolver(*args, **kwargs, dag=self)
        self._add_step(resolver)
        return resolver

    @validate_call
    def add_step(
        self,
        name: StepName,
        step: Step,
    ) -> None:
        """Add a step to the DAG."""
        if step.step_type == StepType.SOURCE:
            self.source(
                location=Location.from_config(step.config.location_config),
                name=SourceStepName(name),
                extract_transform=step.config.extract_transform,
                key_field=step.config.key_field,
                index_fields=step.config.index_fields,
                description=step.description,
                infer_types=False,
                validate_etl=False,
            )
        elif step.step_type == StepType.MODEL:
            self.model(
                name=ModelStepName(name),
                description=step.description,
                model_class=step.config.model_class,
                model_settings=step.config.model_settings,
                left_query=Query.from_config(step.config.left_query, dag=self),
                right_query=Query.from_config(step.config.right_query, dag=self)
                if step.config.right_query
                else None,
            )
        elif step.step_type == StepType.RESOLVER:
            self.resolver(
                name=ResolverStepName(name),
                description=step.description,
                inputs=(self.get_model(i) for i in step.config.inputs),
                resolver_class=step.config.resolver_class,
                resolver_settings=step.config.resolver_settings,
            )
        else:
            raise ValueError(f"Unknown step type {step.step_type}")

    @validate_call
    def get_source(self, name: StepName) -> Source:
        """Get a source by name from the DAG.

        Args:
            name: The name of the source to retrieve.

        Returns:
            The Source object.

        Raises:
            ValueError: If the name doesn't exist in the DAG or isn't a Source.
        """
        if name not in self.nodes:
            raise ValueError(f"Source '{name}' not found in DAG")

        node = self.nodes[name]
        if not isinstance(node, Source):
            raise ValueError(
                f"Node '{name}' is not a Source, it's a {type(node).__name__}"
            )

        return node

    @validate_call
    def get_model(self, name: StepName) -> Model:
        """Get a model by name from the DAG.

        Args:
            name: The name of the model to retrieve.

        Returns:
            The Model object.

        Raises:
            ValueError: If the name doesn't exist in the DAG or isn't a Model.
        """
        if name not in self.nodes:
            raise ValueError(f"Model '{name}' not found in DAG")

        node = self.nodes[name]
        if not isinstance(node, Model):
            raise ValueError(
                f"Node '{name}' is not a Model, it's a {type(node).__name__}"
            )

        return node

    @validate_call
    def get_resolver(self, name: StepName) -> Resolver:
        """Get a resolver by name from the DAG."""
        if name not in self.nodes:
            raise ValueError(f"Resolver '{name}' not found in DAG")

        node = self.nodes[name]
        if not isinstance(node, Resolver):
            raise ValueError(
                f"Node '{name}' is not a Resolver, it's a {type(node).__name__}"
            )

        return node

    def query(self, *args: Any, **kwargs: Any) -> Query:
        """Create Query object."""
        return Query(*args, **kwargs, dag=self)

    def draw(
        self,
        status: DAGExecutionStatus | None = None,
        mode: Literal["tree", "list"] = "tree",
    ) -> str:
        """Create a string representation of the DAG.

        In tree mode, nodes are shown in a dependency tree.

        In list mode, nodes are shown in execution order as a numbered list.

        If `status` is provided, it will show the status of each node.
        The status indicators are:

        * ✅ Done
        * 🔄 Working
        * ⏸️ Awaiting
        * ⏭️ Skipped

        Node type indicators are:

        * 💎 Resolver
        * ⚙️ Model
        * 📄 Source

        Args:
            status: Object describing the status of each node.
            mode: "tree" renders the DAG as a tree structure (default).
                "list" renders nodes in flat execution order.

        Returns:
            String representation of the DAG with status indicators.
        """
        # Handle empty DAG
        if not self.nodes:
            return "Empty DAG"

        step_numbers: dict[str, int] = {
            name: i + 1 for i, name in enumerate(self.sequence)
        }

        def _get_node_indicator(name: str) -> str:
            """Determine the status indicator for a node."""
            if name not in status:
                return "⏸️"
            elif status[name] == DAGNodeExecutionStatus.SKIPPED:
                return "⏭️"
            elif status[name] == DAGNodeExecutionStatus.DOING:
                return "🔄"
            elif status[name] == DAGNodeExecutionStatus.DONE:
                return "✅"

        def _get_node_type_indicator(name: str) -> str:
            """Determine the type indicator for a node."""
            node = self.nodes[name]
            if isinstance(node, Resolver):
                return "💎"
            if isinstance(node, Model):
                return "⚙️"
            return "📄"

        def _format_node(name: str) -> str:
            """Format node display with step number, status (if present) and type."""
            type_indicator = _get_node_type_indicator(name)
            step = f"[{step_numbers[name]}]" if name in step_numbers else ""
            if status is None:
                return f"{type_indicator} {name} {step}"

            indicator = _get_node_indicator(name)
            return f"{indicator}{type_indicator} {name} {step}"

        header: list[str] = [
            f"Collection: {self.name}",
            f"└── Run: {self._run or '⛓️‍💥 Disconnected'}",
            "",
        ]

        # List mode

        if mode == "list":
            lines: list[str] = header.copy()
            for name in self.sequence:
                step = step_numbers[name]
                type_indicator = _get_node_type_indicator(name)
                if status is None:
                    lines.append(f"{step}. {type_indicator} {name}")
                else:
                    lines.append(
                        f"{step}. {_get_node_indicator(name)}{type_indicator} {name}"
                    )
            return "\n".join(lines)

        # Tree mode

        apex_nodes = self.final_steps
        if not apex_nodes:
            return "No apex nodes found (possible cycle in DAG)"

        result: list[str] = header.copy()

        def format_children(
            node: str, prefix: str = "", ancestors: set[str] | None = None
        ) -> None:
            """Recursively format the children of a node."""
            if ancestors is None:
                ancestors = {node}

            children = self.graph.get(node, [])

            # Format each child
            for i, child in enumerate(children):
                is_last = i == len(children) - 1

                child_display = _format_node(child)

                branch = "└──" if is_last else "├──"
                child_prefix = prefix + ("    " if is_last else "│   ")

                if child in ancestors:
                    result.append(f"{prefix}{branch} {child_display} (cycle)")
                    continue

                result.append(f"{prefix}{branch} {child_display}")
                format_children(child, child_prefix, ancestors | {child})

        # Draw each apex node
        for i, apex_node in enumerate(apex_nodes):
            root_name = apex_node.name

            if i > 0:
                result.append("")  # Blank line between disconnected components

            result.append(_format_node(root_name))

            format_children(root_name, ancestors={root_name})

        return "\n".join(result)

    def __str__(self) -> str:
        """Return string representation of the DAG."""
        return self.draw()

    def new_run(self) -> Self:
        """Start a new local run.

        Retained for API compatibility. In local mode this simply resets the run
        pointer; artifacts are content-addressed in the adapter.
        """
        self.run = RunID(1)
        return self

    def set_client(self, client: Any) -> Self:  # noqa: ANN401
        """Assign a client to all sources at once."""
        for node in self.nodes.values():
            if isinstance(node, Source):
                node.location.set_client(client)

        return self

    def load_default(self) -> Self:
        """Load a previously-collected DAG from storage (not yet supported locally)."""
        raise NotImplementedError(
            "Loading a DAG from storage is not yet supported in local mode."
        )

    def load_pending(self) -> Self:
        """Load a pending run from storage (not yet supported locally)."""
        raise NotImplementedError(
            "Loading a DAG from storage is not yet supported in local mode."
        )

    def run_and_sync(
        self,
        start: str | None = None,
        finish: str | None = None,
        low_memory: bool = False,
        batch_size: int | None = None,
        profile: bool = False,
    ) -> None:
        """Run entire DAG and send results to server.

        Args:
            start: Name of first node to run
            finish: Name of last node to run
            low_memory: Whether to delete data for each node after it is run
            batch_size: The size used for internal batching. Overrides environment
                variable if set.
            profile: whether to log to INFO level the memory usage
        """
        if batch_size is None:
            batch_size = settings.batch_size

        sequence: list[StepName] = self.sequence

        # Identify skipped nodes
        skipped_nodes = []
        if start:
            try:
                start_index = sequence.index(start)
                skipped_nodes = sequence[:start_index]
            except ValueError as e:
                raise ValueError(f"Step {start} not in DAG") from e
        else:
            start_index = 0

        # Determine end index
        if finish:
            try:
                end_index = sequence.index(finish) + 1
                skipped_nodes.extend(sequence[end_index:])
            except ValueError as e:
                raise ValueError(f"Step {finish} not in DAG") from e
        else:
            end_index = len(sequence)

        sequence = sequence[start_index:end_index]

        status = {
            step_name: DAGNodeExecutionStatus.SKIPPED for step_name in skipped_nodes
        }

        for step_name in sequence:
            node = self.nodes[step_name]
            status[step_name] = DAGNodeExecutionStatus.DOING
            logger.info("\n" + self.draw(status=status))
            try:
                if isinstance(node, Source):
                    node.run(batch_size=batch_size)
                elif isinstance(node, Model):
                    node.run(low_memory=low_memory)
                else:
                    node.run()
                node.sync()
                if profile:
                    log_mem_usage()
            except Exception as e:
                logger.error(f"❌ {node.name} failed: {e}")
                raise e
            status[step_name] = DAGNodeExecutionStatus.DONE

            if low_memory:
                node.clear_data()
                logger.info("Cleared node data")
                if profile:
                    log_mem_usage()
        logger.info("\n" + self.draw(status=status))

    def set_default(self) -> None:
        """Validate that the DAG resolves to a single apex resolver.

        Retained for API compatibility. In local mode there is no server-side default
        pointer; this only performs the structural checks.
        """
        # Trigger error if there isn't a single root
        _ = self.default_resolver

        # Trigger error if some sources aren't connected
        if len(self.final_steps) != 1:
            raise ValueError(
                "Found unreachable steps: all steps must be reachable from a "
                "single final resolver before setting the default run."
            )

    def lookup_key(
        self,
        from_source: str,
        to_sources: list[str],
        key: str,
    ) -> dict[str, list[str]]:
        """Matches IDs against the selected backend.

        Args:
            from_source: Name of source the provided key belongs to
            to_sources: Names of sources to find keys in
            key: The value to match from the source. Usually a primary key

        Returns:
            Dictionary mapping source names to list of keys within that source.

        Examples:
            ```python
            dag.lookup_key(
                from_source="companies_house",
                to_sources=[
                    "datahub_companies",
                    "hmrc_exporters",
                ]
                key="8534735",
            )
            ```
        """
        resolution = self._read_resolution(self.default_resolver)

        origin = resolution.filter(
            (pl.col("source") == from_source) & (pl.col("key") == key)
        )
        if origin.height == 0:
            raise MatchboxStepNotFoundError(
                f"Key '{key}' not found in source '{from_source}'"
            )

        root = origin["root"][0]
        cluster = resolution.filter(pl.col("root") == root)

        def keys_in(source_name: str) -> list[str]:
            return cluster.filter(pl.col("source") == source_name)["key"].to_list()

        result: dict[str, list[str]] = {from_source: keys_in(from_source)}
        for target in to_sources:
            result[target] = keys_in(target)
        return result

    @validate_call
    @profile_time(kwarg="node")
    def get_matches(
        self,
        resolver: ResolverStepName | None = None,
        source_filter: list[SourceStepName] | None = None,
        location_names: list[str] | None = None,
    ) -> ResolverMatches:
        """Return ResolverMatches, optionally filtering.

        Args:
            resolver: Name of resolver to query within DAG.
                If not provided, will look for an apex.
            source_filter: An optional list of source step names to filter by.
            location_names: An optional list of location names to filter by.
        """
        resolver = self.get_resolver(resolver) if resolver else self.default_resolver
        if not isinstance(resolver, Resolver):
            raise ValueError("get_matches can only query from resolver nodes")

        available_sources = {
            node_name: self.get_source(node_name) for node_name in resolver.sources
        }

        filtered_source_names = list(available_sources.keys())

        if source_filter:
            filtered_source_names = [
                s for s in filtered_source_names if s in source_filter
            ]

        if location_names:
            filtered_source_names = [
                s
                for s in filtered_source_names
                if available_sources[s].config.location_config.name in location_names
            ]

        if not filtered_source_names:
            raise MatchboxStepNotFoundError("No compatible source was found")

        resolution = self._read_resolution(resolver)

        resolved_sources: list[Source] = []
        query_results: list[pl.DataFrame] = []
        for source_name in filtered_source_names:
            resolved_sources.append(available_sources[source_name])
            query_results.append(
                resolution.filter(pl.col("source") == source_name).select(
                    pl.col("root").alias("id"),
                    pl.col("key"),
                    pl.col("leaf").alias("leaf_id"),
                )
            )

        return ResolverMatches(sources=resolved_sources, query_results=query_results)

    def _read_resolution(self, resolver: Resolver) -> pl.DataFrame:
        """Read a resolver's stored resolution `(root, leaf, key, source)`."""
        if resolver._fp is None:
            raise MatchboxStepNotFoundError(
                f"Resolver '{resolver.name}' has not been run. Call run_and_sync first."
            )
        return self.adapter.read_resolver(resolver._fp)
