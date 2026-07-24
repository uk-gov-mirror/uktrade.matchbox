# Query the result

Once a resolver is collected, its output is a table you can read directly — there is no
resolution happening at query time. Everything was computed when you collected.

## The resolution

```python
entities = plan.collect()
entities.resolution()
```

| `root` | `leaf` | `key` | `source` |
|---|---|---|---|
| 4212… | 8871… | `a1` | `crn` |
| 4212… | 9034… | `a2` | `crn` |
| 4212… | 1180… | `b1` | `dh` |
| 7781… | 4402… | `a3` | `crn` |

One row per source record. `root` is the entity it resolved to, `leaf` its
content-addressed record identity, and `key` its key in the original source.

This table is the whole point: it's complete, it's flat, and analysts can point plain
SQL at it in the DuckDB store.

## Looking up one record

The common operational question — *given this record, what else is the same entity?*

```python
entities.lookup_key(
    from_source="crn",
    to_sources=["dh", "cdms"],
    key="a1",
)
# {"crn": ["a1", "a2"], "dh": ["b1"], "cdms": []}
```

## Matches as a table

```python
matches = entities.get_matches()

matches.as_lookup()  # one column of keys per source, joined on entity
matches.as_dump()  # long form: root, leaf, key, source
matches.as_leaf_sets()  # entities as lists of record identities
```

Filter to the sources you care about:

```python
entities.get_matches(source_filter=["crn", "dh"])
```

### Inspecting one entity

```python
matches.view_cluster(cluster_id, merge_fields=True)
```

Fetches the underlying rows for every record in that cluster, so you can see what was
matched and judge whether it should have been. `merge_fields=True` collapses the
source-qualified columns onto shared names, which makes cross-source clusters readable.

### Comparing two resolutions

```python
merged = matches.merge(other_matches)
```

Combines two resolutions of the same sources, unioning entities that share records.
Useful when comparing methodologies.

## Next

[Evaluate the result :octicons-arrow-right-16:](./evaluate.md){ .md-button .md-button--primary }
