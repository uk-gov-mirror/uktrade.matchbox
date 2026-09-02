# Query the result

Once a resolver is collected, its output is a table you can read directly. Nothing is resolved at query time. Everything was computed when you collected.

## The entities

```python
companies = plan.collect()
companies.entities()
```

| `root` | `leaf` | `key` | `source` |
|---|---|---|---|
| 4212… | 8871… | `a1` | `crn` |
| 4212… | 9034… | `a2` | `crn` |
| 4212… | 1180… | `b1` | `dh` |
| 7781… | 4402… | `a3` | `crn` |

One row per source record. `root` is the ID of the entity it resolved to, `leaf` its content-addressed record identity, and `key` its key in the original source.

This table is the key output of your entity resolution work. The table is complete, flat, and analysts can point plain SQL at it in the DuckDB store. Everything below is a projection of it.

Restrict it to the sources you care about:

```python
companies.entities(sources=["crn", "dh"])
```

## Looking up one record

The common operational question: *given this record, what else is the same entity?*

```python
companies.lookup_key(
    from_source="crn",
    to_sources=["dh", "cdms"],
    key="a1",
)
# {"crn": ["a1", "a2"], "dh": ["b1"], "cdms": []}
```

## Other shapes

```python
companies.get_lookup()  # one column of keys per source, joined on entity
companies.leaf_sets()  # entities as lists of record identities
```

`get_lookup()` is the wide form: a `root` column plus one qualified-key column per source, with nulls where a source has no record in that entity. `leaf_sets()` drops the IDs entirely, so two resolver outputs of the same records can be compared by structure alone. Both take the same `sources` filter as `entities()`.

## Inspecting one entity

```python
companies.view_entity(root, merge_fields=True)
```

Returns the underlying rows for every record in that entity, so you can see what was matched and judge whether it should have been. `merge_fields=True` collapses the source-qualified columns onto shared names, which makes cross-source entities readable. Key columns stay qualified, since they're what tells you where a row came from.

Values come from the extract cached when each source was collected, the data the matching actually saw. This needs no warehouse connection, and it agrees with what the [reviewer](./evaluate.md) puts on screen.
