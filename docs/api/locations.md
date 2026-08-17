# Locations

The methodologies a [source](sources.md) reads through. A source wraps exactly one location, named by `location_class` and configured by `location_settings`, with its client or frame passed separately as a [resource](resources.md).

::: matchlab.sources.base
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"

## Relational databases

::: matchlab.sources.relational
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"

## Dataframes

::: matchlab.sources.dataframe
    options:
        show_root_heading: true
        show_root_full_path: true
        members_order: source
        show_if_no_docstring: true
        docstring_style: google
        show_signature_annotations: true
        separate_signature: true
        filters:
            - "!^[A-Z]$"
            - "!^_"
