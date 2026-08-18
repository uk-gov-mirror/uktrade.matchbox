# Models

The `Model` step, and the methodologies it runs. A model wraps exactly one [deduper](#matchlab.models.dedupers) or [linker](#matchlab.models.linkers), named by `model_class` and configured by `model_settings`.

::: matchlab.models
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

## Dedupers

::: matchlab.models.dedupers
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

## Linkers

::: matchlab.models.linkers
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
