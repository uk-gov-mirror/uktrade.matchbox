Generators for test data with known ground truth. Used throughout matchlab's own test
suite, and useful for sanity-checking a methodology before pointing it at real data.

::: matchlab.core.factories.scenarios
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

::: matchlab.core.factories.sources
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

::: matchlab.core.factories.entities
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

::: matchlab.core.factories.models
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
