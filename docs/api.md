# API reference

The public API is the `Stabilizer` class, importable as `from stabilo import Stabilizer`. A typical session calls `set_ref_frame()` once, then `stabilize()` per frame, and reads the result with `warp_cur_frame()`, `transform_cur_boxes()`, or `get_cur_trans_matrix()`. The estimated matrix maps the **current** frame onto the **reference** frame.

The class attributes named `VALID_*` list the accepted values of the corresponding options, for example `VALID_DETECTORS` for `detector_name`, `VALID_MATCHERS` for `matcher_name`, and `VALID_DEVICES` for `device`. They are checked when the `Stabilizer` is created, so they are the de-facto parameter contract.

::: stabilo.Stabilizer
    options:
      docstring_style: google
      merge_init_into_class: true
      show_root_heading: true
      show_root_full_path: false
      show_source: false
      show_if_no_docstring: true
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      separate_signature: true
      members_order: source
      filters: ["!^_"]
      heading_level: 2

## Configurable keyword arguments

`Stabilizer(**kwargs)` accepts exactly the keys of `stabilo/cfg/default.yaml`, reproduced below with their defaults and valid ranges. The same keys can be set in a file passed to the CLI's `--custom-config`. A keyword argument that is not one of these keys is ignored with a warning that names the nearest valid key; `Stabilizer.configurable_keys()` returns the valid names. The only other keyword argument is `logger`, an optional `logging.Logger` that receives all of stabilo's messages.

```yaml
--8<-- "stabilo/cfg/default.yaml"
```

The meaning of each key is described in [Usage: Configuration parameters](usage.md#8-configuration-parameters).
