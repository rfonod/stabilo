# CLI reference

Installing stabilo provides the `stabilo` console command with three subcommands: `video` stabilizes a video, `tracks` stabilizes per-frame object annotations, and `config` prints or copies the default configuration.

Every `Stabilizer` option in `stabilo/cfg/default.yaml` can be set through a file passed to `--custom-config`, and most also have a dedicated flag. Precedence is **explicit CLI flags > custom config file > built-in defaults**. See [Usage: Command-Line Interface](usage.md#15-command-line-interface) for the workflow and the [API reference](api.md#configurable-keyword-arguments) for every key.

The help texts below are generated from the installed parsers when the site is built, so they always match the released command. On the command line, run `stabilo <subcommand> --help` to see the same text.

## stabilo

<!-- cli-help: -->

## stabilo video

<!-- cli-help: video -->

## stabilo tracks

<!-- cli-help: tracks -->

## stabilo config

<!-- cli-help: config -->
