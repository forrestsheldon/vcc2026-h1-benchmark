from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from . import __version__
from .paths import BenchmarkPaths


def _data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path)


def _paths(args) -> BenchmarkPaths:
    paths = BenchmarkPaths.resolve(args.data_dir)
    print(f"Data directory: {paths.root}", flush=True)
    return paths


def _score_args(args, paths: BenchmarkPaths) -> SimpleNamespace:
    return SimpleNamespace(
        prediction=getattr(args, "prediction", None),
        output=getattr(args, "output", None),
        gene_chunk=getattr(args, "gene_chunk", 512),
        de_threads=getattr(args, "de_threads", 4),
        target_counts=paths.target_counts,
        reference_cells=paths.reference_cells,
        manifest=paths.benchmark_manifest,
        reference_cache=paths.reference_cache,
        scale_bundle=paths.scale,
        controls=paths.controls,
        controls_manifest=paths.controls_manifest,
        score_cache=paths.score_cache,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    show_version = commands.add_parser("version")
    _data_argument(show_version)

    prepare = commands.add_parser("setup")
    _data_argument(prepare)
    prepare.add_argument("--h1", type=Path)
    prepare.add_argument("--remove-source", action="store_true")
    prepare.add_argument("--asset", type=Path, help=argparse.SUPPRESS)

    inspect = commands.add_parser("check")
    _data_argument(inspect)

    validate = commands.add_parser("validate")
    validate.add_argument("prediction", type=Path)
    _data_argument(validate)

    scoring = commands.add_parser("score")
    scoring.add_argument("prediction", type=Path)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--gene-chunk", type=int, default=512)
    scoring.add_argument("--de-threads", type=int, default=4)
    _data_argument(scoring)

    control = commands.add_parser("score-control-baseline")
    control.add_argument("--output", type=Path, required=True)
    control.add_argument("--gene-chunk", type=int, default=512)
    control.add_argument("--de-threads", type=int, default=4)
    _data_argument(control)

    plot = commands.add_parser("plot")
    plot.add_argument(
        "results",
        nargs="+",
        help="scoring result directories, optionally written as LABEL=PATH",
    )
    plot.add_argument("--targets", nargs="+")
    plot.add_argument("--output", type=Path, required=True)
    _data_argument(plot)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    paths = _paths(args)
    if args.command == "version":
        print(f"vcc2026-h1-benchmark {__version__}")
        print(f"cell-eval2 {version('cell-eval2')}")
        print(f"pdex {version('pdex')}")
    elif args.command == "setup":
        from .artifacts import setup

        result = setup(
            paths,
            h1=args.h1,
            remove_source=args.remove_source,
            asset_path=args.asset,
        )
        print(f"Ready: benchmark {result['benchmark_version']}")
    elif args.command == "check":
        from .artifacts import check

        result = check(paths)
        print(f"Ready: benchmark {result['benchmark_version']}")
    elif args.command == "validate":
        from .artifacts import check
        from .scorer import validate_command

        check(paths)
        validate_command(_score_args(args, paths))
    elif args.command == "score":
        from .artifacts import check
        from .scorer import score

        check(paths)
        score(_score_args(args, paths))
    elif args.command == "score-control-baseline":
        from .artifacts import check
        from .scorer import score_control_baseline

        check(paths)
        score_control_baseline(_score_args(args, paths))
    else:
        from .artifacts import check
        from .plot import parse_results, plot_metric_profiles, reference_de_path

        check(paths)
        table = plot_metric_profiles(
            parse_results(args.results),
            paths.reference_cells,
            reference_de_path(paths.benchmark_dir),
            args.output,
            args.targets,
        )
        print(f"Wrote {args.output} and {args.output.with_suffix('.csv')}")
        print(f"Plotted {table['target_gene'].nunique()} perturbations")


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
