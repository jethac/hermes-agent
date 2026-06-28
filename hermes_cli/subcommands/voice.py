"""``hermes voice`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_voice_parser(subparsers, *, cmd_voice: Callable) -> None:
    """Attach realtime voice operational commands to ``subparsers``."""

    voice_parser = subparsers.add_parser(
        "voice",
        help="Realtime voice tools and launch profiles",
        description="Generate and check Hermes realtime voice launch profiles.",
    )
    voice_subparsers = voice_parser.add_subparsers(dest="voice_command")

    dgx_parser = voice_subparsers.add_parser(
        "dgx-spark",
        help="Generate a headless DGX Spark KAME launch/preflight pack",
        description=(
            "Generate compose, env, launch, benchmark, and preflight artifacts "
            "for a DGX Spark KAME voice stack."
        ),
    )
    from hermes_cli.realtime_voice_dgx_spark import add_dgx_spark_arguments

    add_dgx_spark_arguments(dgx_parser)
    dgx_parser.set_defaults(func=cmd_voice, voice_command="dgx-spark")
    voice_parser.set_defaults(func=cmd_voice)
