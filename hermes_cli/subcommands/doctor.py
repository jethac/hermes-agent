"""``hermes doctor`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_doctor_parser(subparsers, *, cmd_doctor: Callable) -> None:
    """Attach the ``doctor`` subcommand to ``subparsers``."""
    # =========================================================================
    # doctor command
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check configuration and dependencies",
        description="Diagnose issues with Hermes Agent setup",
    )
    doctor_parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix issues automatically"
    )
    doctor_parser.add_argument(
        "--realtime-voice",
        action="store_true",
        help="Treat realtime voice live-conversation readiness as a required doctor gate",
    )
    doctor_parser.add_argument(
        "--realtime-voice-smoke",
        action="store_true",
        help="Run the strict realtime voice gate plus a sidecar websocket protocol smoke",
    )
    doctor_parser.add_argument(
        "--realtime-voice-audio-fixture",
        metavar="PATH",
        default=None,
        help="Run the strict realtime voice gate plus a sidecar audio fixture smoke",
    )
    doctor_parser.add_argument(
        "--realtime-voice-audio-codec",
        choices=("webm_opus", "opus", "pcm16"),
        default="webm_opus",
        help="Codec for --realtime-voice-audio-fixture (default: webm_opus)",
    )
    doctor_parser.add_argument(
        "--ack",
        metavar="ADVISORY_ID",
        default=None,
        help=(
            "Acknowledge a security advisory by ID and exit. After ack, the "
            "advisory will no longer trigger startup banners. Run `hermes "
            "doctor` first to see active advisories and their IDs."
        ),
    )
    doctor_parser.set_defaults(func=cmd_doctor)
