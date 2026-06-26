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
        "--discord-realtime-voice-smoke",
        action="store_true",
        help="Run the strict realtime voice gate plus a local Discord voice bridge smoke",
    )
    doctor_parser.add_argument(
        "--discord-voice-live-probe",
        action="store_true",
        help="Join a real Discord voice channel, install the receiver, play mixer audio, and leave",
    )
    doctor_parser.add_argument(
        "--discord-voice-live-probe-require-inbound",
        action="store_true",
        help="With --discord-voice-live-probe, fail unless inbound live speech frames are observed",
    )
    doctor_parser.add_argument(
        "--discord-voice-live-probe-wait-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait in the voice channel during --discord-voice-live-probe",
    )
    doctor_parser.add_argument(
        "--discord-voice-live-probe-channel-id",
        default="",
        help="Voice channel ID for --discord-voice-live-probe; defaults to DISCORD_VOICE_CHANNEL_ID",
    )
    doctor_parser.add_argument(
        "--discord-voice-live-probe-channel-name",
        default="",
        help="Voice channel name for --discord-voice-live-probe; defaults to DISCORD_VOICE_CHANNEL_NAME or General",
    )
    doctor_parser.add_argument(
        "--realtime-voice-alpha",
        action="store_true",
        help=(
            "Run the documented realtime voice private-alpha evidence set: protocol smoke, "
            "required English/Japanese audio fixtures, full audio-session smokes, "
            "required English/Japanese TTS phrases, and barge-in smoke"
        ),
    )
    doctor_parser.add_argument(
        "--realtime-voice-audio-fixture",
        action="append",
        metavar="PATH",
        default=None,
        help="Run the strict realtime voice gate plus a sidecar audio fixture smoke; repeat for multiple fixtures",
    )
    doctor_parser.add_argument(
        "--realtime-voice-audio-codec",
        choices=("webm_opus", "opus", "pcm16"),
        default="webm_opus",
        help="Codec for --realtime-voice-audio-fixture (default: webm_opus)",
    )
    doctor_parser.add_argument(
        "--realtime-voice-tts-smoke",
        action="append",
        metavar="TEXT",
        default=None,
        help="Run the strict realtime voice gate plus a sidecar TTS first-audio smoke; repeat for multiple phrases",
    )
    doctor_parser.add_argument(
        "--realtime-voice-barge-in-smoke",
        action="append",
        metavar="TEXT",
        default=None,
        help="Run the strict realtime voice gate plus a sidecar barge-in acknowledgement smoke; repeat for multiple phrases",
    )
    doctor_parser.add_argument(
        "--realtime-voice-report",
        metavar="PATH",
        default=None,
        help="Write realtime voice smoke results as JSON for CI/release gates",
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
