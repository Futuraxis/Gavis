"""Botzone Local AI long-polling client.

Botzone's Local AI API is the opposite direction from the upload zip:
this process runs on the user's machine, polls Botzone for match requests,
then responds in ``X-Match-<match_id>`` headers on the next poll.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from layer4_interface.botzone.runner import decide


@dataclass
class LocalAIMatch:
    match_id: str
    requests: list[Any] = field(default_factory=list)
    responses: list[Any] = field(default_factory=list)
    pending_response: Any | None = None
    has_unsubmitted_response: bool = False

    def add_request(self, raw_request: str) -> None:
        self.requests.append(_decode_request(raw_request))
        self.pending_response = None
        self.has_unsubmitted_response = False

    def decide(self) -> Any:
        if self.has_unsubmitted_response:
            return self.pending_response
        try:
            decision = decide(self._decision_payload())
            self.pending_response = decision.response
        except Exception as exc:  # noqa: BLE001 - Local AI must keep polling.
            self.pending_response = _fallback_response(self.requests)
            latest = self.requests[-1] if self.requests else None
            print(
                "Botzone localai decision failed for "
                f"{self.match_id}: {type(exc).__name__}: {exc}; "
                f"latest_request={_short_repr(latest)}; fallback={_header_value(self.pending_response)}",
                file=sys.stderr,
                flush=True,
            )
        self.has_unsubmitted_response = True
        return self.pending_response

    def mark_submitted(self) -> None:
        if self.has_unsubmitted_response:
            self.responses.append(self.pending_response)
        self.has_unsubmitted_response = False

    def _decision_payload(self) -> dict[str, Any]:
        latest = self.requests[-1] if self.requests else None
        if isinstance(latest, dict) and isinstance(latest.get("requests"), list):
            return latest
        if isinstance(latest, list):
            responses = latest[1::2] if latest and len(latest) > 1 else self.responses
            requests = latest[0::2] if latest and len(latest) > 1 else latest
            return {"requests": requests, "responses": responses}
        return {"requests": self.requests, "responses": self.responses}


def run(
    url: str,
    *,
    poll_interval: float = 2.0,
    once: bool = False,
    create_game: str | None = None,
    players: list[str] | None = None,
    initdata: str = "",
) -> None:
    matches: dict[str, LocalAIMatch] = {}
    if create_game:
        created = runmatch(url, create_game, players or [], initdata)
        print(f"created match: {created}", flush=True)

    while True:
        try:
            submitted = fetch_once(url, matches)
            for match_id in submitted:
                print(f"submitted response for {match_id}", flush=True)
            for match in matches.values():
                if match.requests and not match.has_unsubmitted_response:
                    response = match.decide()
                    print(f"response ready for {match.match_id}: {_header_value(response)}", flush=True)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, http.client.RemoteDisconnected) as exc:
            print(f"Botzone localai poll failed: {exc}; retrying in {poll_interval:g}s", file=sys.stderr, flush=True)
            time.sleep(poll_interval)
        if once:
            return


def fetch_once(url: str, matches: dict[str, LocalAIMatch]) -> list[str]:
    req = urllib.request.Request(url)
    submitted: list[str] = []
    pending_matches: list[LocalAIMatch] = []
    for match_id, match in list(matches.items()):
        if match.has_unsubmitted_response:
            req.add_header(f"X-Match-{match_id}", _header_value(match.pending_response))
            submitted.append(match_id)
            pending_matches.append(match)

    with urllib.request.urlopen(req, timeout=None) as resp:
        text = resp.read().decode("utf-8")
    for match in pending_matches:
        match.mark_submitted()
    _process_botzone_text(text, matches)
    return submitted


def runmatch(localai_url: str, game: str, players: list[str], initdata: str = "") -> str:
    if not players or players.count("me") != 1:
        raise ValueError("players must contain exactly one 'me'")
    req = urllib.request.Request(_runmatch_url(localai_url))
    req.add_header("X-Game", game)
    for idx, player in enumerate(players):
        req.add_header(f"X-Player-{idx}", player)
    if initdata:
        req.add_header("X-Initdata", initdata)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8").strip()


def _process_botzone_text(text: str, matches: dict[str, LocalAIMatch]) -> None:
    lines = text.splitlines()
    if not lines:
        return
    try:
        request_count, result_count = [int(x) for x in lines[0].split()[:2]]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"bad Botzone localai header: {lines[0]!r}") from exc

    idx = 1
    for _ in range(request_count):
        match_id = lines[idx].strip()
        raw_request = lines[idx + 1]
        idx += 2
        match = matches.setdefault(match_id, LocalAIMatch(match_id))
        match.add_request(raw_request)
        print(f"request for {match_id}: {raw_request[:160]}", flush=True)

    for _ in range(result_count):
        if idx >= len(lines):
            break
        parts = lines[idx].split()
        idx += 1
        if not parts:
            continue
        match_id = parts[0]
        matches.pop(match_id, None)
        print(f"match finished: {' '.join(parts)}", flush=True)


def _decode_request(raw_request: str) -> Any:
    text = raw_request.strip()
    if not text:
        return raw_request
    if text[0] in '[{"':
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return raw_request
    return raw_request


def _header_value(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, (str, int, float)):
        return str(response)
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def _fallback_response(requests: list[Any]) -> Any:
    latest = requests[-1] if requests else None
    if isinstance(latest, dict):
        if {"num_players", "my_id", "my_chips", "my_cards", "history"}.issubset(latest):
            return 0
        if isinstance(latest.get("requests"), list):
            return _fallback_response(list(latest["requests"]))
    if isinstance(latest, list):
        return _fallback_response(latest)
    return "PASS"


def _short_repr(value: Any, limit: int = 500) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _runmatch_url(localai_url: str) -> str:
    if localai_url.rstrip("/").endswith("/localai"):
        return localai_url.rstrip("/")[: -len("/localai")] + "/runmatch"
    return localai_url.rstrip("/") + "/runmatch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect local Gavis solvers to Botzone Local AI")
    parser.add_argument(
        "--url",
        default=os.environ.get("BOTZONE_LOCALAI_URL", ""),
        help="Botzone Local AI URL: https://www.botzone.org/api/<uid>/<secret>/localai; can also use BOTZONE_LOCALAI_URL",
    )
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="poll once, useful for smoke tests")
    parser.add_argument("--create-game", help="optional Botzone game name for /runmatch")
    parser.add_argument("--player", action="append", default=[], help="runmatch player slot; use exactly one 'me'")
    parser.add_argument("--initdata", default="", help="optional runmatch X-Initdata header")
    args = parser.parse_args()
    if not args.url:
        parser.error("--url or BOTZONE_LOCALAI_URL is required")
    run(
        args.url,
        poll_interval=args.poll_interval,
        once=args.once,
        create_game=args.create_game,
        players=args.player,
        initdata=args.initdata,
    )


if __name__ == "__main__":
    main()
