from __future__ import annotations

import json
import http.client
import urllib.error
import urllib.request
import zipfile
from io import BytesIO

import pytest

from layer4_interface.botzone import localai
from layer4_interface.botzone.runner import decide
from layer4_interface.botzone.server import make_handler
from layer4_interface.botzone.texas_holdem import legal_responses, parse_texas_holdem_request, reconstruct_betting_state
from scripts.build_botzone_zip import build


class _FakeHTTPResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._text.encode("utf-8")


def test_botzone_decide_from_json_envelope() -> None:
    payload = {
        "requests": [
            json.dumps(
                {
                    "game_id": "moon_chess",
                    "player_id": "p_black",
                    "solver": "mcts",
                    "seed": 42,
                    "budget": 5,
                },
                ensure_ascii=False,
            )
        ],
        "responses": [],
    }

    decision = decide(payload)

    assert decision.response is not None
    assert decision.response["template_id"] == "place_piece"
    assert decision.response["player_id"] == "p_black"
    assert "canonical_key" in decision.response


def test_build_botzone_zip_has_root_main(tmp_path) -> None:
    out = tmp_path / "gavis_botzone.zip"

    build(out)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "__main__.py" in names
    assert len(names) == 1
    assert not any(name.startswith("tests/") for name in names)


def test_build_botzone_zip_injects_remote_endpoint(tmp_path) -> None:
    out = tmp_path / "gavis_botzone.zip"

    build(out, remote_url="https://example.test/botzone/decide", remote_token="secret", remote_timeout=0.42)

    with zipfile.ZipFile(out) as zf:
        main_py = zf.read("__main__.py").decode("utf-8")
    assert 'REMOTE_URL = "https://example.test/botzone/decide"' in main_py
    assert 'REMOTE_TOKEN = "secret"' in main_py
    assert "REMOTE_TIMEOUT = 0.42" in main_py


def test_localai_processes_requests_and_submits_next_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    matches: dict[str, localai.LocalAIMatch] = {}
    seen_headers: list[dict[str, str]] = []

    def fake_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeHTTPResponse:
        del timeout
        seen_headers.append({key.lower(): value for key, value in req.header_items()})
        if len(seen_headers) == 1:
            return _FakeHTTPResponse('1 0\nm1\n{"requests":["0 1 2"],"responses":[]}\n')
        return _FakeHTTPResponse("0 0\n")

    monkeypatch.setattr(localai.urllib.request, "urlopen", fake_urlopen)

    assert localai.fetch_once("https://botzone.test/api/u/s/localai", matches) == []
    assert matches["m1"].requests == [{"requests": ["0 1 2"], "responses": []}]

    matches["m1"].pending_response = "PASS"
    matches["m1"].has_unsubmitted_response = True
    assert localai.fetch_once("https://botzone.test/api/u/s/localai", matches) == ["m1"]

    assert seen_headers[1]["x-match-m1"] == "PASS"
    assert matches["m1"].responses == ["PASS"]
    assert not matches["m1"].has_unsubmitted_response


def test_localai_does_not_mark_response_submitted_on_poll_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    match = localai.LocalAIMatch("m1", requests=["0 1 2"], pending_response="PASS", has_unsubmitted_response=True)

    def failing_urlopen(req: urllib.request.Request, timeout: object = None) -> _FakeHTTPResponse:
        del req, timeout
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(localai.urllib.request, "urlopen", failing_urlopen)

    with pytest.raises(urllib.error.URLError):
        localai.fetch_once("https://botzone.test/api/u/s/localai", {"m1": match})

    assert match.responses == []
    assert match.pending_response == "PASS"
    assert match.has_unsubmitted_response


def test_localai_run_retries_remote_disconnected(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    def disconnected(req: urllib.request.Request, timeout: object = None) -> _FakeHTTPResponse:
        del req, timeout
        raise http.client.RemoteDisconnected("closed")

    monkeypatch.setattr(localai.urllib.request, "urlopen", disconnected)
    monkeypatch.setattr(localai.time, "sleep", sleeps.append)

    localai.run("https://botzone.test/api/u/s/localai", poll_interval=0.25, once=True)

    assert sleeps == [0.25]


def test_localai_runmatch_url_and_header_values() -> None:
    assert localai._runmatch_url("https://www.botzone.org/api/u/secret/localai") == "https://www.botzone.org/api/u/secret/runmatch"
    assert localai._header_value(200) == "200"
    assert localai._header_value({"response": "PLAY W1"}) == '{"response":"PLAY W1"}'


def test_localai_can_decide_from_full_botzone_envelope_request() -> None:
    match = localai.LocalAIMatch("m1")
    match.add_request('{"requests":["0 1 2"],"responses":[]}')

    assert match.decide() == "PASS"


def test_localai_can_decide_from_json_array_request() -> None:
    match = localai.LocalAIMatch("m1")
    match.add_request('["0 1 2"]')

    assert match.decide() == "PASS"


def test_localai_decodes_quoted_mahjong_requests() -> None:
    match = localai.LocalAIMatch("m1")
    match.add_request('"0 0 1"')
    assert match.decide() == "PASS"
    match.mark_submitted()
    match.add_request('"1 0 0 1 1 B4 W6 T5 F2 W1 T9 B2 B3 W7 W9 W8 B8 W1 H2 H1"')
    assert match.decide() == "PASS"
    match.mark_submitted()
    match.add_request('"2 F3"')

    assert match.requests[0] == "0 0 1"
    assert match.decide().startswith("PLAY ")


def test_localai_decision_error_keeps_process_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    match = localai.LocalAIMatch("m1")
    match.add_request('"unknown raw request"')

    def boom(payload: dict) -> object:
        del payload
        raise RuntimeError("bad shape")

    monkeypatch.setattr(localai, "decide", boom)

    assert match.decide() == "PASS"
    assert match.has_unsubmitted_response


def test_localai_current_request_shape_error_falls_back_to_pass() -> None:
    match = localai.LocalAIMatch("m1")
    match.add_request('"unknown raw request"')

    assert match.decide() == "PASS"
    assert match.has_unsubmitted_response


def test_botzone_mahjong_format_handshake_and_draw() -> None:
    first = decide({"requests": ["0 1 2"], "responses": []})
    assert first.response == "PASS"

    second = decide({"requests": ["0 1 2", "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 F3 J1"], "responses": ["PASS"]})
    assert second.response == "PASS"

    third = decide(
        {
            "requests": [
                "0 1 2",
                "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 F3 J1",
                "2 T6",
            ],
            "responses": ["PASS", "PASS"],
        }
    )
    assert isinstance(third.response, str)
    assert third.response.startswith("PLAY ")
    assert third.debug.startswith("mahjong-international/layer3:")


def test_botzone_mahjong_format_claim_priority() -> None:
    gang = decide(
        {
            "requests": [
                "0 1 2",
                "1 0 0 0 0 W1 W1 W1 B1 B2 B3 T1 T2 T3 F1 F2 F3 J1",
                "3 0 PLAY W1",
            ],
            "responses": ["PASS", "PASS"],
        }
    )
    assert gang.response == "GANG"

    peng = decide(
        {
            "requests": [
                "0 1 2",
                "1 0 0 0 0 W1 W1 W3 B1 B2 B3 T1 T2 T3 F1 F2 F3 J1",
                "3 0 PLAY W1",
            ],
            "responses": ["PASS", "PASS"],
        }
    )
    assert isinstance(peng.response, str)
    assert peng.response.startswith("PENG ")

    chi = decide(
        {
            "requests": [
                "0 1 2",
                "1 0 0 0 0 W2 W3 W7 B1 B2 B3 T1 T2 T3 F1 F2 F3 J1",
                "3 0 PLAY W1",
            ],
            "responses": ["PASS", "PASS"],
        }
    )
    assert isinstance(chi.response, str)
    assert chi.response.startswith("CHI W2 ")


def test_botzone_remote_server_decides_mahjong_format() -> None:
    payload = {
        "requests": [
            "0 1 2",
            "1 0 0 0 0 W1 W2 W3 B1 B2 B3 T1 T2 T3 F1 F2 F3 J1",
            "2 T6",
        ],
        "responses": ["PASS", "PASS"],
    }
    body = json.dumps(payload).encode("utf-8")
    handler = object.__new__(make_handler(token="secret"))
    handler.path = "/botzone/decide"
    handler.headers = {"Content-Length": str(len(body)), "Authorization": "Bearer secret"}
    handler.rfile = BytesIO(body)
    sent: list[dict[str, object]] = []

    def send_json(status: object, response_payload: dict[str, object]) -> None:
        sent.append({"status": status, "payload": response_payload})

    handler._send_json = send_json

    handler.do_POST()

    assert sent
    data = sent[0]["payload"]
    assert isinstance(data, dict)
    assert isinstance(data["response"], str)
    assert data["response"].startswith("PLAY ")


def test_botzone_texas_holdem_json_protocol_is_routed_without_mahjong_overlap() -> None:
    payload = {
        "requests": [
            {
                "num_players": 6,
                "dealer_id": 0,
                "my_id": 3,
                "my_chips": 20_000,
                "my_cards": [48, 49],
                "public_cards": [],
                "history": [],
                "hand": 0,
                "max_hand": 18,
                "total_win_chips": [0] * 6,
                "total_win_games": [0] * 6,
            }
        ]
    }

    request = parse_texas_holdem_request(payload)
    betting = reconstruct_betting_state(request)
    decision = decide(payload)

    assert isinstance(decision.response, int)
    assert decision.response in legal_responses(request, betting)
    assert decision.debug.startswith("texas-holdem/")


def test_botzone_texas_holdem_heads_up_uses_layer3() -> None:
    payload = {
        "requests": [
            {
                "num_players": 2,
                "dealer_id": 0,
                "my_id": 1,
                "my_chips": 19_900,
                "my_cards": [48, 49],
                "public_cards": [],
                "history": [],
                "hand": 0,
                "max_hand": 50,
                "total_win_chips": [0, 0],
                "total_win_games": [0, 0],
            }
        ]
    }

    request = parse_texas_holdem_request(payload)
    betting = reconstruct_betting_state(request)
    decision = decide(payload)

    assert decision.response in legal_responses(request, betting)
    assert decision.debug.startswith("texas-holdem/layer3:Hybrid(search):")


def test_botzone_texas_holdem_non_heads_up_falls_back_legally() -> None:
    payload = {
        "requests": [
            {
                "num_players": 6,
                "dealer_id": 0,
                "my_id": 3,
                "my_chips": 20_000,
                "my_cards": [48, 49],
                "public_cards": [],
                "history": [],
                "hand": 0,
                "max_hand": 18,
                "total_win_chips": [0] * 6,
                "total_win_games": [0] * 6,
            }
        ]
    }

    request = parse_texas_holdem_request(payload)
    betting = reconstruct_betting_state(request)
    decision = decide(payload)

    assert isinstance(decision.response, int)
    assert decision.response in legal_responses(request, betting)


def test_botzone_texas_holdem_allin_restricts_response_to_fold_or_allin() -> None:
    payload = {
        "requests": [
            {
                "num_players": 2,
                "dealer_id": 0,
                "my_id": 1,
                "my_chips": 19_900,
                "my_cards": [0, 1],
                "public_cards": [],
                "history": [{"round": 0, "player_id": 0, "action": -2, "action_type": "allin"}],
                "hand": 0,
                "max_hand": 50,
                "total_win_chips": [0, 0],
                "total_win_games": [0, 0],
            }
        ]
    }

    request = parse_texas_holdem_request(payload)
    betting = reconstruct_betting_state(request)
    decision = decide(payload)

    assert legal_responses(request, betting) == frozenset({-1, -2})
    assert decision.response in {-1, -2}
