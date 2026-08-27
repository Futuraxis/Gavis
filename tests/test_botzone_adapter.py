from __future__ import annotations

import json
import zipfile

from layer4_interface.botzone.runner import decide
from layer4_interface.botzone.texas_holdem import legal_responses, parse_texas_holdem_request, reconstruct_betting_state
from scripts.build_botzone_zip import build


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
    assert decision.debug.startswith("texas-holdem:")


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
