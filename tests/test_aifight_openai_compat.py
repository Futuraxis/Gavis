from __future__ import annotations

import json

from layer4_interface.aifight import openai_compat


def test_aifight_compat_returns_botzone_decision_from_chat_prompt() -> None:
    payload = {
        "model": "gavis-local",
        "messages": [
            {
                "role": "user",
                "content": '请决策：```json\n{"requests":["0 1 2"],"responses":[]}\n```',
            }
        ],
    }

    content = openai_compat._decide_chat_completion(payload)

    assert json.loads(content) == "PASS"


def test_aifight_compat_returns_mahjong_layer3_decision_from_chat_prompt() -> None:
    payload = {
        "model": "gavis-local",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "requests": [
                            "0 0 1",
                            "1 0 0 1 1 B4 W6 T5 F2 W1 T9 B2 B3 W7 W9 W8 B8 W1 H2 H1",
                            "2 F3",
                        ],
                        "responses": ["PASS", "PASS"],
                    },
                    ensure_ascii=False,
                ),
            }
        ],
    }

    content = openai_compat._decide_chat_completion(payload)

    assert json.loads(content).startswith("PLAY ")


def test_aifight_compat_returns_texas_layer3_decision_from_chat_prompt() -> None:
    payload = {
        "model": "gavis-local",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
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
                    },
                    ensure_ascii=False,
                ),
            }
        ],
    }

    content = openai_compat._decide_chat_completion(payload)

    assert isinstance(json.loads(content), int)


def test_aifight_compat_chooses_from_legal_actions() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": '{"state":{"turn":"p0"},"legal_actions":[{"action":"fold"},{"action":"call"},{"action":"raise","amount":20}]}',
            }
        ]
    }

    content = openai_compat._decide_chat_completion(payload)

    assert json.loads(content) == {"action": "call"}


def test_aifight_compat_chat_completion_envelope_shape() -> None:
    envelope = openai_compat._chat_completion_envelope({"model": "gavis-local"}, '{"action":"pass"}', "gavis-local")

    assert envelope["object"] == "chat.completion"
    assert envelope["choices"][0]["message"]["role"] == "assistant"
    assert envelope["choices"][0]["message"]["content"] == '{"action":"pass"}'
