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
