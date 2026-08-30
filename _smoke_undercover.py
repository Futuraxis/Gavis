"""临时冒烟：平台 undercover 注册 + 完整人机对局（random 求解器，完成后删除）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from layer2_engine.core.llm import LLMClient
from layer4_interface.frontend.platform.games import GAMES
from layer4_interface.frontend.platform.history import MatchHistory
from layer4_interface.frontend.platform.session import _BUILTIN_FAMILY, PlayManager
from train_cli import default_provider

_orig_available = LLMClient.available
LLMClient.available = staticmethod(lambda *a, **k: False)  # 测试/冒烟强制 random 求解器

spec = GAMES["undercover"]
print("game_id:", spec.game_id)
print("display_name:", spec.display_name)
print("kind:", spec.kind, "| family(map):", _BUILTIN_FAMILY["undercover"])
print("seat_options:", len(spec.seat_options), "| player_counts:", spec.player_counts, "| seat_label:", spec.seat_label)
print("difficulty_budgets:", spec.difficulty_budgets)

mgr = PlayManager(
    provider=default_provider,
    history=MatchHistory(Path(tempfile.mkdtemp())),
    seed=42,
)
s = mgr.start("undercover", "p0", "easy")
snap = s.snapshot()
print("start → phase:", snap["phase"], "| turn:", snap["turn"], "| family:", snap["family"], "| ai_mode:", snap["ai_mode"])
print("my_role:", snap["my_role"], "| my_word:", snap["my_word"], "| alive:", len(snap["alive"]))
print("legal:", snap["legal"][:4])

guard = 0
rounds = []
last_round = None
while not s.over and guard < 300:
    snap = s.snapshot()
    if snap["turn"] != snap["player_pid"] or not snap["legal"]:
        print("  stopped early: turn", snap["turn"], "phase", snap["phase"], "legal", len(snap["legal"]))
        break
    r = snap.get("round")
    if r != last_round:
        rounds.append(r)
        last_round = r
    act = snap["legal"][0]
    if act["type"] == "speak":
        payload = {"type": "speak", "text": "一个水果，很常见"}
    else:
        payload = {"type": act["type"], "target": act["target"]}
    mgr.move(s.game_id, payload)
    guard += 1
snap = s.snapshot()
print("final → over:", s.over, "| winner:", snap.get("winner"), "| phase:", snap["phase"], "| rounds seen:", rounds)
print("discourse entries:", len(snap.get("discourse", [])))
print("OK")

LLMClient.available = staticmethod(_orig_available)