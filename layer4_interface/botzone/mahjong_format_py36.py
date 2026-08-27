"""Python 3.6 compatible Mahjong-Format-Test bot.

This file is intentionally standalone for Botzone's old python3 runtime.
Do not import the rest of Gavis here: the project uses newer Python
syntax that Botzone may reject before the bot can print a response.
"""

import json
import sys
import traceback


def main():
    try:
        payload = json.loads(sys.stdin.read().strip())
        response, debug = decide(payload)
        out = {"response": response, "debug": debug[:1024], "data": "", "globaldata": ""}
        sys.stdout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    except Exception as exc:
        out = {
            "response": "PASS",
            "debug": ("%s: %s\n%s" % (type(exc).__name__, exc, traceback.format_exc()))[:1024],
            "data": "",
            "globaldata": "",
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")))


def decide(payload):
    requests = [str(x) for x in payload.get("requests", [])]
    responses = [str(x) for x in payload.get("responses", [])]
    turn_id = len(responses)
    if turn_id < 2:
        return "PASS", "mahjong-format-py36: handshake turn=%d" % turn_id

    state = reconstruct(requests, responses, turn_id)
    current = tokens(requests[turn_id]) if turn_id < len(requests) else []
    if not current:
        return "PASS", "mahjong-format-py36: empty request"

    if current[0] == "2" and len(current) >= 2:
        drawn = current[1]
        state["hand"].append(drawn)
        discard = choose_discard(state["hand"])
        remove_one(state["hand"], discard)
        return "PLAY %s" % discard, "mahjong-format-py36: draw %s play %s" % (drawn, discard)

    return "PASS", "mahjong-format-py36: pass"


def reconstruct(requests, responses, turn_id):
    state = {"player_id": -1, "quan": -1, "hand": [], "peng_tiles": []}
    if requests:
        first = tokens(requests[0])
        if len(first) >= 3 and first[0] == "0":
            state["player_id"] = safe_int(first[1], -1)
            state["quan"] = safe_int(first[2], -1)
    if len(requests) >= 2:
        second = tokens(requests[1])
        if len(second) >= 18 and second[0] == "1":
            state["hand"] = list(second[5:18])

    limit = min(turn_id, len(requests))
    for i in range(2, limit):
        req = tokens(requests[i])
        resp = tokens(responses[i]) if i < len(responses) else []
        apply_past_turn(state, req, resp)
    return state


def apply_past_turn(state, req, resp):
    if not req:
        return
    if req[0] == "2" and len(req) >= 2:
        state["hand"].append(req[1])
        apply_own_response(state, resp)
        return
    if req[0] != "3" or len(req) < 3:
        return
    actor = safe_int(req[1], -1)
    if actor == state["player_id"]:
        apply_own_public_action(state, req[2], req)


def apply_own_response(state, resp):
    if not resp:
        return
    op = resp[0]
    if op == "PLAY" and len(resp) >= 2:
        remove_one(state["hand"], resp[1])
    elif op == "GANG" and len(resp) >= 2:
        for _ in range(4):
            remove_one(state["hand"], resp[1])
    elif op == "BUGANG" and len(resp) >= 2:
        remove_one(state["hand"], resp[1])


def apply_own_public_action(state, op, req):
    if op == "PLAY" and len(req) >= 4:
        remove_one(state["hand"], req[3])
    elif op == "PENG" and len(req) >= 4:
        tile = req[3]
        remove_one(state["hand"], tile)
        remove_one(state["hand"], tile)
        state["peng_tiles"].append(tile)
    elif op == "CHI" and len(req) >= 4:
        for tile in chi_side_tiles(req[3]):
            remove_one(state["hand"], tile)
    elif op == "BUGANG" and len(req) >= 4:
        remove_one(state["hand"], req[3])


def choose_discard(hand):
    if not hand:
        return "W1"
    counts = {}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    return min(hand, key=lambda tile: (tile_score(tile, counts), tile))


def tile_score(tile, counts):
    if len(tile) < 2:
        return 0
    suit = tile[0]
    rank = safe_int(tile[1:], 0)
    n = counts.get(tile, 0)
    score = 0
    if n >= 3:
        score += 80
    elif n == 2:
        score += 40
    if suit not in ("W", "B", "T"):
        return score - 30 if n == 1 else score
    if n == 1:
        score -= 10
    for delta in (-1, 1):
        if counts.get("%s%d" % (suit, rank + delta), 0):
            score += 20
    return score


def chi_side_tiles(middle):
    if len(middle) < 2 or middle[0] not in ("W", "B", "T"):
        return []
    rank = safe_int(middle[1:], 0)
    return ["%s%d" % (middle[0], rank - 1), "%s%d" % (middle[0], rank + 1)]


def remove_one(hand, tile):
    try:
        hand.remove(tile)
    except ValueError:
        pass


def tokens(line):
    return line.strip().split()


def safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


if __name__ == "__main__":
    main()
