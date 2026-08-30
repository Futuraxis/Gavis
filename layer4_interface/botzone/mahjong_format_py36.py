"""Python 3.6 compatible Mahjong-Format-Test bot.

This file is intentionally standalone for Botzone's old python3 runtime.
Do not import the rest of Gavis here: the project uses newer Python
syntax that Botzone may reject before the bot can print a response.
"""

import json
import socket
import sys
import traceback
import urllib.error
import urllib.request


REMOTE_URL = ""
REMOTE_TOKEN = ""
REMOTE_TIMEOUT = 0.75


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
    remote = call_remote(payload)
    if remote is not None:
        return remote

    texas = decide_texas_fallback(payload)
    if texas is not None:
        return texas

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
        gang = choose_gang_after_draw(state)
        if gang:
            return gang, "mahjong-format-py36: draw %s %s" % (drawn, gang)
        discard = choose_discard(state["hand"])
        remove_one(state["hand"], discard)
        return "PLAY %s" % discard, "mahjong-format-py36: draw %s play %s" % (drawn, discard)

    if current[0] == "3" and len(current) >= 4:
        response = decide_public_request(state, current)
        return response, "mahjong-format-py36: public %s -> %s" % (" ".join(current[:4]), response)

    return "PASS", "mahjong-format-py36: pass"


def decide_texas_fallback(payload):
    current = current_request(payload)
    if not isinstance(current, dict):
        return None
    required = ("num_players", "dealer_id", "my_id", "my_chips", "my_cards", "public_cards", "history")
    if any(k not in current for k in required):
        return None
    try:
        req = parse_texas_request(current)
        betting = texas_betting_state(req)
        legal = texas_legal(req, betting)
        action = texas_heuristic(req, betting, legal)
        if action not in legal:
            action = 0 if 0 in legal else -1
        return action, "texas-fallback-py36: action=%s" % action
    except Exception:
        return -1, "texas-fallback-py36: fold-on-error"


def current_request(payload):
    requests = payload.get("requests")
    value = requests[-1] if isinstance(requests, list) and requests else payload.get("request", payload)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    return value


def parse_texas_request(raw):
    return {
        "num_players": int(raw["num_players"]),
        "dealer_id": int(raw["dealer_id"]),
        "my_id": int(raw["my_id"]),
        "my_chips": int(raw["my_chips"]),
        "my_cards": [int(x) for x in raw["my_cards"]],
        "public_cards": [int(x) for x in raw["public_cards"]],
        "history": list(raw["history"]),
        "hand": int(raw.get("hand", 0)),
        "max_hand": int(raw.get("max_hand", 1)),
        "total_win_chips": [int(x) for x in raw.get("total_win_chips", [])],
    }


def texas_next(player, offset, count):
    return (player + offset) % count


def texas_betting_state(req):
    n = req["num_players"]
    current_round = 0
    for item in req["history"]:
        current_round = max(current_round, int(item.get("round", 0)))
    bets = [0] * n
    if current_round == 0:
        bets[texas_next(req["dealer_id"], 1, n)] = 50
        bets[texas_next(req["dealer_id"], 2, n)] = 100
        round_bet = 100
        min_raise = 200
    else:
        round_bet = 0
        min_raise = 100
    all_in_locked = False
    for item in req["history"]:
        if int(item.get("round", 0)) != current_round:
            continue
        player = int(item["player_id"])
        action = int(item["action"])
        action_type = item.get("action_type")
        if action_type == "fold":
            bets[player] = -1
        elif action_type == "allin":
            bets[player] = -2
            all_in_locked = True
        elif action_type in ("call", "check"):
            if bets[player] >= 0:
                bets[player] = round_bet
        elif action_type == "raise":
            if bets[player] >= 0:
                bets[player] += action
                round_bet = max(round_bet, bets[player])
                min_raise = max(min_raise, 2 * action)
    return {"round": current_round, "bets": bets, "round_bet": round_bet, "min_raise": min_raise, "allin": all_in_locked}


def texas_legal(req, betting):
    mine = betting["bets"][req["my_id"]]
    if mine < 0:
        return set([-1])
    legal = set([-1, -2])
    if betting["allin"]:
        return legal
    to_call = betting["round_bet"] - mine
    if 0 <= to_call < req["my_chips"]:
        legal.add(0)
    if betting["min_raise"] > 0 and betting["min_raise"] < req["my_chips"]:
        legal.add(betting["min_raise"])
    return legal


def texas_heuristic(req, betting, legal):
    ranks = sorted([card // 4 + 2 for card in req["my_cards"]], reverse=True)
    suited = req["my_cards"][0] % 4 == req["my_cards"][1] % 4
    pair = ranks[0] == ranks[1]
    premium = (pair and ranks[0] >= 10) or (ranks[0] >= 13 and ranks[1] >= 10) or (suited and ranks[0] >= 12 and ranks[1] >= 10)
    strong = premium or pair or (ranks[0] >= 11 and ranks[1] >= 9) or (suited and ranks[0] >= 10 and ranks[1] >= 8)
    if betting["allin"]:
        return -2 if premium and -2 in legal else -1
    if betting["round_bet"] == betting["bets"][req["my_id"]]:
        if premium and betting["min_raise"] in legal:
            return betting["min_raise"]
        return 0
    to_call = betting["round_bet"] - betting["bets"][req["my_id"]]
    if not strong and to_call > max(200, req["my_chips"] // 12):
        return -1
    return 0 if 0 in legal else -1


def call_remote(payload):
    if not REMOTE_URL:
        return None
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if REMOTE_TOKEN:
        headers["Authorization"] = "Bearer %s" % REMOTE_TOKEN
    req = urllib.request.Request(REMOTE_URL, data=body, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=REMOTE_TIMEOUT)
        try:
            data = json.loads(resp.read().decode("utf-8"))
        finally:
            resp.close()
    except (urllib.error.URLError, socket.timeout, TimeoutError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "response" not in data:
        return None
    debug = str(data.get("debug", "remote"))
    return data.get("response"), "remote: %s" % debug


def reconstruct(requests, responses, turn_id):
    state = {"player_id": -1, "quan": -1, "hand": [], "peng_tiles": [], "last_discard": None, "last_discard_player": -1}
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
    op = req[2]
    if op == "PLAY" and len(req) >= 4:
        state["last_discard"] = req[3]
        state["last_discard_player"] = actor
    elif op == "PENG":
        state["last_discard"] = req[3] if len(req) >= 4 else None
        state["last_discard_player"] = actor
    elif op == "CHI":
        state["last_discard"] = req[4] if len(req) >= 5 else None
        state["last_discard_player"] = actor
    elif op in ("GANG", "BUGANG", "DRAW", "BUHUA"):
        state["last_discard"] = None
        state["last_discard_player"] = -1


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
        tile = state.get("last_discard")
        if tile is None:
            return
        remove_one(state["hand"], tile)
        remove_one(state["hand"], tile)
        state["peng_tiles"].append(tile)
        remove_one(state["hand"], req[3])
    elif op == "CHI" and len(req) >= 4:
        for tile in chi_hand_tiles(req[3], state.get("last_discard")):
            remove_one(state["hand"], tile)
        if len(req) >= 5:
            remove_one(state["hand"], req[4])
    elif op == "BUGANG" and len(req) >= 4:
        remove_one(state["hand"], req[3])


def choose_gang_after_draw(state):
    counts = count_tiles(state["hand"])
    for tile in sorted(counts):
        if counts[tile] >= 4:
            return "GANG %s" % tile
    for tile in sorted(state.get("peng_tiles", [])):
        if counts.get(tile, 0) >= 1:
            return "BUGANG %s" % tile
    return None


def decide_public_request(state, req):
    actor = safe_int(req[1], -1)
    if actor == state["player_id"]:
        return "PASS"
    op = req[2]
    if op == "BUGANG":
        return "PASS"
    if op != "PLAY" or len(req) < 4:
        return "PASS"

    tile = req[3]
    counts = count_tiles(state["hand"])
    if counts.get(tile, 0) >= 3:
        return "GANG"
    if counts.get(tile, 0) >= 2:
        remaining = list(state["hand"])
        remove_one(remaining, tile)
        remove_one(remaining, tile)
        discard = choose_discard(remaining)
        return "PENG %s" % discard

    if actor == (state["player_id"] - 1) % 4:
        chi = best_chi_response(state["hand"], tile)
        if chi is not None:
            return "CHI %s %s" % (chi[0], chi[1])
    return "PASS"


def best_chi_response(hand, claimed):
    candidates = []
    for middle in chi_middles_for_claim(claimed):
        needed = chi_hand_tiles(middle, claimed)
        if not needed or not has_tiles(hand, needed):
            continue
        remaining = list(hand)
        for tile in needed:
            remove_one(remaining, tile)
        if not remaining:
            continue
        discard = choose_discard(remaining)
        before_counts = count_tiles(hand)
        remaining_counts = count_tiles(remaining)
        before = sum(tile_score(tile, before_counts) for tile in needed)
        after = tile_score(discard, remaining_counts)
        candidates.append((after - before, middle, discard))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def choose_discard(hand):
    if not hand:
        return "W1"
    counts = {}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    return min(hand, key=lambda tile: (tile_score(tile, counts), tile))


def count_tiles(hand):
    counts = {}
    for tile in hand:
        counts[tile] = counts.get(tile, 0) + 1
    return counts


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


def chi_middles_for_claim(claimed):
    if len(claimed) < 2 or claimed[0] not in ("W", "B", "T"):
        return []
    suit = claimed[0]
    rank = safe_int(claimed[1:], 0)
    middles = []
    for middle_rank in (rank - 1, rank, rank + 1):
        if 2 <= middle_rank <= 8 and middle_rank - 1 <= rank <= middle_rank + 1:
            middles.append("%s%d" % (suit, middle_rank))
    return middles


def chi_hand_tiles(middle, claimed):
    if len(middle) < 2 or middle[0] not in ("W", "B", "T"):
        return []
    rank = safe_int(middle[1:], 0)
    sequence = ["%s%d" % (middle[0], rank - 1), middle, "%s%d" % (middle[0], rank + 1)]
    if claimed not in sequence:
        return []
    remaining = list(sequence)
    remaining.remove(claimed)
    return remaining


def has_tiles(hand, tiles):
    counts = count_tiles(hand)
    for tile in tiles:
        if counts.get(tile, 0) <= 0:
            return False
        counts[tile] -= 1
    return True


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
