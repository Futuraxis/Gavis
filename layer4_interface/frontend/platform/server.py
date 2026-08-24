"""Platform frontend server — unified hub for games, benchmark and history.

Serves the built React app from ``platform-frontend/dist/`` and exposes
the platform API under ``/api/*``.  Uses only the stdlib, mirroring the
per-game servers under ``layer4_interface/frontend/``.

Run::

    python -m layer4_interface.frontend.platform.server [--host H] [--port P]
"""

from __future__ import annotations

import argparse
import urllib.parse
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from layer1_translator import translate_rules_json

from ...online_learning import LearningManager, LearningStore, OnlineModelStore
from ..common.http_utils import BodyTooLargeError, read_json_body, send_error_json, send_json
from .benchmark import SOLVER_OPTIONS, BenchmarkRunner
from .games import GAMES, PlayError
from .history import HistoryError, MatchHistory
from .session import PlayManager

ROOT = Path(__file__).resolve().parents[3]
DIST_DIR = ROOT / "platform-frontend" / "dist"
DEFAULT_DATA_DIR = ROOT / "data" / "matches"
PORT = 8770


def make_handler(
    manager: PlayManager,
    history: MatchHistory,
    benchmark: BenchmarkRunner,
    dist_dir: Path = DIST_DIR,
    learning: LearningManager | None = None,
) -> type:
    """Build a handler class bound to the given platform services.

    The handler is produced by a factory (rather than holding module-level
    state like the play apps) so tests can mount their own services.
    """

    class PlatformHandler(SimpleHTTPRequestHandler):
        server_version = "GavisPlatform/0.1"

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, directory=str(dist_dir), **kwargs)

        # ── CORS ──────────────────────────────────────────────────
        # 决策记录（审计 3.6，2026-08-13）：本服务定位为**本机开发工具**
        # （默认绑定 127.0.0.1），因此 CORS 保持通配且不引入认证。对外网
        # /局域网暴露前必须先收紧 CORS 到同源并加鉴权（见
        # docs/design/security-notes.md）。

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        # CORS 只由 end_headers 统一发出一遍（审查 P2：双重 CORS 头移除）
        def end_headers(self) -> None:
            self._send_cors_headers()
            super().end_headers()

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        # ── Routing ───────────────────────────────────────────────

        def do_GET(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/api/games":
                    self._handle_games()
                elif path == "/api/history":
                    self._handle_history_list()
                elif path.startswith("/api/history/"):
                    self._handle_history_get(path[len("/api/history/") :])
                elif path == "/api/benchmark":
                    self._handle_benchmark_list()
                elif path == "/api/benchmark/status":
                    self._handle_benchmark_status()
                elif path == "/api/learning/status":
                    self._handle_learning_status()
                elif path == "/api/learning/apply":
                    self._handle_learning_apply()
                elif path == "/api/learning/config":
                    self._handle_learning_config()
                elif path.startswith("/api/"):
                    send_error_json(self, HTTPStatus.NOT_FOUND, f"未知接口: {path}")
                else:
                    if not dist_dir.is_dir():
                        send_json(
                            self,
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {
                                "ok": False,
                                "error": "前端未构建，请先运行: cd platform-frontend && npm run build",
                            },
                        )
                        return
                    super().do_GET()
            except (PlayError, HistoryError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort envelope for the client
                self.log_error("internal error: %s", exc)
                send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, "服务器内部错误")
                if not dist_dir.is_dir():
                    send_json(
                        self,
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {
                            "ok": False,
                            "error": "前端未构建，请先运行: cd platform-frontend && npm run build",
                        },
                    )
                    return
                super().do_GET()

        def do_POST(self) -> None:
            path = urllib.parse.urlsplit(self.path).path
            try:
                if path == "/api/match/start":
                    self._handle_match_start()
                elif path == "/api/match/move":
                    self._handle_match_move()
                elif path == "/api/match/state":
                    self._handle_match_state()
                elif path == "/api/benchmark/start":
                    self._handle_benchmark_start()
                elif path == "/api/rules/translate":
                    self._handle_rules_translate()
                elif path == "/api/learning/apply":
                    self._handle_learning_apply()
                elif path == "/api/learning/config":
                    self._handle_learning_config()
                else:
                    send_error_json(self, HTTPStatus.NOT_FOUND, f"未知接口: {path}")
            except BodyTooLargeError as exc:
                send_error_json(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            except (PlayError, HistoryError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, str(exc))
            except (KeyError, TypeError, ValueError) as exc:
                send_error_json(self, HTTPStatus.BAD_REQUEST, f"参数错误: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort envelope for the client
                self.log_error("internal error: %s", exc)
                send_error_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, "服务器内部错误")

        # ── API handlers ──────────────────────────────────────────

        def _handle_games(self) -> None:
            games = []
            for spec in GAMES.values():
                games.append(
                    {
                        "game_id": spec.game_id,
                        "display_name": spec.display_name,
                        "description": spec.description,
                        "kind": spec.kind,
                        "board_size": spec.board_size,
                        "seat_options": list(spec.seat_options),
                        "seat_label": spec.seat_label,
                        "player_counts": list(spec.player_counts),
                        "difficulties": list(spec.difficulty_budgets),
                        "solver_options": list(SOLVER_OPTIONS.get(spec.game_id, ())),
                    }
                )
            send_json(self, HTTPStatus.OK, {"ok": True, "games": games})

        def _handle_match_start(self) -> None:
            payload = read_json_body(self)
            session = manager.start(
                payload["game_id"],
                str(payload.get("player_pid", "random")),
                str(payload.get("difficulty", "normal")),
                int(payload.get("player_count", 2)),
            )
            send_json(self, HTTPStatus.OK, {"ok": True, "session": session.snapshot()})

        def _handle_match_move(self) -> None:
            payload = read_json_body(self)
            action = payload.get("action")
            if not isinstance(action, dict):
                raise KeyError("缺少 action")
            snapshot = manager.move(payload["game_id"], action)
            send_json(self, HTTPStatus.OK, {"ok": True, "session": snapshot})

        def _handle_match_state(self) -> None:
            payload = read_json_body(self)
            send_json(self, HTTPStatus.OK, {"ok": True, "session": manager.get(payload["game_id"]).snapshot()})

        def _handle_history_list(self) -> None:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            limit = int(query.get("limit", ["100"])[0])
            game_id = query.get("game_id", [None])[0]
            send_json(self, HTTPStatus.OK, {"ok": True, "matches": history.list_matches(limit=limit, game_id=game_id)})

        def _handle_history_get(self, match_id: str) -> None:
            send_json(self, HTTPStatus.OK, {"ok": True, "match": history.get(urllib.parse.unquote(match_id))})

        def _handle_benchmark_list(self) -> None:
            send_json(self, HTTPStatus.OK, {"ok": True, "jobs": [asdict(j) for j in benchmark.list_jobs()]})

        def _handle_benchmark_status(self) -> None:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            job_id = query.get("job_id", [""])[0]
            job = benchmark.status(job_id)
            if job is None:
                send_error_json(self, HTTPStatus.NOT_FOUND, f"未知评测任务: {job_id}")
                return
            send_json(self, HTTPStatus.OK, {"ok": True, "job": asdict(job)})

        def _handle_benchmark_start(self) -> None:
            payload = read_json_body(self)
            budget = payload.get("budget")
            job = benchmark.start(
                payload["game_id"],
                str(payload["solver_a"]),
                str(payload["solver_b"]),
                int(payload.get("iterations", 10)),
                budget=int(budget) if budget is not None else None,
            )
            send_json(self, HTTPStatus.OK, {"ok": True, "job_id": job.job_id, "job": asdict(job)})

        def _handle_rules_translate(self) -> None:
            payload = read_json_body(self)
            response = translate_rules_json(
                str(payload.get("rule_text", "")),
                source_lang=str(payload.get("source_lang", "zh")),
                game_name=payload.get("game_name"),
                external_frontend=payload.get("external_frontend"),
                run_engine_validation=bool(payload.get("run_engine_validation", True)),
                use_llm=bool(payload.get("use_llm", False)),
                llm_model_path=payload.get("llm_model_path"),
            )
            validation = response.validation
            send_json(
                self,
                HTTPStatus.OK,
                {
                    "ok": True,
                    "rules_json": response.rules_json,
                    "confidence": response.confidence,
                    "validation": asdict(validation) if validation is not None else None,
                },
            )

        # ── Online learning ────────────────────────────────────
        # 在线学习：捕获在 PlayManager 内进行（learning 钩子包裹求解器
        # 句柄并记录每步决策）；本组接口只读状态与触发 apply（建表 +
        # 门禁评估 + 发布），apply 在 HTTP 请求线程内同步执行（默认
        # 局数/预算较小），失败不破坏既有模型（保留上一版本）。

        def _handle_learning_status(self) -> None:
            send_json(self, HTTPStatus.OK, {"ok": True, "learning": learning.status_all()})

        def _handle_learning_apply(self) -> None:
            payload = read_json_body(self)
            game_id = payload.get("game_id")
            if game_id:
                send_json(self, HTTPStatus.OK, {"ok": True, "result": asdict(learning.apply(str(game_id)))})
                return
            results = [asdict(learning.apply(item["game_id"])) for item in learning.status_all()]
            send_json(self, HTTPStatus.OK, {"ok": True, "results": results})

        def _handle_learning_config(self) -> None:
            payload = read_json_body(self)
            game_id = str(payload.get("game_id", ""))
            if game_id not in GAMES:
                raise PlayError(f"未知游戏: {game_id}")
            learning.set_enabled(game_id, bool(payload.get("enabled", True)))
            send_json(self, HTTPStatus.OK, {"ok": True, "learning": learning.status(game_id)})

    return PlatformHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Gavis 平台前端服务 (游戏大厅 / 对战 / 评测 / 历史 / 在线学习)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--learning-min-samples", type=int, default=30, help="apply 的最低人类决策样本数")
    parser.add_argument("--learning-gate-episodes", type=int, default=20, help="门禁短赛局数")
    parser.add_argument("--learning-gate-budget", type=int, default=300, help="门禁搜索预算")
    parser.add_argument(
        "--learning-interval", type=float, default=0.0, help="后台 auto-apply 间隔秒（0 = 仅手动 apply）"
    )
    args = parser.parse_args()

    # 求解器由注册表装配（SolverProvider 注入，L4 不 import L3）
    from train_cli import default_provider

    history = MatchHistory(args.data_dir)
    # 在线学习：捕获数据在 data/online_learning/（已 gitignore），
    # 与用户可见的 data/matches/ 严格分离。
    learning_dir = args.data_dir.parent / "online_learning"
    learning_store = LearningStore(learning_dir)
    model_store = OnlineModelStore(learning_dir / "models")
    default_provider.online_models = model_store  # 新对局读取已发布模型
    learning = LearningManager(
        store=learning_store,
        model_store=model_store,
        provider=default_provider,
        seed=42,
        min_samples=args.learning_min_samples,
        gate_episodes=args.learning_gate_episodes,
        gate_budget=args.learning_gate_budget,
    )
    manager = PlayManager(provider=default_provider, history=history, seed=42, learning=learning)
    benchmark = BenchmarkRunner(provider=default_provider, seed=42)
    if args.learning_interval > 0:
        learning.start_auto(interval_seconds=args.learning_interval)
        print(f"在线学习 auto-apply 已启动: 每 {args.learning_interval}s 检查一次")
    handler = make_handler(manager, history, benchmark, learning=learning)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Gavis 平台服务: http://{args.host}:{args.port}  (API 前缀 /api, 静态目录 {DIST_DIR})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if args.learning_interval > 0:
            learning.stop_auto()
        httpd.server_close()


if __name__ == "__main__":
    main()
