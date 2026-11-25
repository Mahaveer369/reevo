"""Interactive Web Server & API for the ReEvo Berth Allocation System (BAP/BACAP/DCBAP), JSSP & TSP-MMAS Research Workbench."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT_DIR = Path(__file__).parent.resolve()
STATIC_DIR = ROOT_DIR / "static"
INITIAL_CACHE_PATH = ROOT_DIR / "benchmarks" / "results" / "initial_state_cache.json"
BAP_CACHE_PATH = ROOT_DIR / "benchmarks" / "results" / "bap_initial_state_cache.json"

_CACHE_LOCK = threading.Lock()
_CACHED_INITIAL_PAYLOAD = None
_CACHED_BAP_PAYLOAD = None


def _get_bap_initial_state_payload() -> dict:
    global _CACHED_BAP_PAYLOAD
    with _CACHE_LOCK:
        if _CACHED_BAP_PAYLOAD is not None:
            return _CACHED_BAP_PAYLOAD

        if BAP_CACHE_PATH.exists():
            try:
                _CACHED_BAP_PAYLOAD = json.loads(
                    BAP_CACHE_PATH.read_text(encoding="utf-8")
                )
                return _CACHED_BAP_PAYLOAD
            except Exception:
                pass

        from benchmarks.bap_benchmark import run_bap_benchmarks

        _CACHED_BAP_PAYLOAD = run_bap_benchmarks(save_files=True)
        return _CACHED_BAP_PAYLOAD


def _get_initial_state_payload() -> dict:
    global _CACHED_INITIAL_PAYLOAD
    with _CACHE_LOCK:
        if _CACHED_INITIAL_PAYLOAD is not None:
            return _CACHED_INITIAL_PAYLOAD

        if INITIAL_CACHE_PATH.exists():
            try:
                _CACHED_INITIAL_PAYLOAD = json.loads(
                    INITIAL_CACHE_PATH.read_text(encoding="utf-8")
                )
                return _CACHED_INITIAL_PAYLOAD
            except Exception:
                pass

        from problems.jssp_constructive.benchmark_runner import (
            inspect_single_instance_schedule,
            run_benchmark_suite,
        )
        from problems.jssp_constructive.parser import list_benchmark_instances
        from problems.jssp_constructive.reevo_heuristics import REEVO_QWEN_CODE

        sched = inspect_single_instance_schedule(
            instance_name="ft06",
            solver_id="multistart_reevo_cpls",
            n_starts=5,
            ls_steps=15,
        )
        bench = run_benchmark_suite(
            instance_names=[
                "ft06",
                "ft10",
                "ft20",
                "la01",
                "la02",
                "la03",
                "la04",
                "la05",
                "la16",
                "orb01",
                "abz5",
                "abz6",
            ],
            n_seeds=1,
            n_starts=5,
            ls_steps=15,
        )
        instances_meta = [
            {
                "name": inst.name,
                "family": inst.family,
                "n_jobs": inst.n_jobs,
                "n_machines": inst.n_machines,
                "optimum": inst.optimum,
                "lower_bound": inst.lower_bound,
                "split": inst.split,
            }
            for inst in list_benchmark_instances()
        ]
        _CACHED_INITIAL_PAYLOAD = {
            "instances": instances_meta,
            "default_code": REEVO_QWEN_CODE,
            "benchmark": bench,
            "schedule": sched,
        }
        try:
            INITIAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            INITIAL_CACHE_PATH.write_text(
                json.dumps(_CACHED_INITIAL_PAYLOAD), encoding="utf-8"
            )
        except Exception:
            pass
        return _CACHED_INITIAL_PAYLOAD


class WorkbenchHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class WorkbenchRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_file(self, filepath: Path) -> None:
        if not filepath.exists() or not filepath.is_file():
            self.send_error(404, "File not found")
            return
        ctype, _ = mimetypes.guess_type(str(filepath))
        ctype = ctype or "application/octet-stream"
        data = filepath.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"{ctype}; charset=utf-8"
            if ctype.startswith("text/") or ctype == "application/javascript"
            else ctype,
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json(
                {"status": "ok", "framework": "reevo-bap-jssp-tsp-workbench"}
            )
            return

        if path == "/api/bap/initial-state":
            self._send_json(_get_bap_initial_state_payload())
            return

        if path == "/api/bap/schedule":
            from problems.berth_allocation.instances import get_bap_instance
            from problems.berth_allocation.solvers import (
                BAP_SOLVER_METADATA,
                BAP_SOLVER_REGISTRY,
            )

            inst_name = query.get("instance", ["xu25_bacap_20"])[0]
            solver_name = query.get("solver", ["reevo_vrpagent_bap"])[0]
            try:
                bap_cache = _get_bap_initial_state_payload()
                pre = bap_cache.get("precomputed_schedules", {}).get(inst_name, {})
                inst = get_bap_instance(inst_name)
                if solver_name in pre and "fifp_greedy" in pre:
                    sched_dict = pre[solver_name]
                    base_dict = pre["fifp_greedy"]
                else:
                    if solver_name not in BAP_SOLVER_REGISTRY:
                        raise ValueError(f"Unknown solver '{solver_name}'")
                    sched_dict = BAP_SOLVER_REGISTRY[solver_name](inst).to_dict()
                    base_dict = BAP_SOLVER_REGISTRY["fifp_greedy"](inst).to_dict()

                self._send_json(
                    {
                        "instance": inst.to_dict(),
                        "solver": solver_name,
                        "solver_meta": BAP_SOLVER_METADATA.get(solver_name, {}),
                        "schedule": sched_dict,
                        "baseline_schedule": base_dict,
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/bap/rolling-horizon":
            from problems.berth_allocation.instances import get_bap_instance
            from problems.berth_allocation.solvers import (
                compare_rolling_horizon_strategies,
            )

            inst_name = query.get("instance", ["chen26_table10"])[0]
            try:
                bap_cache = _get_bap_initial_state_payload()
                rh_cached = bap_cache.get("rolling_horizon", {}).get(inst_name)
                if rh_cached is None:
                    inst = get_bap_instance(inst_name)
                    rh_cached = compare_rolling_horizon_strategies(inst)
                self._send_json(rh_cached)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/api/initial-state":
            self._send_json(_get_initial_state_payload())
            return

        if path == "/api/schedule":
            from problems.jssp_constructive.benchmark_runner import (
                inspect_single_instance_schedule,
            )

            inst_name = query.get("instance", ["ft06"])[0]
            solver_id = query.get("solver", ["multistart_reevo_cpls"])[0]
            n_starts = int(query.get("n_starts", ["8"])[0])
            ls_steps = int(query.get("ls_steps", ["30"])[0])
            try:
                res = inspect_single_instance_schedule(
                    instance_name=inst_name,
                    solver_id=solver_id,
                    n_starts=max(1, min(30, n_starts)),
                    ls_steps=max(0, min(100, ls_steps)),
                )
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if path == "/" or path == "/index.html":
            self._serve_file(STATIC_DIR / "index.html")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (STATIC_DIR / rel).resolve()
            if str(target).startswith(str(STATIC_DIR)):
                self._serve_file(target)
                return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            body = json.loads(raw_body)
        except Exception:
            body = {}

        if parsed.path == "/api/bap/evaluate-custom":
            from problems.berth_allocation.eval import evaluate_custom_bap_heuristic

            custom_code = body.get("custom_code", "").strip()
            inst_name = body.get("instance", "xu25_bacap_20")
            if not custom_code:
                self._send_json(
                    {"error": "Custom BAP Python scorer code cannot be empty."},
                    status=400,
                )
                return
            try:
                res = evaluate_custom_bap_heuristic(
                    code_str=custom_code, instance_name=inst_name
                )
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/bap/run-benchmark":
            from benchmarks.bap_benchmark import run_bap_benchmarks

            try:
                res = run_bap_benchmarks(save_files=True)
                global _CACHED_BAP_PAYLOAD
                with _CACHE_LOCK:
                    _CACHED_BAP_PAYLOAD = res
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/run-benchmark":
            from problems.jssp_constructive.benchmark_runner import run_benchmark_suite

            subset = body.get("subset", "all")
            n_seeds = max(1, min(10, int(body.get("n_seeds", 3))))
            n_starts = max(1, min(25, int(body.get("n_starts", 8))))
            ls_steps = max(0, min(80, int(body.get("ls_steps", 30))))
            custom_code = body.get("custom_code")

            if subset == "ft":
                names = ["ft06", "ft10", "ft20"]
            elif subset == "la":
                names = ["la01", "la02", "la03", "la04", "la05", "la16"]
            elif subset == "hard":
                names = ["ft10", "ft20", "la16", "orb01", "abz5", "abz6"]
            else:
                names = [
                    "ft06",
                    "ft10",
                    "ft20",
                    "la01",
                    "la02",
                    "la03",
                    "la04",
                    "la05",
                    "la16",
                    "orb01",
                    "abz5",
                    "abz6",
                ]

            try:
                res = run_benchmark_suite(
                    instance_names=names,
                    n_seeds=n_seeds,
                    n_starts=n_starts,
                    ls_steps=ls_steps,
                    custom_code=custom_code,
                )
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/evaluate-custom":
            from problems.jssp_constructive.benchmark_runner import (
                inspect_single_instance_schedule,
                run_benchmark_suite,
            )

            custom_code = body.get("custom_code", "").strip()
            inst_name = body.get("instance", "la01")
            n_starts = max(1, min(25, int(body.get("n_starts", 8))))
            ls_steps = max(0, min(80, int(body.get("ls_steps", 30))))
            if not custom_code:
                self._send_json(
                    {"error": "Priority heuristic Python code cannot be empty."},
                    status=400,
                )
                return
            try:
                sched = inspect_single_instance_schedule(
                    instance_name=inst_name,
                    solver_id="custom_user_cpls",
                    n_starts=n_starts,
                    ls_steps=ls_steps,
                    custom_code=custom_code,
                )
                suite = run_benchmark_suite(
                    instance_names=["ft06", "ft10", "la01", "la02", "orb01", "abz5"],
                    n_seeds=2,
                    n_starts=n_starts,
                    ls_steps=ls_steps,
                    custom_code=custom_code,
                )
                if suite.get("custom_error"):
                    self._send_json({"error": suite["custom_error"]}, status=400)
                    return
                self._send_json({"schedule": sched, "benchmark": suite})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        self.send_error(404, "Not Found")


def run_server(port: int = 3000) -> None:
    server = WorkbenchHTTPServer(("0.0.0.0", port), WorkbenchRequestHandler)
    print(
        f"[*] ReEvo Berth Allocation, JSSP & TSP-MMAS Research Workbench listening on http://0.0.0.0:{port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    run_server(port=port)
