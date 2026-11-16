"""D12: dependency-free web console server.

Only Python's standard library is used.  The server is read-mostly:

* GET endpoints expose the schema and the artifact catalog;
* POST /api/experiment writes YAML under ``configs/experiments/generated/`` and
  returns the shared plan (identities + feature-cache state + next command);
* POST /api/plan returns the same plan without writing anything;
* POST /api/features builds the feature cache through the existing
  ``sddbench extract-features`` CLI (a job, not an inline request);
* POST /api/run launches the existing ``sddbench run`` CLI in a subprocess and
  refuses (409) exactly when the plan is not ready;
* POST /api/split runs ``sddbench split`` synchronously (it is fast).

It binds to 127.0.0.1 by default and has no authentication: do not expose it
to a public network.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..config import ConfigError, load_paths
from ..features.registry import feature_dim
from ..modeling.plan import describe_plan
from .catalog import build_catalog, find_dataset
from .schema import build_configs, get_schema, write_configs

STATIC_DIR = Path(__file__).resolve().parent / "static"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.+-]+$")
MEDIA_NAME = re.compile(r"^[A-Za-z0-9_.+-]+\.png$")
MAX_BODY = 2 * 1024 * 1024


class PlanBlocked(RuntimeError):
    """Raised when a training request reaches the server before the plan is ready."""

    def __init__(self, plan: dict):
        self.plan = plan
        super().__init__("; ".join(plan.get("errors") or ["run is not ready"]))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _stamp() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


class JobManager:
    """Track subprocess-backed training jobs and their logs."""

    def __init__(self, repo_root: Path, workspace: Path, paths_file: Path):
        self.repo_root = repo_root
        self.workspace = workspace
        self.paths_file = paths_file
        self.jobs_dir = workspace / "web" / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start(self, config_path: Path, dry_run: bool) -> dict:
        cmd = [sys.executable, "-u", "-m", "sddbench.cli", "run",
               "--config", str(config_path),
               "--paths", str(self.paths_file),
               "--repo-root", str(self.repo_root)]
        if dry_run:
            cmd.append("--dry-run")
        return self.start_command(cmd, "run", {
            "dry_run": dry_run,
            "config_path": str(config_path.relative_to(self.repo_root)),
        })

    def start_features(self, command: list[str]) -> dict:
        """Build the feature cache through the same CLI the user would run."""
        return self.start_command(list(command), "features", {})

    def start_command(self, cmd: list[str], kind: str, extra: dict) -> dict:
        job_id = f"{_stamp()}_{uuid.uuid4().hex[:6]}"
        log_path = self.jobs_dir / f"{job_id}.log"
        job = {
            "job_id": job_id,
            "kind": kind,
            "status": "running",
            "returncode": None,
            "run_id": None,
            "started_at": _now(),
            "finished_at": None,
            "log_path": str(log_path),
            **extra,
        }
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"$ {' '.join(cmd)}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd, cwd=self.repo_root, stdout=log_file, stderr=subprocess.STDOUT,
            text=True)
        with self._lock:
            self.jobs[job_id] = job
        self._write(job)
        threading.Thread(target=self._watch, args=(job, proc, log_file),
                         daemon=True).start()
        return job

    def _watch(self, job: dict, proc: subprocess.Popen, log_file) -> None:
        code = proc.wait()
        log_file.close()
        text = ""
        try:
            text = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        match = re.findall(r"\[run\] completed:\s*(\S+)", text)
        job["run_id"] = match[-1] if match else None
        job["returncode"] = code
        job["status"] = "completed" if code == 0 else "failed"
        job["finished_at"] = _now()
        self._write(job)

    def _write(self, job: dict) -> None:
        try:
            (self.jobs_dir / f"{job['job_id']}.json").write_text(
                json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self.jobs.get(job_id)
        if job:
            return job
        path = self.jobs_dir / f"{job_id}.json"
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return out[:50]

    def log_tail(self, job: dict, lines: int = 200) -> str:
        try:
            text = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])


class AppState:
    def __init__(self, repo_root: str | Path, paths_file: str | Path):
        self.repo_root = Path(repo_root).resolve()
        self.paths_file = Path(paths_file).expanduser().resolve()
        self.paths = load_paths(self.paths_file)
        self.workspace = Path(self.paths["workspace_root"])
        self.jobs = JobManager(self.repo_root, self.workspace, self.paths_file)
        self._catalog = None
        self._catalog_at = 0.0
        self._lock = threading.Lock()

    def catalog(self, force: bool = False) -> dict:
        with self._lock:
            if force or self._catalog is None or time.time() - self._catalog_at > 3.0:
                self._catalog = build_catalog(self.repo_root, self.workspace)
                self._catalog_at = time.time()
            return self._catalog

    # ------------------------------------------------------------ actions
    def _resolve_experiment(self, body: dict) -> dict:
        """Shared validation/config resolution for plan, save and build-features."""
        dataset_id = str(body.get("dataset") or "")
        catalog = self.catalog(force=True)
        dataset = find_dataset(catalog, dataset_id)
        if not dataset:
            raise ValueError(f"unknown dataset {dataset_id!r}")
        prepared_id = str(body.get("prepared_id") or "")
        split_id = str(body.get("split_id") or "")
        if not SAFE_ID.match(prepared_id):
            raise ValueError("prepared_id is required")
        if not SAFE_ID.match(split_id):
            raise ValueError("split_id is required")
        prepared = next((p for p in dataset["prepared"] if p["job_id"] == prepared_id), None)
        if prepared is None:
            raise ValueError(f"prepared job not found: {prepared_id}")
        if prepared["status"] != "approved" or not prepared["has_manifest"]:
            raise ValueError(f"prepared job {prepared_id} is not approved")
        split = next((s for s in dataset["splits"] if s["split_id"] == split_id), None)
        if split is None:
            raise ValueError(f"split not found: {split_id}")
        if split.get("prepared_id") not in (None, "", prepared_id):
            raise ValueError("split was built from a different prepared job")

        experiment_id = str(body.get("experiment_id") or "").strip()
        if not experiment_id:
            experiment_id = f"web_{dataset_id}_{_stamp()}"
        if not SAFE_ID.match(experiment_id):
            raise ValueError("experiment_id may only contain letters, digits, _ . + -")
        values = {k: v for k, v in (body.get("values") or {}).items()
                  if not str(k).startswith("__")}
        experiment, preprocess, features = build_configs(
            self.repo_root, dataset_id, dataset["raw_config"], experiment_id,
            prepared_id, split_id, values)
        if dataset["grouping"] == "file" and experiment["evaluation"]["unit"] != "recording":
            raise ValueError("grouping=file 时评价单位必须是 recording")
        return {
            "dataset_id": dataset_id,
            "dataset": dataset,
            "prepared": prepared,
            "split": split,
            "experiment_id": experiment_id,
            "experiment": experiment,
            "preprocess": preprocess,
            "features": features,
        }

    def _plan(self, experiment: dict) -> dict:
        return describe_plan(experiment, self.workspace, self.repo_root,
                             paths_file=self.paths_file)

    def _summary(self, resolved: dict) -> dict:
        prepared, split = resolved["prepared"], resolved["split"]
        return {
            "experiment_id": resolved["experiment_id"],
            "config": resolved["experiment"],
            "feature_dim": feature_dim(resolved["features"]),
            "primary_unit": resolved["experiment"]["evaluation"]["unit"],
            "prepared": {
                "n_samples": (prepared.get("counts") or {}).get("n_manifest"),
                "n_groups": (prepared.get("counts") or {}).get("n_groups"),
                "n_subjects": (prepared.get("counts") or {}).get("n_subjects"),
            },
            "split": {
                "n_samples": (split.get("summary") or {}).get("n_samples"),
                "by_split": (split.get("summary") or {}).get("by_split"),
            },
        }

    def save_experiment(self, body: dict) -> dict:
        resolved = self._resolve_experiment(body)
        paths = write_configs(self.repo_root, resolved["experiment"],
                              resolved["preprocess"], resolved["features"])
        plan = self._plan(resolved["experiment"])
        return {**self._summary(resolved), "paths": paths, "plan": plan}

    def plan_experiment(self, body: dict) -> dict:
        """Read-only: same configs/identities as save, nothing written."""
        resolved = self._resolve_experiment(body)
        return {**self._summary(resolved), "plan": self._plan(resolved["experiment"])}

    def build_features(self, body: dict) -> dict:
        """Save the configs, then run the exact extract-features action in a job."""
        saved = self.save_experiment(body)
        plan = saved["plan"]
        if plan["ready"]:
            return {"skipped": True, "reason": "feature cache already ready",
                    "experiment_id": saved["experiment_id"], "plan": plan}
        action = next((a for a in plan["actions"]
                       if a.get("action") == "build_features"), None)
        if not action:
            raise ValueError("cannot build features: " + "; ".join(plan["errors"]))
        job = self.jobs.start_features(action["command"])
        return {"job": job, "experiment_id": saved["experiment_id"],
                "paths": saved["paths"], "plan": plan}

    def create_split(self, body: dict) -> dict:
        dataset_id = str(body.get("dataset") or "")
        prepared_id = str(body.get("prepared_id") or "")
        if not SAFE_ID.match(prepared_id):
            raise ValueError("prepared_id is required")
        seed = int(body.get("seed") or 0)
        train = float(body.get("train_frac", 0.6))
        val = float(body.get("val_frac", 0.2))
        test = float(body.get("test_frac", 0.2))
        if min(train, val, test) <= 0 or abs(train + val + test - 1.0) > 1e-6:
            raise ValueError("train/val/test 比例必须为正且合计为 1.0")
        cmd = [sys.executable, "-m", "sddbench.cli", "split",
               "--dataset", dataset_id, "--prepared", prepared_id,
               "--seed", str(seed), "--train-frac", str(train),
               "--val-frac", str(val), "--test-frac", str(test),
               "--paths", str(self.paths_file), "--repo-root", str(self.repo_root)]
        proc = subprocess.run(cmd, cwd=self.repo_root, text=True, capture_output=True,
                              timeout=300)
        output = (proc.stdout or "") + (proc.stderr or "")
        match = re.search(r"\[split\] id=(\S+)", output)
        if proc.returncode != 0 or not match:
            raise ValueError(f"split failed (exit {proc.returncode}):\n{output[-2000:]}")
        return {"split_id": match.group(1), "output": output}


def _media_path(app: AppState, run_id: str, name: str) -> Path | None:
    if not SAFE_ID.match(run_id) or not MEDIA_NAME.match(name):
        return None
    candidate = app.workspace / "runs" / run_id / "figures" / name
    if candidate.is_file():
        return candidate
    for archived in (app.repo_root / "results_summary").glob(f"*/{run_id}/figures/{name}"):
        if archived.is_file():
            return archived
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "sddbench-web/0.1"

    def log_message(self, fmt, *args):  # keep the console readable
        pass

    @property
    def app(self) -> AppState:
        return self.server.app  # type: ignore[attr-defined]

    # ------------------------------------------------------------ helpers
    def _json(self, payload, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    # ------------------------------------------------------------ routes
    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/"):]
                target = (STATIC_DIR / rel).resolve()
                if STATIC_DIR not in target.parents or not target.is_file():
                    self._json({"error": "not found"}, 404)
                    return
                ctype = {
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".svg": "image/svg+xml",
                    ".html": "text/html; charset=utf-8",
                }.get(target.suffix, "application/octet-stream")
                self._file(target, ctype)
                return
            if path == "/api/health":
                self._json({"ok": True, "time": _now()})
                return
            if path == "/api/schema":
                self._json(get_schema())
                return
            if path == "/api/catalog":
                self._json(self.app.catalog())
                return
            if path == "/api/jobs":
                self._json({"jobs": self.app.jobs.list()})
                return
            if path.startswith("/api/jobs/"):
                job = self.app.jobs.get(path.rsplit("/", 1)[-1])
                if not job:
                    self._json({"error": "job not found"}, 404)
                    return
                self._json({**job, "log": self.app.jobs.log_tail(job)})
                return
            if path.startswith("/media/run/"):
                parts = path[len("/media/run/"):].split("/")
                if len(parts) != 2:
                    self._json({"error": "bad media path"}, 400)
                    return
                target = _media_path(self.app, parts[0], parts[1])
                if not target:
                    self._json({"error": "not found"}, 404)
                    return
                self._file(target, "image/png")
                return
            self._json({"error": "not found"}, 404)
        except Exception as exc:  # pragma: no cover - last-resort guard
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            body = self._body()
            if path == "/api/experiment":
                self._json(self.app.save_experiment(body), 201)
                return
            if path == "/api/plan":
                self._json(self.app.plan_experiment(body))
                return
            if path == "/api/features":
                self._json(self.app.build_features(body), 201)
                return
            if path == "/api/run":
                saved = self.app.save_experiment(body)
                dry_run = bool(body.get("dry_run"))
                if not dry_run and not saved["plan"]["ready"]:
                    raise PlanBlocked(saved["plan"])
                config_path = (self.app.repo_root / saved["paths"]["experiment"]).resolve()
                job = self.app.jobs.start(config_path, dry_run)
                self._json({"job": job, "saved": saved}, 201)
                return
            if path == "/api/split":
                self._json(self.app.create_split(body), 201)
                return
            self._json({"error": "not found"}, 404)
        except PlanBlocked as exc:
            self._json({"error": str(exc), "plan": exc.plan}, 409)
        except (ValueError, ConfigError) as exc:
            self._json({"error": str(exc)}, 400)
        except subprocess.TimeoutExpired:
            self._json({"error": "command timed out"}, 504)
        except Exception as exc:  # pragma: no cover
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def make_server(repo_root: str | Path, paths_file: str | Path,
                host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    app = AppState(repo_root, paths_file)
    server = ThreadingHTTPServer((host, port), Handler)
    server.app = app  # type: ignore[attr-defined]
    return server


def serve(repo_root: str | Path, paths_file: str | Path,
          host: str = "127.0.0.1", port: int = 8765) -> None:
    server = make_server(repo_root, paths_file, host, port)
    print(f"[serve] sddbench web console: http://{host}:{server.server_address[1]}")
    print("[serve] read-only catalog + config generation + training jobs; "
          "no authentication, bind to 127.0.0.1 only.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopping")
    finally:
        server.server_close()
