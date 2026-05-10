"""
Checkpoint manager for overnight experiment runs.

Tracks status of every (phase, job_id) pair so that interrupted runs
can resume exactly where they left off.

Checkpoint file format (JSON):
{
  "version": "1",
  "run_id": "overnight_20260508_180000",
  "started_at": "2026-05-08T18:00:00Z",
  "last_updated": "2026-05-08T21:30:00Z",
  "jobs": {
    "baselines__seed1__static__vn_hcmc_officesmall": {
      "status": "done",          # pending | running | done | failed
      "started_at": "...",
      "finished_at": "...",
      "output": "results/raw/baselines_20260508/...",
      "error": null
    },
    ...
  }
}
"""
from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CHECKPOINT_VERSION = "1"


class Checkpoint:
    """Persistent checkpoint for a multi-job experiment run."""

    def __init__(self, path: str | Path, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            self._data = json.loads(self.path.read_text())
            print(f"[checkpoint] Resuming from {self.path} "
                  f"({self._count_done()} jobs already done)")
        else:
            self._data = {
                "version":      _CHECKPOINT_VERSION,
                "run_id":       run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S"),
                "started_at":   _now(),
                "last_updated": _now(),
                "jobs":         {},
            }
            self._flush()
            print(f"[checkpoint] New run: {self._data['run_id']} → {self.path}")

    # ── public API ────────────────────────────────────────────────────────────

    def is_done(self, job_id: str) -> bool:
        return self._data["jobs"].get(job_id, {}).get("status") == "done"

    def mark_running(self, job_id: str, meta: dict[str, Any] | None = None) -> None:
        self._data["jobs"][job_id] = {
            "status":     "running",
            "started_at": _now(),
            "finished_at": None,
            "output":     None,
            "error":      None,
            **(meta or {}),
        }
        self._flush()

    def mark_done(self, job_id: str, output: str | None = None,
                  meta: dict[str, Any] | None = None) -> None:
        entry = self._data["jobs"].get(job_id, {})
        entry.update({
            "status":      "done",
            "finished_at": _now(),
            "output":      output,
            "error":       None,
            **(meta or {}),
        })
        self._data["jobs"][job_id] = entry
        self._flush()
        print(f"[checkpoint] ✓ {job_id}")

    def mark_failed(self, job_id: str, error: str) -> None:
        entry = self._data["jobs"].get(job_id, {})
        entry.update({
            "status":      "failed",
            "finished_at": _now(),
            "error":       error,
        })
        self._data["jobs"][job_id] = entry
        self._flush()
        print(f"[checkpoint] ✗ {job_id}: {error}")

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for v in self._data["jobs"].values():
            s = v.get("status", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    @property
    def run_id(self) -> str:
        return self._data["run_id"]

    # ── internals ─────────────────────────────────────────────────────────────

    def _count_done(self) -> int:
        return sum(1 for v in self._data["jobs"].values() if v.get("status") == "done")

    def _flush(self) -> None:
        self._data["last_updated"] = _now()
        # Atomic write: temp file → rename
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Graceful shutdown handler ─────────────────────────────────────────────────

class GracefulExit:
    """
    Catches SIGINT / SIGTERM and sets a flag so the run loop can finish
    the current job before exiting cleanly.

    Usage:
        stopper = GracefulExit()
        for job in jobs:
            if stopper.triggered:
                print("Interrupted — resumable via checkpoint")
                break
            run_job(job)
    """

    def __init__(self) -> None:
        self.triggered = False
        signal.signal(signal.SIGINT,  self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame: Any) -> None:
        print(f"\n[checkpoint] Signal {signum} received — finishing current job then stopping.")
        print("[checkpoint] Re-run the same command to resume.")
        self.triggered = True
