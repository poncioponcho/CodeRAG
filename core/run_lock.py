import json
import os
import time
from dataclasses import dataclass
from typing import Optional


LOCK_DIR = ".coderag_locks"
FRONTEND_LOCK = os.path.join(LOCK_DIR, "frontend.json")
BATCH_LOCK = os.path.join(LOCK_DIR, "batch.json")


@dataclass(frozen=True)
class LockInfo:
    kind: str
    pid: int
    started_at: float
    updated_at: float
    note: str = ""


def _now() -> float:
    return time.time()


def _ensure_lock_dir() -> None:
    os.makedirs(LOCK_DIR, exist_ok=True)


def _read_lock(path: str) -> Optional[LockInfo]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LockInfo(
            kind=data.get("kind", ""),
            pid=int(data.get("pid", 0)),
            started_at=float(data.get("started_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            note=data.get("note", ""),
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_lock(path: str, info: LockInfo) -> None:
    _ensure_lock_dir()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": info.kind,
                "pid": info.pid,
                "started_at": info.started_at,
                "updated_at": info.updated_at,
                "note": info.note,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp, path)


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def touch_frontend_lock(note: str = "", ttl_sec: int = 300) -> bool:
    """
    Streamlit 会频繁 rerun：每次调用都刷新 updated_at。
    若 batch lock 仍“活着”，返回 False 表示前端应进入只读/禁用状态。
    """
    batch = _read_lock(BATCH_LOCK)
    if batch and (is_process_alive(batch.pid) or (_now() - batch.updated_at) < ttl_sec):
        return False

    now = _now()
    info = LockInfo(
        kind="frontend",
        pid=os.getpid(),
        started_at=now,
        updated_at=now,
        note=note,
    )
    _write_lock(FRONTEND_LOCK, info)
    return True


def acquire_batch_lock(note: str = "", ttl_sec: int = 300) -> None:
    """
    批处理脚本（生成/评估）入口调用：
    - 若检测到前端活跃锁，则直接抛错退出（避免并行抢 Ollama）。
    - 否则写入 batch 锁，并在 finally 里 release。
    """
    frontend = _read_lock(FRONTEND_LOCK)
    if frontend and (is_process_alive(frontend.pid) or (_now() - frontend.updated_at) < ttl_sec):
        raise RuntimeError(
            "检测到前端正在运行（frontend lock active）。请先关闭/停止前端，再运行 batch 脚本。"
        )

    now = _now()
    info = LockInfo(
        kind="batch",
        pid=os.getpid(),
        started_at=now,
        updated_at=now,
        note=note,
    )
    _write_lock(BATCH_LOCK, info)


def refresh_batch_lock(note: str = "") -> None:
    cur = _read_lock(BATCH_LOCK)
    if not cur:
        return
    info = LockInfo(
        kind="batch",
        pid=cur.pid or os.getpid(),
        started_at=cur.started_at or _now(),
        updated_at=_now(),
        note=note or cur.note,
    )
    _write_lock(BATCH_LOCK, info)


def release_batch_lock() -> None:
    try:
        os.remove(BATCH_LOCK)
    except FileNotFoundError:
        pass
    except Exception:
        pass

