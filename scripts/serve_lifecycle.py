"""
Exercise the whole llama-server lifecycle against the real binary.

Unit tests can fake ``/health`` and ``/v1/models``; this drives the actual
llama-server, because the failure being fixed was a *race in the real thing*:

    a server is running but does not accept the stored API key

Six states, in the order a user meets them:

    1. nothing on the port            -> starts, and the key it saves works
    2. the server is up, key matches  -> reuses it, and does NOT mint a new key
    3. the server is up, key is stale -> refuses, explains, and keeps the old key
    4. force_restart                  -> takes the port back, new key works
    5. two starts at once             -> one owner, and its key is the stored one
    6. the server is killed           -> reported as down, not as a bad key

    python scripts/serve_lifecycle.py [--port 8081]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35 import llama_runtime as rt  # noqa: E402

PORT = 8081
BASE = f"http://127.0.0.1:{PORT}"
MODEL_DIR = r"D:\qwen\models"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def kill_all() -> None:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    for _ in range(24):
        if not rt.port_in_use("127.0.0.1", PORT):
            return
        time.sleep(0.25)


def start(**kwargs) -> rt.ServerStatus:
    return rt.ensure_server(model_dir=MODEL_DIR, context=8192, port=PORT,
                            timeout=240, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    global PORT, BASE
    PORT, BASE = args.port, f"http://127.0.0.1:{args.port}"

    if rt.installed_server() is None:
        print("llama-server is not installed; nothing to test")
        return 2

    print(f"lifecycle test on port {PORT}")
    print("=" * 74)

    # ---- 1. clean start -------------------------------------------------
    print("\n1. clean start")
    kill_all()
    rt.save_server_key("STALE-KEY-FROM-BEFORE", BASE, 1, "old.gguf")
    status = start()
    check("starts when the port is free", status.ok, status.message[:70])
    saved = rt.load_server_key()
    check("the saved key is the one the server accepts",
          rt.probe_server(BASE, saved)[0] == "up",
          f"key {saved[:6]}...")
    first_pid = status.server.proc.pid if status.server and status.server.proc else 0

    # ---- 2. reuse -------------------------------------------------------
    print("\n2. reuse a healthy server")
    before = rt.load_server_key()
    again = start()
    check("reuses rather than starting a second one", again.state == "reused",
          again.state)
    check("does not mint a new key when reusing",
          rt.load_server_key() == before,
          "key unchanged" if rt.load_server_key() == before else "KEY CHANGED")
    still = subprocess.run(["tasklist", "/FI", f"PID eq {first_pid}", "/FO", "CSV",
                            "/NH"], capture_output=True, text=True).stdout
    check("the original process is still the one running",
          str(first_pid) in still, f"pid {first_pid}")

    # ---- 3. a stale key on a live server --------------------------------
    print("\n3. stale key, server still alive")
    rt.save_server_key("DELIBERATELY-WRONG", BASE, 1, "old.gguf")
    refused = start()
    check("refuses instead of adopting it", not refused.ok)
    check("the message names the cause", "not accept" in refused.message,
          refused.message[:60])
    check("the message offers the recovery", "force-restart" in refused.message)
    check("the wrong key is NOT overwritten by a failed start",
          rt.load_server_key() == "DELIBERATELY-WRONG",
          rt.load_server_key()[:12])

    # ---- 4. force restart ------------------------------------------------
    print("\n4. force_restart takes the port back")
    t0 = time.time()
    forced = start(force_restart=True)
    check("succeeds with force_restart", forced.ok, forced.message[:70])
    check("took the port back and the new key works",
          rt.probe_server(BASE, rt.load_server_key())[0] == "up",
          f"{time.time() - t0:.1f}s")
    new_pid = forced.server.proc.pid if forced.server and forced.server.proc else 0
    check("it really is a different process", new_pid != first_pid,
          f"{first_pid} -> {new_pid}")

    # ---- 5. two starts at once ------------------------------------------
    print("\n5. two concurrent starts (the actual root cause)")
    kill_all()
    rt.save_server_key("BEFORE-RACE", BASE, 1, "old.gguf")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(start) for _ in range(2)]
        outcomes = [f.result() for f in futures]
    ok_count = sum(1 for s in outcomes if s.ok)
    check("at least one start succeeded", ok_count >= 1,
          f"{ok_count}/2 said ok")
    check("the stored key is one the live server accepts",
          rt.probe_server(BASE, rt.load_server_key())[0] == "up",
          f"key {rt.load_server_key()[:6]}...")
    listeners = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                               capture_output=True, text=True).stdout
    owners = {line.split()[-1] for line in listeners.splitlines()
              if f":{PORT}" in line and "LISTENING" in line}
    check("exactly one process owns the port", len(owners) == 1,
          f"owners={owners}")

    # ---- 6. killed from outside -----------------------------------------
    print("\n6. the server dies underneath us")
    kill_all()
    state, message = rt.probe_server(BASE, rt.load_server_key())
    check("reported as down, not as a bad key", state == "down",
          f"{state}: {message[:50]}")
    recovered = start()
    check("and it can simply be started again", recovered.ok,
          recovered.message[:60])

    # ---- summary ---------------------------------------------------------
    kill_all()
    passed = sum(1 for _n, ok, _d in _results if ok)
    print("\n" + "=" * 74)
    print(f"RESULT  {passed}/{len(_results)} checks passed")
    for name, ok, detail in _results:
        if not ok:
            print(f"  failed: {name}  {detail}")
    if len(sys.argv) > 2 and sys.argv[2].endswith(".json"):
        Path(sys.argv[2]).write_text(
            json.dumps([{"check": n, "ok": o, "detail": d}
                        for n, o, d in _results], indent=2, ensure_ascii=False),
            encoding="utf-8")
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
