"""
Publish the repository through GitHub's REST API instead of `git push`.

Why: from this machine `github.com:443` is intermittently unreachable -- ten
consecutive `git push` attempts failed with timeouts and connection resets --
while `api.github.com` answers in under 100 ms, every time.  A push needs the
git endpoint; the Git Data API does not, so the same 42 files can be published
over the endpoint that actually works.

    blobs -> tree -> commit -> ref

The token is read from the file, used in an Authorization header, and never
printed, logged or written anywhere.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FULL_NAME = sys.argv[2] if len(sys.argv) > 2 else ""
COMMIT_MESSAGE = sys.argv[3] if len(sys.argv) > 3 else ""

#: Where the token lives.  Deliberately NOT hard-coded to one machine: this
#: script is published, and a personal path in a public repository is a small
#: leak that costs nothing to avoid.  Resolution order:
#:
#:   1. a path given as the 4th argument
#:   2. $QWEN35_GITHUB_TOKEN_FILE
#:   3. the token itself in $GITHUB_TOKEN / $GH_TOKEN
TOKEN_FILE = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(
    os.environ.get("QWEN35_GITHUB_TOKEN_FILE", "github-token.txt"))
API = "https://api.github.com"


def token() -> str:
    if TOKEN_FILE.is_file():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise SystemExit(
        "no token: pass a token file as the 4th argument, set "
        "QWEN35_GITHUB_TOKEN_FILE, or export GITHUB_TOKEN / GH_TOKEN")


def call(method: str, path: str, tok: str, payload: dict | None = None,
         *, attempts: int = 5) -> dict:
    """One API call, retried: the link is good but not perfect."""
    body = json.dumps(payload).encode() if payload is not None else None
    last = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            API + path, data=body, method=method,
            headers={"Authorization": f"Bearer {tok}",
                     "Accept": "application/vnd.github+json",
                     "User-Agent": "qwen35-harness",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            # 4xx other than rate limiting will not fix itself
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"{method} {path} -> HTTP {exc.code} {detail}")
            last = f"HTTP {exc.code} {detail}"
        except Exception as exc:  # noqa: BLE001 - network flakiness
            last = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"{method} {path} failed after {attempts} attempts: {last}")


def tracked_files() -> list[str]:
    """Exactly what git would track, so .gitignore is respected."""
    out = subprocess.run(["git", "ls-files"], cwd=REPO_DIR, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        raise SystemExit("git ls-files failed: " + out.stderr[:200])
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def bootstrap(full_name: str, tok: str) -> tuple[str, str]:
    """Return ``(parent_commit, base_tree)`` to build the new commit on.

    A brand-new repository has no commits, and GitHub refuses
    ``POST /git/blobs`` with ``409 Git Repository is empty`` -- there is nothing
    for a blob to belong to.  The Contents API *can* write into an empty repo, so
    on the first publish the README goes in first; that creates ``main`` and a
    root commit, and the real tree is built on top of it.

    On every later publish the existing head is used instead, so this is a normal
    fast-forwarding commit rather than a fresh root with a "Add README" step.
    """
    try:
        ref = call("GET", f"/repos/{full_name}/git/ref/heads/main", tok)
        parent = ref["object"]["sha"]
        commit = call("GET", f"/repos/{full_name}/git/commits/{parent}", tok)
        # GET /git/commits/{sha} returns the commit fields at the top level;
        # only the /commits list wraps them in a "commit" object.
        print(f"  on top of  : {parent[:12]} "
              f"({str(commit.get('message', '')).splitlines()[0][:44]})")
        return parent, commit["tree"]["sha"]
    except RuntimeError as exc:
        if "404" not in str(exc):
            raise
    readme = (REPO_DIR / "README.md").read_bytes()
    result = call("PUT", f"/repos/{full_name}/contents/README.md", tok, {
        "message": "Add README",
        "content": base64.b64encode(readme).decode("ascii"),
        "branch": "main",
    })
    commit_sha = result["commit"]["sha"]
    print(f"  bootstrap  : {commit_sha[:12]} (README, root commit)")
    return commit_sha, result["commit"]["tree"]["sha"]


def main() -> int:
    tok = token()
    if not FULL_NAME:
        raise SystemExit("usage: publish.py <repo-dir> <owner/name>")
    files = tracked_files()
    print(f"  repository : {FULL_NAME}")
    print(f"  files      : {len(files)}")

    parent, base_tree = bootstrap(FULL_NAME, tok)

    # 1. one blob per file
    tree: list[dict] = []
    total = 0
    for index, rel in enumerate(files, 1):
        raw = (REPO_DIR / rel).read_bytes()
        total += len(raw)
        blob = call("POST", f"/repos/{FULL_NAME}/git/blobs", tok, {
            "content": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
        })
        mode = "100755" if raw[:2] == b"#!" else "100644"
        tree.append({"path": rel, "mode": mode, "type": "blob",
                     "sha": blob["sha"]})
        if index % 10 == 0 or index == len(files):
            print(f"    blob {index}/{len(files)}")
    print(f"  uploaded   : {total / 1024:.0f} KB of content in {len(tree)} blobs")

    # 2. the tree, based on the bootstrap commit's tree so nothing is orphaned
    created = call("POST", f"/repos/{FULL_NAME}/git/trees", tok,
                   {"tree": tree, "base_tree": base_tree})
    print(f"  tree       : {created['sha'][:12]}")

    # 3. the commit, on top of the existing head
    message = COMMIT_MESSAGE or (
        "qwen35-harness 1.0.0\n\n"
        "A lightweight, sandboxed agent harness built specifically for "
        "Qwen3.5-0.8B (vision, 262144-token context, hybrid Gated-DeltaNet "
        "attention: 18 linear + 6 full-attention layers, so only those 6 need a "
        "KV cache).\n\n"
        "The premise is that a 0.8B model is not a trustworthy principal, so the "
        "reliability lives in the harness rather than in the prompt: GBNF "
        "grammar synthesis with nullability and termination validation, "
        "validate-and-revert on every write, path confinement and an argv-based "
        "command allowlist enforced below the model, observation masking with a "
        "harness-maintained ledger, and token-aware conversion of attached "
        "documents (PDF, Word, Excel, PowerPoint, CSV, JSON, HTML, images, "
        "scanned pages) into page-marked Markdown inside the workspace.\n\n"
        "Contains qwen35/ (stdlib only), unit tests, evaluation suites, a "
        "capability probe and a GUI smoke test.\n\n"
        "License: MIT.  The optional PDF path prefers PyMuPDF (AGPL-3.0 / "
        "commercial dual-licensed); it is an optional accelerator with a "
        "standard-library fallback.  See README.\n")
    commit = call("POST", f"/repos/{FULL_NAME}/git/commits", tok, {
        "message": message, "tree": created["sha"], "parents": [parent],
    })
    print(f"  commit     : {commit['sha'][:12]}")

    # 4. point main at it
    ref = call("PATCH", f"/repos/{FULL_NAME}/git/refs/heads/main", tok, {
        "sha": commit["sha"], "force": True,
    })
    print(f"  ref        : {ref['ref']} -> {ref['object']['sha'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
