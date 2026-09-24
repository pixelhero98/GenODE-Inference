"""Current source must not identify a private checkout or repository owner."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF_REPOSITORY = re.compile(r"github[.]com/[^/\s]+/" + "GenODE" + r"[-]Inference", re.IGNORECASE)
PERSONAL_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+[.][A-Za-z]{2,}")
ABSOLUTE_ASSET_PATH = re.compile(r"(?:[A-Za-z]:[/\\](?:Users|home)|/(?:home|scratch|projects)/)[^\s\"']+")
SECRET = re.compile(r"(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")


def test_tracked_public_text_contains_no_private_identifiers():
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    findings = []
    for name in filter(None, names):
        path = ROOT / name
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        for label, pattern in (
            ("repository self-link", SELF_REPOSITORY),
            ("personal email", PERSONAL_EMAIL),
            ("absolute asset path", ABSOLUTE_ASSET_PATH),
            ("credential", SECRET),
        ):
            if pattern.search(content):
                findings.append(f"{name}: {label}")
    assert not findings, "\n".join(findings)
