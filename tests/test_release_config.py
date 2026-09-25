"""Guards for the release-please setup: the version marker and the manifest must stay in sync with __version__."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import world_to_beamng

ROOT = Path(__file__).parent.parent


def test_version_line_carries_the_release_please_marker():
    source = (ROOT / "world_to_beamng" / "__init__.py").read_text(encoding="utf-8")

    line = next(l for l in source.splitlines() if l.startswith("__version__"))
    assert "x-release-please-version" in line  # release-please only rewrites marked lines


def test_manifest_and_package_version_agree():
    manifest = json.loads((ROOT / ".release-please-manifest.json").read_text(encoding="utf-8"))

    assert manifest["."] == world_to_beamng.__version__


def test_config_updates_the_version_file_and_uses_conventional_sections():
    config = json.loads((ROOT / "release-please-config.json").read_text(encoding="utf-8"))
    package = config["packages"]["."]

    assert {"type": "generic", "path": "world_to_beamng/__init__.py"} in package["extra-files"]
    assert config["include-component-in-tag"] is False  # tags are plain vX.Y.Z
    assert {s["type"] for s in package["changelog-sections"]} >= {"feat", "fix", "perf"}
