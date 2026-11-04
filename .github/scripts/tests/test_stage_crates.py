# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "stage_crates", Path(__file__).resolve().parents[1] / "stage_crates.py"
)
stage_crates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage_crates)


class RewriteVersionsTests(unittest.TestCase):
    def rewrite(self, workspace, package):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "member" / "Cargo.toml"
            manifest.parent.mkdir()
            (root / "Cargo.toml").write_text(workspace)
            manifest.write_text(package)
            stage_crates.rewrite_versions(
                root, {"member": {"manifest_path": str(manifest)}},
                "0.8.0", "0.8.0-nightly.20261013.gabcdef0",
            )
            return (root / "Cargo.toml").read_text(), manifest.read_text()

    def test_inherited_version_does_not_rewrite_dependency_table(self):
        workspace = '[workspace.package]\nversion = "0.8.0"\n'
        package = (
            '[package]\nname = "member"\nversion.workspace = true\n'
            '[dependencies.third-party]\nversion = "0.8.0"\n'
        )
        stamped, member = self.rewrite(workspace, package)
        self.assertIn('version = "0.8.0-nightly.20261013.gabcdef0"', stamped)
        self.assertEqual(member, package)

    def test_workspace_section_does_not_rewrite_dependency_table(self):
        workspace = (
            '[workspace.package]\nedition = "2024"\n'
            '[workspace.dependencies.third-party]\nversion = "0.8.0"\n'
        )
        stamped, _ = self.rewrite(workspace, '[package]\nversion.workspace = true\n')
        self.assertEqual(stamped, workspace)

    def test_literal_versions_and_internal_pins_are_stamped(self):
        workspace = (
            '[workspace.package]\nversion = "0.8.0"\n[workspace.dependencies]\n'
            'member = { path = "member", version = "0.8.0" }\n'
            'third-party = { version = "0.8.0" }\n'
        )
        package = '[package]\nauthors = ["Example"]\nversion = "0.8.0"\n'
        stamped, member = self.rewrite(workspace, package)
        self.assertIn('version = "=0.8.0-nightly.20261013.gabcdef0"', stamped)
        self.assertIn('third-party = { version = "0.8.0" }', stamped)
        self.assertIn('version = "0.8.0-nightly.20261013.gabcdef0"', member)

    def test_stamp_only_reads_version_after_workspace_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "Cargo.toml").write_text(
                '[workspace.package]\nauthors = ["Example"]\nversion = "0.8.0"\n'
            )
            member = root / "member.toml"
            member.write_text('[package]\nversion.workspace = true\n')
            metadata = {
                "workspace_members": ["member"],
                "packages": [{
                    "id": "member", "name": "member", "version": "0.8.0",
                    "manifest_path": str(member), "dependencies": [],
                }],
            }
            arguments = [
                "stage_crates.py", "--root", str(root), "--output-dir", str(root / "dist"),
                "--stage-version", "0.8.0-nightly.20261013.gabcdef0", "--stamp-only",
            ]
            with patch.object(stage_crates, "cargo_metadata", return_value=metadata), \
                    patch.object(stage_crates.subprocess, "run") as cargo, \
                    patch("sys.argv", arguments):
                cargo.return_value.returncode = 0
                self.assertEqual(stage_crates.main(), 0)
                cargo.assert_called_once_with(["cargo", "update", "--workspace"], cwd=root)
            self.assertIn('version = "0.8.0-nightly.20261013.gabcdef0"', (root / "Cargo.toml").read_text())


if __name__ == "__main__":
    unittest.main()
