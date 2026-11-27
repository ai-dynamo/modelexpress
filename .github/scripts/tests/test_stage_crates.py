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


class RegistryModeTests(unittest.TestCase):
    INDEX = "sparse+https://host.example/artifactory/api/cargo/mx-cargo/index/"

    def test_index_prefers_explicit_value_then_derives_from_url_and_repo(self):
        env = {"ARTIFACTORY_CARGO_INDEX": self.INDEX, "ARTIFACTORY_URL": "https://other", "ARTIFACTORY_CARGO_REPO_NAME": "x"}
        with patch.dict(stage_crates.os.environ, env, clear=True):
            self.assertEqual(stage_crates.registry_index(), self.INDEX)
        for url in ("https://host.example", "https://host.example/", "https://host.example/artifactory"):
            env = {"ARTIFACTORY_URL": url, "ARTIFACTORY_CARGO_REPO_NAME": "mx-cargo"}
            with patch.dict(stage_crates.os.environ, env, clear=True):
                self.assertEqual(stage_crates.registry_index(), self.INDEX, url)
        with patch.dict(stage_crates.os.environ, {"ARTIFACTORY_URL": "https://host.example"}, clear=True):
            self.assertEqual(stage_crates.registry_index(), "")

    def test_inject_registry_adds_registry_to_internal_path_deps_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "Cargo.toml"
            manifest.write_text(
                'modelexpress-common = { path = "c", version = "=0.8.0-rc.1", default-features = false }\n'
                'modelexpress-client = { workspace = true }\n'
                'serde = { version = "1.0" }\n'
            )
            stage_crates.inject_registry([manifest], ["modelexpress-common", "modelexpress-client"], "artifactory", {})
            text = manifest.read_text()
            self.assertIn('default-features = false, registry = "artifactory" }', text)
            self.assertIn("modelexpress-client = { workspace = true }", text)
            self.assertIn('serde = { version = "1.0" }', text)

    def test_cargo_config_preserves_settings_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".cargo").mkdir()
            (root / ".cargo" / "config.toml").write_text("[build]\nrustflags = [\"-C\", \"x\"]\n")
            for _ in range(2):
                stage_crates.write_cargo_config(root, "artifactory", self.INDEX)
            text = (root / ".cargo" / "config.toml").read_text()
            self.assertIn("rustflags", text)
            self.assertEqual(text.count("[registries]"), 1)
            (root / ".cargo" / "config.toml").write_text('[registries.other]\nindex = "x"\n')
            with self.assertRaises(RuntimeError):
                stage_crates.write_cargo_config(root, "artifactory", self.INDEX)

    def test_crates_io_alias_is_refused(self):
        arguments = ["stage_crates.py", "--registry", "crates-io"]
        with patch("sys.argv", arguments):
            self.assertEqual(stage_crates.main(), 1)

    def test_registry_mode_requires_credentials(self):
        with patch.dict(stage_crates.os.environ, {}, clear=True), \
                patch("sys.argv", ["stage_crates.py", "--registry", "artifactory"]):
            self.assertEqual(stage_crates.main(), 1)

    def test_publish_is_fail_soft_and_skips_dependents_of_a_failure(self):
        pkgs = {
            "a": {"version": "1", "manifest_path": "/m/a/Cargo.toml", "dependencies": []},
            "b": {"version": "1", "manifest_path": "/m/b/Cargo.toml", "dependencies": [{"name": "a"}]},
            "c": {"version": "1", "manifest_path": "/m/c/Cargo.toml", "dependencies": []},
        }
        def publish(manifest, alias, env):
            if "/a/" in manifest:
                raise RuntimeError("boom")
            return "published"
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(stage_crates, "write_cargo_config"), \
                patch.object(stage_crates, "inject_registry"), \
                patch.object(stage_crates, "crate_exists", return_value=False), \
                patch.object(stage_crates, "publish", side_effect=publish), \
                patch.object(stage_crates.subprocess, "run") as run, \
                patch("builtins.print") as out:
            run.return_value.returncode = 0
            rc = stage_crates.publish_to_registry(Path(tmp), pkgs, ["a", "b", "c"], "artifactory", self.INDEX, "t")
        printed = "\n".join(str(c.args[0]) for c in out.call_args_list if c.args)
        self.assertEqual(rc, 1)
        self.assertIn("STAGED_CRATE=c", printed)
        self.assertIn("FAILED_CRATE=a boom", printed)
        self.assertIn("FAILED_CRATE=b dependency a failed", printed)
        self.assertNotIn("STAGED_CRATE=b", printed)

    def test_existing_crate_is_staged_without_publishing(self):
        pkgs = {"a": {"version": "1", "manifest_path": "/m/a/Cargo.toml", "dependencies": []}}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(stage_crates, "write_cargo_config"), \
                patch.object(stage_crates, "inject_registry"), \
                patch.object(stage_crates, "crate_exists", return_value=True), \
                patch.object(stage_crates, "publish") as publish, \
                patch.object(stage_crates.subprocess, "run") as run, \
                patch("builtins.print") as out:
            run.return_value.returncode = 0
            rc = stage_crates.publish_to_registry(Path(tmp), pkgs, ["a"], "artifactory", self.INDEX, "t")
        self.assertEqual(rc, 0)
        publish.assert_not_called()
        self.assertIn("STAGED_CRATE=a", "\n".join(str(c.args[0]) for c in out.call_args_list if c.args))


if __name__ == "__main__":
    unittest.main()
