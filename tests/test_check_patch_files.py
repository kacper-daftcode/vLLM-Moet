import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_patch_files",
    ROOT / "tools" / "check_patch_files.py",
)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


class PatchGuardTest(unittest.TestCase):
    def test_v025_release_is_strictly_source_bound(self):
        release = guard.RELEASES["0.25.0"]
        self.assertEqual(release["source"], "SOURCE-v025.txt")
        self.assertEqual(release["branch"], "moet-v0.25.0")
        self.assertEqual(release["base_tag"], "v0.25.0")
        self.assertEqual(
            release["base_sha"],
            "702f4814fe54fabff350d43cb753ae3e47c0c276",
        )
        self.assertTrue(release["require_source"])
        self.assertTrue(release["require_pushed_source"])

    def test_normalized_diff_ignores_only_representation_details(self):
        left = (
            b"diff --git a/x b/x\n"
            b"index abc123..def456 100644\n"
            b"@@ -1 +1 @@\n"
            b" \n"
        )
        right = (
            b"diff --git a/x b/x\n"
            b"index 111111111..222222222 100644\n"
            b"@@ -1 +1 @@\n"
            b"\n"
        )
        self.assertEqual(guard.normalized(left), guard.normalized(right))

        changed = right.replace(b"@@ -1 +1 @@", b"@@ -1 +2 @@")
        self.assertNotEqual(guard.normalized(left), guard.normalized(changed))

    def test_canonical_patch_strips_blank_context_markers(self):
        release = {"normalize_blank_context": True}
        raw = b"@@ -1 +1 @@\n \n+value\n"
        self.assertEqual(
            guard.canonical_patch(raw, release),
            b"@@ -1 +1 @@\n\n+value\n",
        )

    def test_strict_release_fails_when_source_clone_is_missing(self):
        release = dict(guard.RELEASES["0.25.0"])
        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "SOURCE-v025.txt"
            source.write_text("0" * 40 + "\n")
            self.assertEqual(
                guard.check_source(release, "unused.patch", str(source), None),
                1,
            )

    def test_strict_release_accepts_only_published_branch_source(self):
        with tempfile.TemporaryDirectory() as tempdir:
            temp = Path(tempdir)
            repo = temp / "vllm"
            repo.mkdir()
            run_git(repo, "init", "-q")
            run_git(repo, "config", "user.name", "test")
            run_git(repo, "config", "user.email", "test@example.com")

            tracked = repo / "tracked.txt"
            tracked.write_text("base\n")
            run_git(repo, "add", "tracked.txt")
            run_git(repo, "commit", "-q", "-m", "base")
            base_sha = run_git(repo, "rev-parse", "HEAD")
            run_git(repo, "tag", "v0.25.0")
            run_git(repo, "switch", "-q", "-c", "moet-v0.25.0")

            tracked.write_text("source\n")
            run_git(repo, "commit", "-qam", "source")
            source_sha = run_git(repo, "rev-parse", "HEAD")

            raw_patch = subprocess.run(
                ["git", "-C", str(repo), "diff", "v0.25.0", source_sha],
                check=True,
                capture_output=True,
            ).stdout
            patch = temp / "overlay.patch"
            patch.write_bytes(
                guard.canonical_patch(
                    raw_patch,
                    {"normalize_blank_context": True},
                )
            )
            source = temp / "SOURCE-v025.txt"
            source.write_text(source_sha + "\n")

            release = {
                "branch": "moet-v0.25.0",
                "base_tag": "v0.25.0",
                "base_sha": base_sha,
                "require_source": True,
                "require_pushed_source": True,
            }
            self.assertEqual(
                guard.check_source(
                    release,
                    str(patch),
                    str(source),
                    str(repo),
                ),
                1,
            )

            run_git(
                repo,
                "update-ref",
                "refs/remotes/fork/moet-v0.25.0",
                source_sha,
            )
            self.assertEqual(
                guard.check_source(
                    release,
                    str(patch),
                    str(source),
                    str(repo),
                ),
                0,
            )

    def test_strict_update_reports_missing_bound_base_tag(self):
        with tempfile.TemporaryDirectory() as tempdir:
            repo = Path(tempdir) / "vllm"
            repo.mkdir()
            run_git(repo, "init", "-q")
            run_git(repo, "config", "user.name", "test")
            run_git(repo, "config", "user.email", "test@example.com")

            tracked = repo / "tracked.txt"
            tracked.write_text("source\n")
            run_git(repo, "add", "tracked.txt")
            run_git(repo, "commit", "-q", "-m", "source")
            run_git(repo, "branch", "moet-v0.25.0")
            source_sha = run_git(repo, "rev-parse", "moet-v0.25.0")
            run_git(
                repo,
                "update-ref",
                "refs/remotes/fork/moet-v0.25.0",
                source_sha,
            )

            release = {
                "patch": "unused.patch",
                "manifest": "unused.txt",
                "source": "unused-source.txt",
                "branch": "moet-v0.25.0",
                "base_tag": "v0.25.0",
                "base_sha": "0" * 40,
                "fork_candidates": [],
                "require_pushed_source": True,
            }
            with self.assertRaisesRegex(
                SystemExit,
                "required base tag v0.25.0 is missing",
            ):
                guard.update_sourced_release("0.25.0", release, str(repo))


if __name__ == "__main__":
    unittest.main()
