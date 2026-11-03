"""tools/check_patch_files.py against a fork clone whose worktree holds a path
named like the base tag (issue #14: `git diff v0.24.0 <sha>` is ambiguous
without a `--` terminator)."""
import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_patch_files", ROOT / "tools" / "check_patch_files.py")
cpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpf)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True).stdout.strip()


class TagNamedPathInWorktree(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.fork = tmp / "fork"
        self.fork.mkdir()
        _git(self.fork, "init", "-q", "-b", "main")
        (self.fork / "a.py").write_text("print(1)\n")
        _git(self.fork, "add", "a.py")
        _git(self.fork, "commit", "-q", "-m", "base")
        _git(self.fork, "tag", cpf.BASE_TAG)
        _git(self.fork, "checkout", "-q", "-b", cpf.BRANCH)
        (self.fork / "a.py").write_text("print(2)\n")
        _git(self.fork, "commit", "-q", "-am", "moet change")
        self.sha = _git(self.fork, "rev-parse", "HEAD")
        # the ambiguity: an untracked directory named like the tag
        (self.fork / cpf.BASE_TAG).mkdir()
        # a patch/ trio of our own, so the test never touches the repo's
        patch_dir = tmp / "patch"
        patch_dir.mkdir()
        self._saved = (cpf.PATCH, cpf.LIST, cpf.SOURCE)
        cpf.PATCH = str(patch_dir / "p.patch")
        cpf.LIST = str(patch_dir / "FILES.txt")
        cpf.SOURCE = str(patch_dir / "SOURCE.txt")
        Path(cpf.PATCH).write_bytes(
            _git(self.fork, "diff", cpf.BASE_TAG, self.sha, "--").encode()
            + b"\n")
        Path(cpf.LIST).write_text("a.py\n")
        Path(cpf.SOURCE).write_text(self.sha + "\n")

    def tearDown(self):
        cpf.PATCH, cpf.LIST, cpf.SOURCE = self._saved
        self.tmp.cleanup()

    def test_check_source_is_not_confused_by_the_path(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cpf.check_source(str(self.fork)), 0)

    def test_update_regenerates_through_the_ambiguous_name(self):
        # ROOT's `git diff --quiet` on the temp patch path is relative to the
        # repo root; point the dirtiness probe at the temp dir instead.
        saved_root = cpf.ROOT
        cpf.ROOT = str(Path(self.tmp.name))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cpf.update(str(self.fork)), 0)
        finally:
            cpf.ROOT = saved_root
        self.assertIn("a.py", Path(cpf.LIST).read_text())
        self.assertIn(self.sha, Path(cpf.SOURCE).read_text())


if __name__ == "__main__":
    unittest.main()
