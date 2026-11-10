"""tools/k3_b300/apply_file_patches.py and make_file_patches.py: a file patch set reproduces its target tree byte
for byte or changes nothing, and the two sets of Dockerfile.b300-k31 are consistent with each other and with the
Dockerfile's steps. CPU only, no image."""
import contextlib
import importlib.util
import io
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "k3_b300"
_spec = importlib.util.spec_from_file_location("make_file_patches", TOOLS / "make_file_patches.py")
mk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk)
ap = mk.A                                                # the applier module the maker itself uses

BASE = {
    "pkg/a.py": b"".join(b"line %d\n" % i for i in range(1, 41)),
    "pkg/sub/b.py": b"alpha\nbeta\ngamma\ndelta\n",
    "pkg/untouched.py": b"same\n",
}


def edited(base):
    out = dict(base)
    lines = base["pkg/a.py"].split(b"\n")
    lines[0] = b"first line, changed"                    # a hunk at the top of the file
    lines[19:21] = [b"twenty", b"twenty and a half", b"twenty-one"]
    del lines[30]
    lines[-2] = b"last line, changed"                    # ... and at its end
    out["pkg/a.py"] = b"\n".join(lines)
    out["pkg/sub/b.py"] = b"alpha\ngamma\ndelta\nepsilon\n"
    out["pkg/sub/new.py"] = b"created by the set\nsecond line\n"
    return out


def write_tree(root, files):
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def read_tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def run(module, argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = module.main([str(a) for a in argv])
    return rc, out.getvalue(), err.getvalue()


class Trees(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def cut(self, before, after, name="t"):
        """Write the two trees and the set cut from them; returns (set path, root holding a copy of `before`)."""
        tag = f"{len(list(self.tmp.iterdir())):02d}"
        b, a, root = self.tmp / f"{tag}-before", self.tmp / f"{tag}-after", self.tmp / f"{tag}-root"
        for d, files in ((b, before), (a, after), (root, before)):
            d.mkdir()
            write_tree(d, files)
        patch = self.tmp / f"{tag}.patch"
        rc, _, err = run(mk, ["--name", name, "--before", b, "--after", a, "--out", patch])
        self.assertEqual(rc, 0, err)
        return patch, root


class RoundTrip(Trees):
    def test_set_reproduces_the_target_tree(self):
        after = edited(BASE)
        patch, root = self.cut(BASE, after)
        rc, out, err = run(ap, [patch, "--root", root])
        self.assertEqual(rc, 0, err)
        self.assertEqual(read_tree(root), after)
        self.assertIn("pkg/a.py: patched", out)
        self.assertIn("pkg/sub/new.py: created", out)
        self.assertNotIn("untouched", patch.read_text())

    def test_second_run_changes_nothing(self):
        after = edited(BASE)
        patch, root = self.cut(BASE, after)
        self.assertEqual(run(ap, [patch, "--root", root])[0], 0)
        rc, out, _ = run(ap, [patch, "--root", root])
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("already applied"), 3)
        self.assertEqual(read_tree(root), after)

    def test_check_writes_nothing(self):
        patch, root = self.cut(BASE, edited(BASE))
        rc, out, _ = run(ap, [patch, "--root", root, "--check"])
        self.assertEqual(rc, 0)
        self.assertEqual(out.count("(not written)"), 3)
        self.assertEqual(read_tree(root), BASE)

    def test_no_temporary_file_is_left_behind(self):
        after = edited(BASE)
        patch, root = self.cut(BASE, after)
        run(ap, [patch, "--root", root])
        self.assertEqual(sorted(read_tree(root)), sorted(after))

    def test_patched_file_keeps_its_permission_bits(self):
        after = edited(BASE)
        patch, root = self.cut(BASE, after)
        (root / "pkg/a.py").chmod(0o750)
        (root / "pkg/sub/b.py").chmod(0o604)
        self.assertEqual(run(ap, [patch, "--root", root])[0], 0)
        self.assertEqual(read_tree(root), after)
        self.assertEqual((root / "pkg/a.py").stat().st_mode & 0o7777, 0o750)
        self.assertEqual((root / "pkg/sub/b.py").stat().st_mode & 0o7777, 0o604)

    def test_last_line_without_a_newline(self):
        for old, new in ((b"a\nb\nc", b"a\nb\nc\n"), (b"a\nb\nc\n", b"a\nb\nc"), (b"a\nb\nc", b"a\nB\nc"),
                         (b"a\nb\nc", b"a\nb\nd"), (b"", b"x"), (b"x", b"x\ny")):
            with self.subTest(old=old, new=new):
                patch, root = self.cut({"f.py": old}, {"f.py": new})
                self.assertEqual(run(ap, [patch, "--root", root])[0], 0)
                self.assertEqual(read_tree(root), {"f.py": new})

    def test_only_a_newline_ends_a_line(self):
        # "\r", form feed and the Unicode separators are line content, not line ends: the set stays a plain
        # unified diff of "\n"-terminated lines
        old = "a\r\nb\x0cc\nd\u2028e\nf\x1cg\nk\rl\n".encode()
        new = "a\r\nb\x0cC\nd\u2028E\nf\x1cg\nk\rL\nh\r\n".encode()
        patch, root = self.cut({"f.py": old}, {"f.py": new})
        text = patch.read_bytes()
        self.assertIn(b"@@ -1,5 +1,6 @@\n", text)
        self.assertNotIn(ap.NO_NEWLINE, text)
        self.assertIn(b"\n-k\rl\n+k\rL\n+h\r\n", text)
        self.assertEqual(run(ap, [patch, "--root", root])[0], 0)
        self.assertEqual(read_tree(root), {"f.py": new})

    def test_lines_that_read_like_diff_headers(self):
        old = b"x\n-- removed, reads '--- ...' in the diff\n++ kept\ny\n@@ -1 +1 @@ kept too\nz\n"
        new = b"x\n++ added, reads '+++ ...' in the diff\n++ kept\ny\n@@ -1 +1 @@ kept too\nZ\n"
        patch, root = self.cut({"f.py": old}, {"f.py": new})
        self.assertEqual(run(ap, [patch, "--root", root])[0], 0)
        self.assertEqual(read_tree(root), {"f.py": new})

    def test_blank_context_lines_may_lose_their_space(self):
        old = b"a\n\nb\n\nc\n"
        new = b"a\n\nB\n\nc\n"
        patch, root = self.cut({"f.py": old}, {"f.py": new})
        stripped = b"\n".join(b"" if line == b" " else line for line in patch.read_bytes().split(b"\n"))
        self.assertNotEqual(stripped, patch.read_bytes())
        patch.write_bytes(stripped)
        self.assertEqual(run(ap, [patch, "--root", root])[0], 0)
        self.assertEqual(read_tree(root), {"f.py": new})


class Refusals(Trees):
    def assert_refused(self, patch, root, tree, message):
        for extra in (["--check"], []):
            rc, _, err = run(ap, [patch, "--root", root, *extra])
            self.assertEqual(rc, 1)
            self.assertIn(message, err)
            self.assertEqual(read_tree(root), tree, "a refused set must not write anything")

    def test_another_base_file_stops_the_whole_set(self):
        patch, root = self.cut(BASE, edited(BASE))
        other = dict(BASE, **{"pkg/sub/b.py": BASE["pkg/sub/b.py"] + b"# local edit\n"})
        write_tree(root, other)
        self.assert_refused(patch, root, other, "pkg/sub/b.py has sha256")

    def test_missing_file(self):
        patch, root = self.cut(BASE, edited(BASE))
        (root / "pkg/a.py").unlink()
        tree = read_tree(root)
        self.assert_refused(patch, root, tree, "pkg/a.py is missing")

    def test_target_that_is_not_a_regular_file(self):
        patch, root = self.cut(BASE, edited(BASE))
        (root / "pkg/sub/new.py").mkdir()
        self.assert_refused(patch, root, BASE, "pkg/sub/new.py is not a regular file")
        (root / "pkg/sub/new.py").rmdir()
        (root / "pkg/sub/b.py").unlink()
        (root / "pkg/sub/b.py").symlink_to("../untouched.py")
        tree = read_tree(root)
        self.assert_refused(patch, root, tree, "pkg/sub/b.py is not a regular file")
        self.assertTrue((root / "pkg/sub/b.py").is_symlink())

    def test_created_file_already_there_with_other_content(self):
        patch, root = self.cut(BASE, edited(BASE))
        tree = dict(BASE, **{"pkg/sub/new.py": b"someone else's file\n"})
        write_tree(root, tree)
        self.assert_refused(patch, root, tree, "the set was cut against no file")

    def test_edited_added_line_fails_the_result_hash(self):
        patch, root = self.cut(BASE, edited(BASE))
        patch.write_bytes(patch.read_bytes().replace(b"+twenty and a half", b"+twenty and a quarter"))
        self.assert_refused(patch, root, BASE, "the patched file has sha256")

    def test_edited_context_line_does_not_match(self):
        patch, root = self.cut(BASE, edited(BASE))
        text = patch.read_bytes()
        self.assertIn(b"\n line 18\n", text)
        patch.write_bytes(text.replace(b"\n line 18\n", b"\n line 18 \n"))
        self.assert_refused(patch, root, BASE, "the set expects")

    def test_manifest_and_sections_must_name_the_same_files(self):
        patch, root = self.cut(BASE, edited(BASE))
        lines = patch.read_bytes().split(b"\n")
        dropped = [line for line in lines if not (line.startswith(b"# sha256") and line.endswith(b"pkg/sub/b.py"))]
        self.assertEqual(len(dropped), len(lines) - 1)
        patch.write_bytes(b"\n".join(dropped))
        self.assert_refused(patch, root, BASE, "manifest and diff sections name different files")

    def test_path_outside_the_root(self):
        for outside in (b"../f.py", b"sub/../../f.py", b"/abs/f.py"):
            with self.subTest(path=outside):
                patch, root = self.cut({"f.py": b"a\n"}, {"f.py": b"b\n"})
                text = patch.read_bytes()
                for old in (b" f.py\n", b"--- a/f.py\n", b"+++ b/f.py\n"):
                    self.assertEqual(text.count(old), 1)
                    text = text.replace(old, old[:-5] + outside + b"\n")
                patch.write_bytes(text)
                self.assert_refused(patch, root, {"f.py": b"a\n"}, "path leaves the root")

    def test_truncated_set(self):
        patch, root = self.cut(BASE, edited(BASE))
        lines = patch.read_bytes().split(b"\n")
        self.assertEqual(lines[-1], b"")
        self.assertEqual(lines[-4:-1], [b"@@ -0,0 +1,2 @@", b"+created by the set", b"+second line"])
        patch.write_bytes(b"\n".join(lines[:-3]) + b"\n")      # the last hunk without its two lines
        self.assert_refused(patch, root, BASE, "the last hunk is truncated")
        patch.write_bytes(b"\n".join(lines[:-2])[:-4])          # ... ending inside its first line
        self.assert_refused(patch, root, BASE, "the last hunk is truncated")
        patch.write_bytes(b"\n".join(lines[:-3])[:-4])          # ... ending inside its header
        self.assert_refused(patch, root, BASE, "malformed hunk header")

    def test_maker_refuses_what_a_set_cannot_carry(self):
        cases = (
            ({"f.py": b"a\n", "g.py": b"b\n"}, {"f.py": b"A\n"}, "does not delete"),
            ({"f.py": b"a\n"}, {"f.py": b"\xff\xfe\n"}, "not UTF-8"),
            ({"f.py": b"a\n"}, {"f.py": b"a\n"}, "identical"),
        )
        for n, (before, after, message) in enumerate(cases):
            with self.subTest(message=message):
                b, a = self.tmp / f"m{n}-before", self.tmp / f"m{n}-after"
                for d, files in ((b, before), (a, after)):
                    d.mkdir()
                    write_tree(d, files)
                out = self.tmp / f"m{n}.patch"
                rc, _, err = run(mk, ["--name", "t", "--before", b, "--after", a, "--out", out])
                self.assertEqual(rc, 1)
                self.assertIn(message, err)
                self.assertFalse(out.exists())


    def test_maker_emits_only_a_set_that_reproduces_the_tree(self):
        b, a, out = self.tmp / "before", self.tmp / "after", self.tmp / "t.patch"
        for d, files in ((b, BASE), (a, edited(BASE))):
            d.mkdir()
            write_tree(d, files)
        real = mk.diff_text
        mk.diff_text = lambda rel, old, new: real(rel, old, new).replace(b"+twenty and a half", b"+twenty and a quarter")
        try:
            rc, _, err = run(mk, ["--name", "t", "--before", b, "--after", a, "--out", out])
        finally:
            mk.diff_text = real
        self.assertEqual(rc, 1)
        self.assertIn("does not reproduce the file", err)
        self.assertFalse(out.exists())


class K31Sets(unittest.TestCase):
    """The three sets Dockerfile.b300-k31 applies."""

    NAMES = ("r2", "r3", "api")

    @classmethod
    def setUpClass(cls):
        cls.sets = {}
        for name in cls.NAMES:
            set_name, manifest, hunks = ap.parse((TOOLS / "k31" / f"{name}.patch").read_bytes())
            cls.sets[name] = (set_name, {path: (before, after) for path, before, after in manifest}, hunks)
        cls.dockerfile = (ROOT / "Dockerfile.b300-k31").read_text()

    def test_sets_parse_and_name_themselves(self):
        self.assertEqual([self.sets[name][0] for name in self.NAMES], ["k31-r2", "k31-r3", "k31-api"])
        self.assertEqual([len(self.sets[name][1]) for name in self.NAMES], [10, 8, 2])
        for _, manifest, hunks in self.sets.values():
            for path, (before, after) in manifest.items():
                self.assertTrue(path.startswith("vllm/") and path.endswith(".py"), path)
                self.assertRegex(after, r"^[0-9a-f]{64}$")
                self.assertRegex(before, r"^([0-9a-f]{64}|-)$")
                self.assertNotEqual(before, after)
                self.assertTrue(hunks[path], path)

    def test_r2_creates_exactly_the_context_kv_module(self):
        created = [p for p, (before, _) in self.sets["r2"][1].items() if before == "-"]
        self.assertEqual(created, ["vllm/models/kimi_k3/nvidia/specialized_decode/drafter/context_kv.py"])
        for name in ("r3", "api"):
            self.assertFalse([p for p, (before, _) in self.sets[name][1].items() if before == "-"], name)

    def test_r3_is_cut_against_the_files_r2_leaves(self):
        r2, r3 = self.sets["r2"][1], self.sets["r3"][1]
        shared = sorted(set(r2) & set(r3))
        self.assertEqual([p.rsplit("/", 1)[1] for p in shared], ["linears.py", "markov.py", "model.py"])
        for path in shared:
            self.assertEqual(r3[path][0], r2[path][1], path)

    def test_api_set_is_independent_of_the_decode_step_sets(self):
        api = set(self.sets["api"][1])
        self.assertEqual(sorted(api), ["vllm/entrypoints/openai/responses/utils.py", "vllm/renderers/kimi_k3.py"])
        self.assertFalse(api & (set(self.sets["r2"][1]) | set(self.sets["r3"][1])))

    def test_dockerfile_steps_have_a_build_arg_and_a_label(self):
        for name in self.NAMES:
            arg = f"VLLM_MOET_K31_{name.upper()}_PATCH"
            self.assertIn(f"\nARG {arg}=1\nRUN if [ \"${arg}\" = \"1\" ]; then ", self.dockerfile)
            self.assertIn(f'com.vllm-moet.k31.{name}="${{{arg}}}"', self.dockerfile)

    def test_dockerfile_steps_compile_the_files_of_their_set(self):
        for name in self.NAMES:
            block = re.search(rf"RUN if \[[^\n]*k31/{name}\.patch --check.*?; fi\n", self.dockerfile, re.S)
            self.assertIsNotNone(block, name)
            compiled = block.group(0).split("py_compile", 1)[1].split("&&", 1)[0]
            listed = {"vllm/" + token for token in compiled.replace("\\", " ").split()}
            self.assertEqual(listed, set(self.sets[name][1]), name)

    def test_dockerfile_copies_the_applier_and_the_sets(self):
        self.assertIn("COPY tools/k3_b300/apply_file_patches.py /opt/vllm-moet/k3_b300/", self.dockerfile)
        self.assertIn("COPY tools/k3_b300/k31/ /opt/vllm-moet/k3_b300/k31/", self.dockerfile)
        self.assertEqual(sorted(p.name for p in (TOOLS / "k31").iterdir()), ["api.patch", "r2.patch", "r3.patch"])


if __name__ == "__main__":
    unittest.main()
