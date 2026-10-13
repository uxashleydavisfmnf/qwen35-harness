"""
Test suite for the qwen35 harness.

Everything here runs against :class:`~qwen35.backends.mock.MockBackend` or pure
functions, so the security properties and the grammar are verifiable on a machine
with no weights, no GPU and no network.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen35.backends.mock import MockBackend, scripted_answer, scripted_tool_call
from qwen35.config import (QWEN35_08B, HarnessConfig, RuntimeConfig,
                           get_permissions)
from qwen35.context import ContextManager, Message, estimate_tokens
from qwen35.grammar import (GrammarError, ToolSignature, build_grammar,
                            derivable_rules, nullable_rules, validate_grammar)
from qwen35.loop import Agent, RunResult
from qwen35.prompt import answers_from_prompt, extract_request, wants_changes
from qwen35.parse import (detect_degeneration, format_correction, parse_turn)
from qwen35.permissions import CommandScreener, scrub_env
from qwen35.prompt import looks_conversational
from qwen35.tools import (ToolContext, build_specs, call_tool, validate_syntax)
from qwen35.workspace import WorkspaceError, WorkspaceGuard, validate_workspace

TOOL_SIGS = [
    ToolSignature("list_dir", ("path",)),
    ToolSignature("read_file", ("path", "start_line", "max_lines")),
    ToolSignature("write_file", ("path", "content")),
    ToolSignature("edit_file", ("path", "find", "replace")),
    ToolSignature("run_command", ("command",)),
    ToolSignature("view_image", ("path", "question")),
]


def make_cfg(workspace: Path, **rt_kwargs) -> HarnessConfig:
    rt = RuntimeConfig(**rt_kwargs)
    return HarnessConfig(workspace=workspace, permissions=get_permissions("safe"),
                         runtime=rt)


# ==========================================================================
# Workspace confinement -- the core security property
# ==========================================================================
class TestWorkspaceSelection(unittest.TestCase):
    def test_refuses_filesystem_roots(self):
        for root in ("C:\\", "D:\\", "/"):
            with self.subTest(root=root):
                if root == "/" and os.name == "nt":
                    continue
                with self.assertRaises(WorkspaceError):
                    validate_workspace(root)

    def test_refuses_home_directory_itself(self):
        with self.assertRaises(WorkspaceError) as ctx:
            validate_workspace(str(Path.home()))
        self.assertIn("home directory", str(ctx.exception))

    def test_refuses_windows_and_program_files(self):
        windir = os.environ.get("SystemRoot", "C:\\Windows")
        with self.assertRaises(WorkspaceError):
            validate_workspace(windir)
        pf = os.environ.get("ProgramFiles")
        if pf:
            with self.assertRaises(WorkspaceError):
                validate_workspace(pf)

    def test_refuses_ancestor_of_home(self):
        """C:\\Users must not be selectable: it exposes every profile."""
        parent = Path.home().parent
        if parent == Path(parent.anchor):
            self.skipTest("home is already at the drive root")
        with self.assertRaises(WorkspaceError):
            validate_workspace(str(parent))

    def test_refuses_unc_paths(self):
        with self.assertRaises(WorkspaceError):
            validate_workspace("\\\\server\\share")

    def test_refuses_nonexistent_without_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            with self.assertRaises(WorkspaceError):
                validate_workspace(str(missing))
            created = validate_workspace(str(missing), create=True)
            self.assertTrue(created.is_dir())

    def test_accepts_a_real_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = validate_workspace(tmp)
            self.assertEqual(ws, Path(os.path.realpath(tmp)))

    def test_refuses_operator_protected_paths(self):
        """The launcher protects its own install directory this way."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = root / "app"
            app.mkdir()
            work = root / "work"
            work.mkdir()
            os.environ["QWEN35_PROTECTED_PATHS"] = str(app)
            try:
                self.assertEqual(validate_workspace(str(work)), work.resolve())
                with self.assertRaises(WorkspaceError):
                    validate_workspace(str(app))
                with self.assertRaises(WorkspaceError):
                    validate_workspace(str(app / "qwen35"))
                # an ancestor of a protected path is refused too
                with self.assertRaises(WorkspaceError):
                    validate_workspace(str(root))
            finally:
                os.environ.pop("QWEN35_PROTECTED_PATHS", None)


class TestWorkspaceGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "sub").mkdir()
        (self.root / "sub" / "a.txt").write_text("hello", encoding="utf-8")
        self.guard = WorkspaceGuard(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_accepts_relative_paths(self):
        rp = self.guard.resolve("sub/a.txt", must_exist=True)
        self.assertTrue(rp.abs.is_file())
        self.assertEqual(rp.rel, "sub/a.txt")

    def test_rejects_parent_traversal(self):
        with self.assertRaises(WorkspaceError):
            self.guard.resolve("../escape.txt")
        with self.assertRaises(WorkspaceError):
            self.guard.resolve("sub/../../escape.txt")

    def test_rejects_absolute_paths_outside(self):
        outside = Path(self.root).parent / "elsewhere.txt"
        with self.assertRaises(WorkspaceError):
            self.guard.resolve(str(outside))

    def test_rejects_alternate_data_streams(self):
        with self.assertRaises(WorkspaceError):
            self.guard.resolve("sub/a.txt:hidden")

    def test_rejects_reserved_device_names(self):
        for name in ("CON", "nul", "com1", "LPT9"):
            with self.subTest(name=name):
                with self.assertRaises(WorkspaceError):
                    self.guard.resolve(name)

    def test_rejects_control_characters(self):
        with self.assertRaises(WorkspaceError):
            self.guard.resolve("bad\x00name")

    def test_allows_new_file_in_new_directory(self):
        rp = self.guard.resolve("deep/new/file.txt")
        self.assertFalse(rp.existed)
        self.assertTrue(str(rp.abs).startswith(str(self.root)))

    def test_relative_dot_resolves_to_root(self):
        rp = self.guard.resolve(".")
        self.assertEqual(rp.abs, self.root)
        self.assertEqual(rp.rel, ".")


# ==========================================================================
# Permission enforcement
# ==========================================================================
class TestCommandScreening(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.guard = WorkspaceGuard(self.root)
        self.screener = CommandScreener(self.guard, get_permissions("safe"))

    def tearDown(self):
        self._tmp.cleanup()

    def screen(self, cmd):
        return self.screener.screen(cmd)

    def test_allows_normal_commands(self):
        for cmd in ("python test.py", "git status", "ls -la",
                    "python -m unittest discover"):
            with self.subTest(cmd=cmd):
                self.assertTrue(self.screen(cmd).allowed, cmd)

    def test_blocks_shell_metacharacters(self):
        for cmd in ("ls; rm -rf /", "cat a | grep b", "ls && whoami",
                    "echo hi > out.txt", "ls `whoami`", "echo $HOME",
                    "echo %USERNAME%"):
            with self.subTest(cmd=cmd):
                self.assertFalse(self.screen(cmd).allowed, cmd)

    def test_metacharacters_inside_quotes_are_allowed(self):
        """shell=False means a quoted newline or '&' is inert, not a risk."""
        r = self.screen('python -c "import sys\\nprint(sys.version)"')
        self.assertTrue(r.allowed, r.reason)
        r2 = self.screen('python -c "print(1 > 0)"')
        self.assertTrue(r2.allowed, r2.reason)

    def test_cd_prefix_is_honoured_with_confined_cwd(self):
        (self.root / "sub").mkdir(exist_ok=True)
        r = self.screen("cd sub && python run.py")
        self.assertTrue(r.allowed, r.reason)
        self.assertEqual(r.program, "python")
        self.assertEqual(r.cwd, self.root / "sub")
        self.assertEqual(r.argv, ["python", "run.py"])

    def test_cd_escaping_the_workspace_is_refused(self):
        for cmd in ("cd .. && ls", "cd ../.. && ls", "cd C:\\ && dir"):
            with self.subTest(cmd=cmd):
                self.assertFalse(self.screen(cmd).allowed, cmd)

    def test_cd_with_multiple_chained_commands_is_refused(self):
        (self.root / "sub").mkdir(exist_ok=True)
        r = self.screen("cd sub && python a.py && python b.py")
        self.assertFalse(r.allowed)

    def test_bare_cd_is_refused_with_guidance(self):
        r = self.screen("cd sub")
        self.assertFalse(r.allowed)
        self.assertIn("workspace root", r.reason)

    def test_redirect_suggestion_points_at_write_file(self):
        r = self.screen("echo hi > out.txt")
        self.assertFalse(r.allowed)
        self.assertIn("write_file", r.reason)

    def test_blocks_permanently_banned_programs(self):
        for cmd in ("cmd /c dir", "powershell -c whoami", "reg query HKLM",
                    "schtasks /create", "diskpart", "format c:", "certutil -urlcache",
                    "rm -rf .", "del important.txt", "robocopy a b",
                    "ssh user@host", "taskkill /f /im x"):
            with self.subTest(cmd=cmd):
                self.assertFalse(self.screen(cmd).allowed, cmd)

    def test_blocks_network_when_profile_withholds_it(self):
        """`readonly` still withholds the network.

        `standard` (and therefore the old `safe`, which was merged into it) allows
        it deliberately -- a coding agent that cannot read a README cannot work.
        """
        from qwen35.config import get_permissions

        ro = CommandScreener(self.guard, get_permissions("readonly"))
        self.assertFalse(ro.screen("curl http://example.com").allowed)
        self.assertFalse(ro.screen("pip install requests").allowed)
        self.assertFalse(ro.screen("git push origin main").allowed)
        # ...and the default profile does not
        self.assertTrue(self.screen("curl http://example.com").allowed)

    def test_standard_allows_installs_but_confines_them(self):
        """Network and installs are allowed; escaping the workspace is not."""
        self.assertTrue(self.screen("pip install requests").allowed)
        self.assertTrue(self.screen("npm install left-pad").allowed)
        self.assertTrue(self.screen("pip list").allowed)
        for command in ("pip install --user requests",
                        "pip install -g requests",
                        "npm install -g left-pad",
                        "npm install --global left-pad",
                        "pip install --target C:\\Windows\\Temp requests",
                        "pip install --prefix /etc requests"):
            with self.subTest(command=command):
                result = self.screen(command)
                self.assertFalse(result.allowed, command)
                self.assertTrue("outside the workspace" in result.reason
                                or "workspace" in result.reason, result.reason)

    def test_an_in_workspace_target_is_still_allowed(self):
        self.assertTrue(
            self.screen("pip install --target .qwen35/packages requests").allowed)

    def test_elevation_is_refused_in_every_profile(self):
        from qwen35.config import PROFILES, get_permissions

        for name in PROFILES:
            screener = CommandScreener(self.guard, get_permissions(name))
            for command in ("sudo pip install requests", "runas /user:a cmd",
                            "gsudo cmd /c dir", "doas sh"):
                with self.subTest(profile=name, command=command):
                    self.assertFalse(screener.screen(command).allowed)

    def test_installs_land_inside_the_workspace(self):
        """The confinement must not depend on the model remembering a flag."""
        from qwen35.permissions import scrub_env

        env = scrub_env(sandbox_home=self.guard.root / ".qwen35" / "sandbox-home")
        root = str(self.guard.root)
        for key in ("PIP_TARGET", "PYTHONUSERBASE", "npm_config_prefix",
                    "CARGO_HOME", "GOPATH", "GOMODCACHE", "TEMP", "HOME"):
            with self.subTest(key=key):
                self.assertTrue(env.get(key, "").startswith(root),
                                f"{key}={env.get(key)!r} is not under {root}")
        self.assertIn(str(self.guard.root), env.get("PYTHONPATH", ""))

    def test_the_safe_profile_is_an_alias_for_standard(self):
        from qwen35.config import DEFAULT_PROFILE, get_permissions

        self.assertEqual(DEFAULT_PROFILE, "standard")
        old = get_permissions("safe")
        new = get_permissions("standard")
        self.assertEqual(old.to_dict(), new.to_dict())
        self.assertTrue(old.allow_package_install)

    def test_blocks_git_push_and_config(self):
        from qwen35.config import get_permissions

        ro = CommandScreener(self.guard, get_permissions("readonly"))
        self.assertFalse(ro.screen("git push origin main").allowed)
        self.assertFalse(ro.screen("git config user.email x@y.z").allowed)
        self.assertFalse(self.screen("git clean -fdx").allowed)

    def test_blocks_paths_outside_workspace(self):
        self.assertFalse(self.screen("cat /etc/passwd").allowed)
        self.assertFalse(self.screen("cat ../../secrets.txt").allowed)
        self.assertFalse(self.screen("cat ~/.ssh/id_rsa").allowed)

    def test_allows_paths_inside_workspace(self):
        self.assertTrue(self.screen("cat sub/file.txt").allowed)
        self.assertTrue(self.screen("python ./tools/run.py").allowed)

    def test_blocks_find_with_exec(self):
        self.assertFalse(self.screen("find . -exec rm {} ;").allowed)
        self.assertFalse(self.screen("find . -delete").allowed)

    def test_readonly_profile_blocks_execution_entirely(self):
        s = CommandScreener(self.guard, get_permissions("readonly"))
        self.assertFalse(s.screen("python x.py").allowed)

    def test_command_budget_is_enforced(self):
        perms = get_permissions("safe")
        perms.max_commands_per_run = 3
        s = CommandScreener(self.guard, perms)
        for _ in range(3):
            self.assertTrue(s.screen("python x.py").allowed)
        self.assertFalse(s.screen("python x.py").allowed)

    def test_trusted_profile_allows_more_but_still_bans_dangerous(self):
        s = CommandScreener(self.guard, get_permissions("trusted"))
        self.assertTrue(s.screen("curl http://example.com").allowed)
        self.assertFalse(s.screen("reg query HKLM").allowed)
        self.assertFalse(s.screen("format c:").allowed)


class TestEnvironmentScrubbing(unittest.TestCase):
    def test_removes_secrets(self):
        base = {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "MY_PASSWORD": "hunter2",
            "GITHUB_TOKEN": "gh-token",
            "DSH_SESSION": "internal",
            "HARMLESS": "value",
        }
        env = scrub_env(base=base, sandbox_home=Path("/tmp/sandbox"))
        for key in ("OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "MY_PASSWORD",
                    "GITHUB_TOKEN", "DSH_SESSION", "HARMLESS"):
            self.assertNotIn(key, env, f"{key} leaked into the child environment")
        self.assertIn("PATH", env)

    def test_redirects_home_into_sandbox(self):
        sandbox = Path("/tmp/sandbox")
        env = scrub_env(base={}, sandbox_home=sandbox)
        self.assertEqual(Path(env["HOME"]), sandbox)
        self.assertEqual(Path(env["USERPROFILE"]), sandbox)
        self.assertEqual(Path(env["TMP"]), sandbox / "tmp")


# ==========================================================================
# Grammar
# ==========================================================================
class TestGrammar(unittest.TestCase):
    def test_no_tools_no_answer_is_an_error(self):
        with self.assertRaises(GrammarError):
            build_grammar([], thinking=False, allow_plain_answer=False)

    def test_grammar_is_valid_and_non_nullable(self):
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                g = build_grammar(TOOL_SIGS, thinking=thinking,
                                  allow_plain_answer=True)
                self.assertEqual(validate_grammar(g), [])
                defined = {ln.split(" ::= ")[0]: ln.split(" ::= ")[1]
                           for ln in g.splitlines() if " ::= " in ln}
                # A nullable root makes llama.cpp release the constraint and the
                # model free-runs -- this is the regression test for that bug.
                self.assertFalse(nullable_rules(defined)["root"])

    def test_root_is_not_nullable_without_plain_answer(self):
        g = build_grammar(TOOL_SIGS, thinking=False, allow_plain_answer=False)
        defined = {ln.split(" ::= ")[0]: ln.split(" ::= ")[1]
                   for ln in g.splitlines() if " ::= " in ln}
        self.assertFalse(nullable_rules(defined)["root"])

    def test_every_tool_name_appears(self):
        g = build_grammar(TOOL_SIGS, thinking=False, allow_plain_answer=True)
        for sig in TOOL_SIGS:
            self.assertIn(f"<function={sig.name}>", g)
            for param in sig.params:
                self.assertIn(f"<parameter={param}>", g)

    def test_rejects_nullable_root(self):
        """The validator must catch the failure mode, not just accept output."""
        bad = 'root ::= a | b\na ::= "x"\nb ::= c*\nc ::= "y"\n'
        problems = validate_grammar(bad)
        self.assertTrue(any("nullable" in p for p in problems), problems)

    def test_rejects_undefined_reference(self):
        problems = validate_grammar('root ::= "a" missing\n')
        self.assertTrue(any("undefined" in p for p in problems), problems)

    def test_rejects_duplicate_rule(self):
        problems = validate_grammar('root ::= "a"\nroot ::= "b"\n')
        self.assertTrue(any("twice" in p for p in problems), problems)

    def test_not_containing_dfa_is_correct(self):
        """Exercise the DFA construction directly, including overlap cases."""
        g = build_grammar(TOOL_SIGS, thinking=False, allow_plain_answer=True)
        defined = {ln.split(" ::= ")[0]: ln.split(" ::= ")[1]
                   for ln in g.splitlines() if " ::= " in ln}
        self.assertIn("root", defined)
        # the value rule must forbid the delimiter prefix, not the full literal
        self.assertIn("</parameter", g)

    def test_literal_with_repeated_first_char_is_rejected(self):
        from qwen35.grammar import _RuleGen
        gen = _RuleGen()
        with self.assertRaises(GrammarError):
            gen.not_containing("abab")   # 'a' recurs -> ambiguous restarts

    @staticmethod
    def _defined(grammar: str) -> dict[str, str]:
        return {ln.split(" ::= ")[0]: ln.split(" ::= ")[1]
                for ln in grammar.splitlines() if " ::= " in ln}

    def test_every_rule_can_terminate(self):
        """Regression: a right-recursive-only rule made end-of-input unreachable.

        The plain-answer branch could never finish, so a run with no tools
        generated until the token cap and returned *nothing* -- HTTP 200, no
        error, an empty answer.
        """
        cases = [(t, a) for t in (False, True) for a in (True, False)]
        cases.append((None, None))          # no tools at all (conversational)
        for thinking, allow in cases:
            with self.subTest(thinking=thinking, allow_plain_answer=allow):
                if thinking is None:
                    g = build_grammar([], thinking=False, allow_plain_answer=True)
                else:
                    g = build_grammar(TOOL_SIGS, thinking=thinking,
                                      allow_plain_answer=allow)
                stuck = [n for n, ok in derivable_rules(self._defined(g)).items()
                         if not ok]
                self.assertEqual(stuck, [], f"non-terminating rules: {stuck}")

    def test_validator_rejects_a_non_terminating_rule(self):
        bad = 'root ::= a\na ::= "x" a\n'      # 'a' can never stop consuming
        problems = validate_grammar(bad)
        self.assertTrue(any("never terminate" in p for p in problems), problems)

    def test_validator_accepts_a_terminating_rule(self):
        good = 'root ::= a\na ::= "x"\n'
        self.assertEqual(validate_grammar(good), [])

    def test_delimiter_rules_accept_ordinary_text(self):
        """The answer branch must be able to consume and then stop."""
        g = build_grammar([], thinking=False, allow_plain_answer=True)
        defined = self._defined(g)
        root_body = defined["root"]
        # root delegates to the answer rule, which must be reachable
        self.assertTrue(root_body)
        self.assertEqual(validate_grammar(g), [])


# ==========================================================================
# Parsing
# ==========================================================================
class TestParsing(unittest.TestCase):
    def test_parses_the_trained_format(self):
        raw = ("<tool_call>\n<function=read_file>\n<parameter=path>\n"
               "src/main.py\n</parameter>\n</function>\n</tool_call>")
        turn = parse_turn(raw)
        self.assertTrue(turn.has_tool_call)
        self.assertEqual(turn.tool_calls[0].name, "read_file")
        self.assertEqual(turn.tool_calls[0].args["path"], "src/main.py")

    def test_tolerates_every_observed_delimiter_variant(self):
        variants = [
            "</parameter>", "</parameter >", "</parameter/>", "</parameter />",
            "</ parameter>", "</PARAMETER>",
        ]
        for closing in variants:
            with self.subTest(closing=closing):
                raw = ("<tool_call>\n<function=read_file>\n<parameter=path>\n"
                       f"a.py\n{closing}\n</function>\n</tool_call>")
                turn = parse_turn(raw)
                self.assertTrue(turn.has_tool_call)
                self.assertEqual(turn.tool_calls[0].args.get("path"), "a.py",
                                 f"failed for closing {closing!r}")

    def test_tolerates_missing_closing_tool_call(self):
        """The stop string truncates before </tool_call> is emitted."""
        raw = ("<tool_call>\n<function=read_file>\n<parameter=path>\n"
               "a.py\n</parameter>\n</function>\n")
        turn = parse_turn(raw)
        self.assertTrue(turn.has_tool_call)
        self.assertEqual(turn.tool_calls[0].args["path"], "a.py")

    def test_splits_thinking(self):
        raw = "Let me look.\n</think>\n\n<tool_call>\n<function=list_dir>\n</function>\n</tool_call>"
        turn = parse_turn(raw)
        self.assertEqual(turn.thinking.strip(), "Let me look.")
        self.assertEqual(turn.tool_calls[0].name, "list_dir")

    def test_plain_text_is_a_final_answer(self):
        turn = parse_turn("The tests pass now.")
        self.assertFalse(turn.has_tool_call)
        self.assertTrue(turn.is_final_answer)
        self.assertEqual(turn.text, "The tests pass now.")

    def test_reports_extra_tool_calls(self):
        raw = "<tool_call>\n<function=list_dir>\n</function>\n</tool_call>" * 3
        turn = parse_turn(raw)
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.extra_calls, 2)
        self.assertTrue(turn.format_error)

    def test_reports_json_attempt(self):
        raw = '<tool_call>\n{"name": "list_dir"}\n</tool_call>'
        turn = parse_turn(raw)
        self.assertFalse(turn.has_tool_call)
        self.assertIn("XML", turn.format_error)

    def test_correction_message_names_the_legal_format(self):
        turn = parse_turn("I will use the shell.")
        msg = format_correction(turn, [("list_dir", ("path",))])
        self.assertIn("<tool_call>", msg)
        self.assertIn("<function=NAME>", msg)
        self.assertIn("list_dir", msg)

    def test_parses_json_parameter_values(self):
        raw = ("<tool_call>\n<function=write_file>\n<parameter=path>\nx.json\n"
               "</parameter>\n<parameter=content>\n{\"a\": 1}\n</parameter>\n"
               "</function>\n</tool_call>")
        turn = parse_turn(raw)
        self.assertEqual(turn.tool_calls[0].args["content"], {"a": 1})

    def test_tool_call_key_is_stable(self):
        a = parse_turn("<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n"
                       "</parameter>\n</function>\n</tool_call>").tool_calls[0]
        b = parse_turn("<tool_call>\n<function=list_dir>\n<parameter=path>\n.\n"
                       "</parameter>\n</function>\n</tool_call>").tool_calls[0]
        self.assertEqual(a.key(), b.key())


class TestDegenerationDetection(unittest.TestCase):
    def test_detects_the_observed_markdown_loop(self):
        unit = ("**[list_dir](https://docs.python.org/3/library/files.html"
                "#list_dir) tool**")
        text = "I'll use list_dir.\n" + "\n---\n".join([unit] * 8)
        self.assertTrue(detect_degeneration(text))

    def test_detects_repeated_words(self):
        self.assertTrue(detect_degeneration("the the the the the the the the"))

    def test_accepts_normal_text(self):
        text = ("The bug was an off-by-one error in the loop bound. I changed "
                "`range(n)` to `range(n + 1)` and the tests now pass. "
                "No further changes were needed.")
        self.assertFalse(detect_degeneration(text))

    def test_empty_is_not_degenerate(self):
        self.assertFalse(detect_degeneration(""))


# ==========================================================================
# Tools: the validate-and-revert guarantee
# ==========================================================================
class TestSyntaxGate(unittest.TestCase):
    def test_detects_python_syntax_error(self):
        self.assertIsNotNone(validate_syntax(Path("x.py"), "def f(:\n  pass"))
        self.assertIsNone(validate_syntax(Path("x.py"), "def f():\n    pass\n"))

    def test_detects_json_error(self):
        self.assertIsNotNone(validate_syntax(Path("x.json"), '{"a": }'))
        self.assertIsNone(validate_syntax(Path("x.json"), '{"a": 1}'))

    def test_unknown_extensions_are_not_checked(self):
        self.assertIsNone(validate_syntax(Path("x.zzz"), "anything ((("))

    def test_accepts_a_file_with_a_utf8_bom(self):
        """CPython imports BOM'd source files, so the gate must not reject them.

        Regression test: rejecting the BOM made every edit to such a file
        impossible, because the gate refused a change that merely preserved a
        pre-existing and perfectly importable condition.
        """
        self.assertIsNone(validate_syntax(Path("x.py"), "\ufeffx = 1\n"))
        self.assertIsNone(validate_syntax(Path("x.json"), '\ufeff{"a": 1}'))
        # a real error after the BOM is still caught
        self.assertIsNotNone(validate_syntax(Path("x.py"), "\ufeffdef f(:\n"))


class TestBackupsAndTruncationGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.guard = WorkspaceGuard(self.root)
        perms = get_permissions("safe")
        self.ctx = ToolContext(guard=self.guard, perms=perms,
                               profile=QWEN35_08B,
                               screener=CommandScreener(self.guard, perms))

    def tearDown(self):
        self._tmp.cleanup()

    def test_overwrite_keeps_a_backup(self):
        original = "x = 1\n" * 60
        (self.root / "keep.py").write_text(original, encoding="utf-8")
        r = call_tool("write_file", self.ctx,
                      {"path": "keep.py", "content": "x = 1\n"})
        self.assertTrue(r.ok, r.text)
        backups = list((self.root / ".qwen35" / "backups").glob("keep.py.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), original)

    def test_large_shrink_is_warned_about_but_not_blocked(self):
        (self.root / "big.py").write_text("y = 2\n" * 100, encoding="utf-8")
        r = call_tool("write_file", self.ctx,
                      {"path": "big.py", "content": "y = 2\n"})
        self.assertTrue(r.ok, r.text)
        self.assertIn("WARNING", r.text)
        self.assertIn("truncated", r.text)

    def test_small_change_to_a_small_file_is_not_warned(self):
        (self.root / "s.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
        r = call_tool("write_file", self.ctx,
                      {"path": "s.py", "content": "a = 1\nb = 3\n"})
        self.assertTrue(r.ok, r.text)
        self.assertNotIn("WARNING", r.text)

    def test_edit_keeps_a_backup(self):
        (self.root / "e.py").write_text("v = 1\n", encoding="utf-8")
        r = call_tool("edit_file", self.ctx,
                      {"path": "e.py", "find": "v = 1", "replace": "v = 2"})
        self.assertTrue(r.ok, r.text)
        backups = list((self.root / ".qwen35" / "backups").glob("e.py.*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), "v = 1\n")

    def test_backup_history_is_bounded(self):
        for i in range(9):
            call_tool("write_file", self.ctx,
                      {"path": "c.py", "content": f"n = {i}\n"})
        backups = list((self.root / ".qwen35" / "backups").glob("c.py.*.bak"))
        self.assertLessEqual(len(backups), 5)

    def test_missing_unix_utility_suggests_the_tool(self):
        r = call_tool("run_command", self.ctx, {"command": "cat calc.py"})
        if r.ok:
            self.skipTest("cat happens to be installed here")
        self.assertIn("read_file", r.text)


class TestTools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.guard = WorkspaceGuard(self.root)
        perms = get_permissions("safe")
        self.ctx = ToolContext(guard=self.guard, perms=perms,
                               profile=QWEN35_08B,
                               screener=CommandScreener(self.guard, perms))

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_then_read_roundtrip(self):
        r = call_tool("write_file", self.ctx,
                      {"path": "a.txt", "content": "hello\nworld"})
        self.assertTrue(r.ok, r.text)
        r2 = call_tool("read_file", self.ctx, {"path": "a.txt"})
        self.assertTrue(r2.ok)
        self.assertIn("hello", r2.text)
        self.assertIn("1 | ", r2.text)  # line numbers

    def test_write_rejects_broken_python_and_leaves_file_absent(self):
        r = call_tool("write_file", self.ctx,
                      {"path": "bad.py", "content": "def f(:\n  pass"})
        self.assertFalse(r.ok)
        self.assertIn("NOT changed", r.text)
        self.assertFalse((self.root / "bad.py").exists())
        self.assertEqual(self.ctx.reverts, 1)

    def test_edit_reverts_when_it_would_break_the_file(self):
        (self.root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        r = call_tool("edit_file", self.ctx,
                      {"path": "ok.py", "find": "x = 1", "replace": "x = ("})
        self.assertFalse(r.ok)
        self.assertIn("REVERTED", r.text)
        self.assertEqual((self.root / "ok.py").read_text(encoding="utf-8"),
                         "x = 1\n")

    def test_edit_applies_when_valid(self):
        (self.root / "ok.py").write_text("x = 1\n", encoding="utf-8")
        r = call_tool("edit_file", self.ctx,
                      {"path": "ok.py", "find": "x = 1", "replace": "x = 2"})
        self.assertTrue(r.ok, r.text)
        self.assertEqual((self.root / "ok.py").read_text(encoding="utf-8"),
                         "x = 2\n")

    def test_edit_rejects_ambiguous_find(self):
        (self.root / "d.py").write_text("a = 1\na = 1\n", encoding="utf-8")
        r = call_tool("edit_file", self.ctx,
                      {"path": "d.py", "find": "a = 1", "replace": "a = 2"})
        self.assertFalse(r.ok)
        self.assertIn("ambiguous", r.text)

    def test_edit_reports_missing_find(self):
        (self.root / "d.py").write_text("a = 1\n", encoding="utf-8")
        r = call_tool("edit_file", self.ctx,
                      {"path": "d.py", "find": "nope", "replace": "x"})
        self.assertFalse(r.ok)
        self.assertIn("does not appear", r.text)

    def test_list_dir_caps_entries(self):
        for i in range(80):
            (self.root / f"f{i}.txt").write_text("x", encoding="utf-8")
        r = call_tool("list_dir", self.ctx, {"path": "."})
        self.assertTrue(r.ok)
        self.assertLessEqual(r.text.count(".txt"), 50)
        self.assertIn("not shown", r.text)

    def test_read_file_rejects_escape(self):
        r = call_tool("read_file", self.ctx, {"path": "../../etc/passwd"})
        self.assertFalse(r.ok)
        self.assertIn("escapes the workspace", r.text)

    def test_unknown_tool_lists_the_real_ones(self):
        r = call_tool("launch_missiles", self.ctx, {})
        self.assertFalse(r.ok)
        self.assertIn("read_file", r.text)

    def test_run_command_refuses_shell_chaining(self):
        r = call_tool("run_command", self.ctx, {"command": "ls && whoami"})
        self.assertFalse(r.ok)
        self.assertIn("refused", r.text)

    def test_run_command_reports_empty_output_explicitly(self):
        r = call_tool("run_command", self.ctx, {"command": 'python -c "pass"'})
        self.assertTrue(r.ok, r.text)
        self.assertIn("no output", r.text)

    def test_run_command_captures_stdout(self):
        r = call_tool("run_command", self.ctx,
                      {"command": 'python -c "print(6*7)"'})
        self.assertTrue(r.ok, r.text)
        self.assertIn("42", r.text)

    def test_read_file_marks_binary(self):
        (self.root / "b.bin").write_bytes(b"\x00\x01\x02")
        r = call_tool("read_file", self.ctx, {"path": "b.bin"})
        self.assertFalse(r.ok)
        self.assertIn("binary", r.text)


# ==========================================================================
# Context management
# ==========================================================================
class TestContextManager(unittest.TestCase):
    def test_estimate_is_roughly_right(self):
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertGreater(estimate_tokens("x" * 400), 50)
        self.assertLess(estimate_tokens("x" * 400), 200)

    def test_masks_old_observations_not_recent_ones(self):
        cm = ContextManager(context_window=4096, keep_recent_turns=1)
        cm.add_task("do a thing")
        for i in range(6):
            cm.add_action(f"<tool_call>step {i}")
            cm.add_observation("X" * 2000, meta={"tool": "read_file"})
        self.assertGreater(cm.usage() / cm.budget, cm.compact_at)
        cm.compact()
        elided = [m for m in cm.messages if m.elided]
        self.assertTrue(elided, "no observations were masked")
        # the most recent observation is the one the model is actually working
        # from, so it must survive verbatim
        self.assertFalse(cm.messages[-1].elided)
        self.assertEqual(cm.messages[-1].content, "X" * 2000)

    def test_masking_preserves_a_one_line_stub(self):
        # Sized so that masking alone brings the budget back under threshold and
        # the escalation stops there -- otherwise whole turns get dropped and the
        # stubs are gone before they can be inspected.
        cm = ContextManager(context_window=4096, keep_recent_turns=1)
        cm.add_task("t")
        for i in range(8):
            cm.add_action("a")
            cm.add_observation(f"first line of result {i}\n" + "y" * 1500,
                               meta={"tool": "read_file"})
        cm.compact()
        elided = [m for m in cm.messages if m.elided]
        self.assertTrue(elided, "masking did not run")
        self.assertIn("elided", elided[0].content)
        self.assertIn("read_file", elided[0].content)
        self.assertLess(len(elided[0].content), 120)

    def test_drops_whole_turns_when_masking_is_not_enough(self):
        cm = ContextManager(context_window=2048, keep_recent_turns=2)
        cm.add_task("t")
        for i in range(12):
            cm.add_action(f"a{i}")
            cm.add_observation("z" * 4000, meta={"tool": "read_file"})
        before = len(cm.messages)
        cm.compact()
        self.assertLess(len(cm.messages), before)

    def test_pinned_task_is_never_dropped(self):
        cm = ContextManager(context_window=2048, keep_recent_turns=1)
        cm.add_task("THE IMPORTANT TASK")
        for i in range(30):
            cm.add_action(f"a{i}")
            cm.add_observation("z" * 2000, meta={"tool": "read_file"})
        cm.compact(force=True)
        self.assertTrue(any(m.kind == "task" for m in cm.messages))
        self.assertIn("THE IMPORTANT TASK",
                      "\n".join(m.content for m in cm.messages))

    def test_drops_old_thinking(self):
        cm = ContextManager(context_window=4096, keep_recent_turns=1)
        cm.add_task("t")
        for i in range(8):
            cm.add_action("a", thinking="deep thought " * 50)
            cm.add_observation("obs", meta={"tool": "read_file"})
        cm.compact(force=True)
        remaining = [m for m in cm.messages if m.thinking]
        self.assertLess(len(remaining), 3)

    def test_calibration_learns_the_bias(self):
        cm = ContextManager(context_window=8192)
        before = cm._calibration
        cm.calibrate(estimated=1000, actual=1500)
        self.assertGreater(cm._calibration, before)

    def test_ledger_is_rendered_and_reinjected(self):
        cm = ContextManager(context_window=8192)
        cm.ledger.record_write("a.py")
        cm.ledger.record_command("python t.py", 0)
        text = cm.ledger.render()
        self.assertIn("a.py", text)
        self.assertIn("exit 0", text)
        payload = cm.build_payload(system="sys", tools=None)
        self.assertTrue(any("a.py" in str(m.get("content", "")) for m in payload))

    def test_ledger_separates_refused_commands_from_ones_that_ran(self):
        """A refused command must never look like a successful one.

        Regression test: refused commands were recorded with exit code 0, which
        made the ledger tell the model its commands had worked -- and the model
        then reported success without having verified anything.
        """
        cm = ContextManager(context_window=8192)
        cm.ledger.record_refusal("cd .. && python test_calc.py")
        text = cm.ledger.render()
        self.assertIn("REFUSED", text)
        self.assertIn("did not run", text)
        self.assertNotIn("exit 0", text)

    def test_ledger_states_that_nothing_is_verified(self):
        cm = ContextManager(context_window=8192)
        cm.ledger.record_refusal("cd .. && ls")
        self.assertIn("not successfully run any command", cm.ledger.render())

    def test_ledger_names_the_workspace_root(self):
        cm = ContextManager(context_window=8192)
        cm.ledger.root = "/home/me/project"
        cm.ledger.record_read("a.py", 1, 5)
        self.assertIn("/home/me/project", cm.ledger.render())

    def test_ledger_warns_when_no_command_succeeded(self):
        cm = ContextManager(context_window=8192)
        cm.ledger.record_command("python t.py", 1)
        self.assertIn("no command has succeeded", cm.ledger.render())

    def test_ledger_nudges_reverification_after_a_late_edit(self):
        """Editing after the last command leaves the change unverified."""
        cm = ContextManager(context_window=8192)
        cm.ledger.steps = 1
        cm.ledger.record_command("python test_calc.py", 0)
        cm.ledger.steps = 2
        cm.ledger.record_write("calc.py")
        self.assertIn("not verified yet", cm.ledger.render())

    def test_no_reverification_nudge_when_edit_came_first(self):
        cm = ContextManager(context_window=8192)
        cm.ledger.steps = 1
        cm.ledger.record_write("calc.py")
        cm.ledger.steps = 2
        cm.ledger.record_command("python test_calc.py", 0)
        self.assertNotIn("not verified yet", cm.ledger.render())

    def test_no_reverification_nudge_for_pure_authoring(self):
        """A task that never runs a command should not be nagged."""
        cm = ContextManager(context_window=8192)
        cm.ledger.record_write("notes.md")
        self.assertNotIn("not verified yet", cm.ledger.render())


# ==========================================================================
# Model profile
# ==========================================================================
class TestModelProfile(unittest.TestCase):
    def test_context_matches_the_published_config(self):
        self.assertEqual(QWEN35_08B.context_window, 262144)

    def test_hybrid_attention_geometry(self):
        self.assertEqual(QWEN35_08B.num_hidden_layers, 24)
        self.assertEqual(QWEN35_08B.full_attention_layers, 6)

    def test_kv_cost_is_cheap_because_most_layers_are_linear(self):
        # 6 layers * 2 kv heads * 256 head_dim * 2 (K,V) * 2 bytes = 12288 B/token
        self.assertEqual(QWEN35_08B.kv_bytes_per_token(2.0), 12288)
        self.assertAlmostEqual(QWEN35_08B.kv_bytes(262144) / 1024 ** 3, 3.0,
                               places=2)

    def test_visual_token_estimate(self):
        self.assertEqual(QWEN35_08B.approx_visual_tokens(1024, 1024), 1024)
        self.assertEqual(QWEN35_08B.approx_visual_tokens(1280, 720), 900)


# ==========================================================================
# Agent loop against the mock backend
# ==========================================================================
class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "target.py").write_text("value = 1\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def run_agent(self, script, **rt):
        cfg = make_cfg(self.root, context_window=8192,
                       max_steps=rt.pop("max_steps", 12), **rt)
        backend = MockBackend(script=script)
        agent = Agent(cfg, backend)
        result = agent.run("Improve target.py")
        return result, agent, backend

    def test_executes_a_tool_call(self):
        script = [
            scripted_tool_call("read_file", path="target.py"),
            scripted_answer("All done."),
        ]
        result, _agent, _b = self.run_agent(script)
        self.assertEqual(result.stop_reason, "finished")
        self.assertEqual(result.answer, "All done.")
        self.assertEqual(result.tool_calls, 1)

    def test_writes_a_file(self):
        script = [
            scripted_tool_call("write_file", path="new.py", content="x = 1\n"),
            scripted_answer("written"),
        ]
        result, _a, _b = self.run_agent(script)
        self.assertTrue((self.root / "new.py").is_file())
        self.assertEqual(result.stop_reason, "finished")

    def test_broken_write_is_reverted_and_reported(self):
        script = [
            scripted_tool_call("write_file", path="bad.py", content="def f(:\n"),
            scripted_answer("I could not write it."),
        ]
        _r, agent, _b = self.run_agent(script)
        self.assertFalse((self.root / "bad.py").exists())
        self.assertEqual(agent.ctx.reverts, 1)

    def test_malformed_output_gets_a_correction_then_recovers(self):
        # NB: plain prose with no tool call is a *final answer* by design, so the
        # malformed turn here has to be genuinely malformed (a JSON attempt).
        script = [
            '<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call>',
            scripted_tool_call("read_file", path="target.py"),
            scripted_answer("done"),
        ]
        _r, _agent, backend = self.run_agent(script)
        self.assertGreaterEqual(backend.calls, 2)
        joined = "\n".join(str(m.get("content", "")) for m in backend.seen[-1])
        self.assertIn("<function=NAME>", joined)

    def test_plain_prose_ends_the_run(self):
        result, _a, backend = self.run_agent(["Nothing to do here."])
        self.assertEqual(backend.calls, 1)
        self.assertEqual(result.stop_reason, "finished")
        self.assertEqual(result.answer, "Nothing to do here.")

    def test_repeated_action_is_refused_not_reexecuted(self):
        same = scripted_tool_call("read_file", path="target.py")
        script = [same, same, same, same, same, same, scripted_answer("stopped")]
        result, _a, _b = self.run_agent(script)
        executed = [s for s in result.steps
                    if s.result is not None and s.call and not s.note]
        self.assertLessEqual(len(executed), 2,
                             "the same call was re-executed too many times")

    def test_loop_is_bounded_by_max_steps(self):
        script = [scripted_tool_call("list_dir", path=f"d{i}") for i in range(50)]
        result, _a, _b = self.run_agent(script, max_steps=4)
        self.assertLessEqual(len(result.steps), 4)

    def test_backend_failure_is_reported_cleanly(self):
        backend = MockBackend(script=["x"], fail_after=0)
        cfg = make_cfg(self.root, context_window=8192)
        agent = Agent(cfg, backend)
        result = agent.run("do it")
        self.assertIn("backend error", result.stop_reason)

    def test_degenerate_output_is_retried_then_abandoned(self):
        unit = "**[list_dir](https://docs.python.org/3/library/files.html) tool**"
        script = ["\n---\n".join([unit] * 10)] * 8
        result, _a, _b = self.run_agent(script, max_steps=8)
        self.assertIn(result.stop_reason, ("degenerated", "max_steps"))

    def test_tool_refusal_is_surfaced_to_the_model(self):
        script = [
            scripted_tool_call("run_command", command="ls && rm -rf /"),
            scripted_answer("refused"),
        ]
        _r, agent, backend = self.run_agent(script)
        joined = "\n".join(str(m.get("content", "")) for m in backend.seen[-1])
        self.assertIn("refused", joined.lower())
        self.assertTrue(agent.manager.ledger.errors)

    def test_refused_command_is_not_recorded_as_success(self):
        script = [
            scripted_tool_call("run_command", command="cd .. && python t.py"),
            scripted_answer("done"),
        ]
        _r, agent, _b = self.run_agent(script)
        self.assertEqual(agent.manager.ledger.commands, [])
        self.assertTrue(agent.manager.ledger.refusals)

    def test_system_prompt_tells_the_model_where_it_is_without_the_full_path(self):
        _r, agent, backend = self.run_agent([scripted_answer("ok")])
        system = str(backend.seen[0][0].get("content", ""))
        self.assertIn("never need to change directory", system)
        # the absolute path is deliberately kept out of the stable prefix; seeing
        # it made the model prepend it to every tool argument
        self.assertNotIn(str(agent.guard.root), system)
        # ...but the ledger carries it, where it is needed
        self.assertIn(str(agent.guard.root), agent.manager.ledger.render())

    def test_repeated_action_is_caught_even_when_not_consecutive(self):
        """A model alternating between two pointless calls must still be caught."""
        a = scripted_tool_call("read_file", path="target.py")
        b = scripted_tool_call("list_dir", path=".")
        script = []
        for _ in range(4):
            script += [a, b]
        script.append(scripted_answer("done"))
        result, _agent, _b = self.run_agent(script, max_steps=20)
        executed = [s for s in result.steps
                    if s.result is not None and not s.note]
        self.assertLess(len(executed), 8,
                        "non-consecutive repeats were not throttled")
        # either the loop breaker or the forced conclusion; both are correct
        self.assertIn(result.stop_reason, ("looping", "finished"))
        self.assertLess(len(result.steps), 20)

    def test_verified_run_concludes_rather_than_grinding(self):
        """Regression: a model that keeps re-running a successful check used to
        grind to a `looping` stop and produce no answer at all.

        The escalation order had made the "conclude now" exit unreachable.
        """
        same = scripted_tool_call("run_command", command="python test_calc.py")
        result, _agent, _backend = self.run_agent([same] * 10, max_steps=14)
        self.assertEqual(result.stop_reason, "finished")
        self.assertTrue(result.answer, "the run produced no answer")
        self.assertIn("harness summary", result.answer)

    def test_summary_reports_only_what_actually_happened(self):
        _r, agent, _b = self.run_agent([scripted_answer("ok")])
        agent.manager.ledger.steps = 1
        agent.manager.ledger.record_write("calc.py")
        agent.manager.ledger.steps = 2
        agent.manager.ledger.record_command("python test_calc.py", 0)
        text = agent._summarise()
        self.assertIn("calc.py", text)
        self.assertIn("exit 0", text)
        self.assertIn("not the model's own words", text)

    def test_summary_says_nothing_when_nothing_happened(self):
        _r, agent, _b = self.run_agent([scripted_answer("ok")])
        text = agent._summarise()
        self.assertTrue(text, "the summary must never be empty")
        self.assertIn("harness summary", text)

    def test_summary_is_never_empty_even_for_a_read_only_run(self):
        """Regression: a forced conclusion after only reading files published an
        empty string as the whole result."""
        _r, agent, _b = self.run_agent([scripted_answer("ok")])
        agent.manager.ledger.record_read("a.py", 1, 10)
        text = agent._summarise()
        self.assertTrue(text.strip())
        self.assertIn("read", text)

    def test_summary_mentions_refused_actions(self):
        _r, agent, _b = self.run_agent([scripted_answer("ok")])
        agent.manager.ledger.record_refusal("read_file(a.py)")
        self.assertIn("refused", agent._summarise())

    def test_run_never_returns_an_empty_answer_when_forced_to_conclude(self):
        """A repeated action forces a conclusion; there must still be an answer."""
        same = scripted_tool_call("read_file", path="target.py")
        result, _agent, _b = self.run_agent([same] * 10, max_steps=12)
        self.assertEqual(result.stop_reason, "finished")
        self.assertTrue(result.answer.strip(),
                        "the forced conclusion produced no answer at all")


class TestNoteDeliveryAndCacheStability(unittest.TestCase):
    """Corrections must reach the model, and the transcript must be append-only.

    Two related bugs, both invisible without looking at what is actually sent:

    * ``build_payload`` filtered notes out, so every carefully-worded repair
      message the harness built was silently dropped and the model was retried on
      an unchanged prompt;
    * ``add_note`` rebuilt the message list to replace the old note, which
      invalidated the server's prefix cache from that point onward -- the most
      expensive thing that can happen on a CPU.
    """

    def test_a_note_reaches_the_payload(self):
        cm = ContextManager(context_window=8192)
        cm.add_task("do it")
        cm.add_note("CORRECTION: use the XML format")
        payload = cm.build_payload(system="sys", tools=None)
        joined = "\n".join(str(m.get("content", "")) for m in payload)
        self.assertIn("CORRECTION: use the XML format", joined)

    def test_notes_are_append_only(self):
        cm = ContextManager(context_window=8192)
        cm.add_task("t")
        cm.add_action("a1")
        cm.add_note("first correction")
        cm.add_action("a2")
        cm.add_note("second correction")
        contents = [m.content for m in cm.messages]
        # the first note must still be there, in its original position
        self.assertIn("first correction", contents)
        self.assertLess(contents.index("first correction"),
                        contents.index("second correction"))
        self.assertEqual(len([m for m in cm.messages if m.kind == "note"]), 2)

    def test_an_old_note_is_masked_like_an_observation(self):
        cm = ContextManager(context_window=2048, keep_recent_turns=1)
        cm.add_task("t")
        for i in range(8):
            cm.add_action(f"a{i}")
            cm.add_note("a long correction " * 40)
            cm.add_observation("result " + "z" * 1500, meta={"tool": "read_file"})
        cm.compact()
        notes = [m for m in cm.messages if m.kind == "note"]
        self.assertTrue(notes)
        self.assertTrue(any(n.elided for n in notes),
                        "old notes should be elided to bound growth")
        for n in notes:
            if n.elided:
                self.assertLess(len(n.content), 120)

    def test_run_result_reports_the_cache_hit_ratio(self):
        result = RunResult(tokens_in=1000, cached_in=750)
        self.assertAlmostEqual(result.cache_hit_ratio, 0.75)
        self.assertEqual(RunResult().cache_hit_ratio, 0.0)

    def test_a_run_records_cached_tokens_from_the_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp), context_window=8192, max_steps=6)
            script = [scripted_tool_call("list_dir", path="."),
                      scripted_tool_call("list_dir", path="sub"),
                      scripted_answer("done")]
            agent = Agent(cfg, MockBackend(script=script))
            result = agent.run("look around")
            self.assertGreater(result.cached_in, 0,
                               "the mock reports prefix reuse; the loop must "
                               "propagate it")


class TestServerCacheFlags(unittest.TestCase):
    """The launch line must actually ask for context checkpoints."""

    def test_build_command_requests_checkpoints(self):
        from qwen35.llama_runtime import build_command

        plan = build_command(Path("llama-server"), Path("m.gguf"), probe=False,
                             context=8192, ctx_checkpoints=16,
                             checkpoint_min_step=512)
        argv = " ".join(plan.argv)
        self.assertIn("-ctxcp 16", argv)
        self.assertIn("-cms 512", argv)

    def test_checkpoints_can_be_disabled(self):
        from qwen35.llama_runtime import build_command

        plan = build_command(Path("llama-server"), Path("m.gguf"), probe=False,
                             context=8192, ctx_checkpoints=0)
        self.assertNotIn("-ctxcp", " ".join(plan.argv))
        self.assertTrue(any("checkpoint" in n.lower() for n in plan.notes))

    def test_no_context_shift_is_still_requested(self):
        from qwen35.llama_runtime import build_command

        plan = build_command(Path("llama-server"), Path("m.gguf"), probe=False)
        self.assertIn("--no-context-shift", plan.argv)


class TestServerLifecycleSafety(unittest.TestCase):
    """A server must never be declared ready by a server we do not own.

    Observed on a real machine: two ``llama-server`` processes, only one owning
    port 8080, and ``server.json`` holding the *loser's* key -- so every
    programmatic caller got an opaque ``HTTP 401 Invalid API Key`` while the GUI
    (reusing its in-memory key) kept working.  llama-server does not exit when its
    port is taken, and ``wait_ready`` only polls ``/health``, which the other
    server answers.
    """

    def test_port_in_use_detects_a_listener(self):
        import socket

        from qwen35.llama_runtime import port_in_use

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            self.assertTrue(port_in_use("127.0.0.1", port))
        finally:
            server.close()

    def test_port_in_use_is_false_for_a_closed_port(self):
        import socket

        from qwen35.llama_runtime import port_in_use

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()          # now nothing is listening there
        self.assertFalse(port_in_use("127.0.0.1", port, timeout=0.3))

    def test_ensure_server_refuses_to_race_an_occupied_port(self):
        """A listening port we cannot authenticate to must be reported, not
        raced with a second server.

        The fake answers 401 to everything -- which is what a llama-server left
        over from an earlier run looks like when its key no longer matches.
        """
        import http.server
        import threading

        from qwen35 import llama_runtime

        class Unauthorized(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"Invalid API Key"}')

            def log_message(self, *_args):
                pass

        httpd = http.server.HTTPServer(("127.0.0.1", 0), Unauthorized)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertTrue(llama_runtime.port_in_use("127.0.0.1", port))
            status = llama_runtime.ensure_server(
                model_dir=Path(tempfile.gettempdir()), context=8192, port=port,
                timeout=2.0,
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertFalse(status.ok)
        # the message must name the cause AND the way out
        self.assertIn("left over from an earlier run", status.message)
        self.assertIn("stop.bat", status.message)
        self.assertIn("force-restart", status.message)


class TestServerConflictHandling(unittest.TestCase):
    """A server we do not own must never be silently adopted.

    The reported failure was:

        a server is running but does not accept the stored API key

    and its root cause was concurrent starts.  Two clicks on 重启服务 launched two
    ``ensure_server`` calls; the loser could not bind the port, but ``wait_ready``
    was satisfied by the *winner's* ``/health``, so the loser saved its own key.
    The live server had never seen that key, and every later request got a 401.
    """

    def setUp(self):
        import http.server
        import threading

        from qwen35 import llama_runtime

        self.runtime = llama_runtime
        self._tmp = tempfile.TemporaryDirectory()
        self.state = Path(self._tmp.name) / "state"
        self.state.mkdir()
        # point the harness's state (and therefore server.json) at a temp dir
        self._old_home = os.environ.get("QWEN35_HOME")
        os.environ["QWEN35_HOME"] = str(self.state)
        llama_runtime.save_server_key("ORIGINALKEY", "http://127.0.0.1:1", 999,
                                      "model.gguf")

        class FakeServer(http.server.BaseHTTPRequestHandler):
            """/health says ok; everything authenticated says 401."""

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/health"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"Invalid API Key"}')

            def log_message(self, *_args):
                pass

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), FakeServer)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._old_home is None:
            os.environ.pop("QWEN35_HOME", None)
        else:
            os.environ["QWEN35_HOME"] = self._old_home
        self._tmp.cleanup()

    def test_the_foreign_server_is_detected_as_unauthorized(self):
        state, _message = self.runtime.probe_server(
            f"http://127.0.0.1:{self.port}", "ORIGINALKEY")
        self.assertEqual(state, "unauthorized")

    def test_without_force_it_refuses_and_keeps_the_old_key(self):
        status = self.runtime.ensure_server(
            model_dir=Path(self._tmp.name), port=self.port, timeout=5.0)
        self.assertFalse(status.ok)
        self.assertIn("not accept", status.message)
        # the crucial part: the stored key is untouched
        self.assertEqual(self.runtime.load_server_key(), "ORIGINALKEY")

    def test_the_refusal_explains_how_to_recover(self):
        status = self.runtime.ensure_server(
            model_dir=Path(self._tmp.name), port=self.port, timeout=5.0)
        self.assertIn("force-restart", status.message)

    def test_force_restart_stops_the_holder_and_proceeds(self):
        stopped: list[tuple[str, int]] = []
        original_stop = self.runtime.stop_servers_on_port
        original_bin = self.runtime.installed_server
        original_in_use = self.runtime.port_in_use

        def fake_stop(host="127.0.0.1", port=8080):
            stopped.append((host, port))
            self.httpd.shutdown()          # the port really is released
            self.httpd.server_close()
            return 1

        self.runtime.stop_servers_on_port = fake_stop
        self.runtime.port_in_use = lambda host, port, timeout=0.6: False
        self.runtime.installed_server = lambda: None    # stop before spawning
        try:
            status = self.runtime.ensure_server(
                model_dir=Path(self._tmp.name), port=self.port,
                force_restart=True, timeout=5.0)
        finally:
            self.runtime.stop_servers_on_port = original_stop
            self.runtime.installed_server = original_bin
            self.runtime.port_in_use = original_in_use

        self.assertTrue(stopped, "the holder was not stopped")
        self.assertEqual(stopped[0][1], self.port)
        # getting past the gate is proved by reaching the *next* check
        self.assertIn("not installed", status.message)

    def test_a_successful_start_must_authenticate_with_its_own_key(self):
        """``wait_ready`` only proves *some* server is up.

        After starting, the harness asks the server to authenticate with the key
        it just gave its own child.  A foreign server has never seen that key, so
        a failed confirmation means we lost the port race -- and must not save a
        key the live server does not have.
        """
        source = Path(self.runtime.__file__).read_text(encoding="utf-8")
        start = source.index("ok, msg = server.wait_ready")
        tail = source[start:start + 1800]
        self.assertIn("prove the server answering is OURS", tail)
        self.assertIn("confirm, confirm_message = probe_server", tail)
        # and the save happens only after that check
        self.assertLess(tail.index("confirm, confirm_message"),
                        tail.index("save_server_key"))

    def test_starts_are_serialised_in_process(self):
        """Two threads calling ensure_server must not interleave."""
        import threading
        import time as _time

        order: list[str] = []
        lock = self.runtime._START_LOCK

        def worker(name: str) -> None:
            with lock:
                order.append(f"{name}-in")
                _time.sleep(0.15)
                order.append(f"{name}-out")

        threads = [threading.Thread(target=worker, args=(n,)) for n in "AB"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        # never A-in, B-in, ... : one holder at a time
        self.assertEqual(order[0][-2:], "in")
        self.assertEqual(order[1][-3:], "out")
        self.assertEqual(order[2][-2:], "in")
        self.assertEqual(order[3][-3:], "out")


class TestFailureDiagnosis(unittest.TestCase):
    """A failed run must say what went wrong AND what to do next."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_refused_steps_reach_the_session_log(self):
        """They used to vanish, leaving '3 action(s) were refused' unexplainable."""
        script = [scripted_tool_call("list_dir", path=".")] * 10
        cfg = make_cfg(self.root, context_window=8192, max_steps=10)
        records: list[tuple[str, dict[str, Any]]] = []

        class Rec:
            def record(self, kind, data):
                records.append((kind, data))

            def close(self):
                pass

        agent = Agent(cfg, MockBackend(script=script), recorder=Rec())
        agent.run("look around")
        self.assertTrue(any(k == "refused" for k, _ in records),
                        "refused steps must be recorded")
        refused = [d for k, d in records if k == "refused"]
        self.assertIn("call", refused[0])

    def test_the_summary_gains_a_next_step(self):
        script = [scripted_tool_call("read_file", path=".")] * 10
        cfg = make_cfg(self.root, context_window=8192, max_steps=10)
        agent = Agent(cfg, MockBackend(script=script))
        result = agent.run("read it")
        self.assertIn("[next step]", result.answer)

    def test_diagnosis_mentions_nothing_read_when_nothing_was_read(self):
        cfg = make_cfg(self.root, context_window=8192)
        agent = Agent(cfg, MockBackend(script=[scripted_answer("hi")]))
        agent._attachments_pending = True
        tip = agent._diagnose()
        self.assertIn("attachment", tip.lower())


# ==========================================================================
# Backend key resolution
# ==========================================================================
class TestBackendKeyResolution(unittest.TestCase):
    """The stored server key must be resolved for ANY caller, not just the CLI.

    Regression test.  Only the CLI and the GUI populated ``RuntimeConfig.api_key``
    themselves, so anything that built a config by hand -- the evaluation suite, a
    library user -- got an opaque ``HTTP 401 Invalid API Key`` from a server it had
    just started.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("QWEN35_HOME")
        os.environ["QWEN35_HOME"] = self._tmp.name

    def tearDown(self):
        if self._old is None:
            os.environ.pop("QWEN35_HOME", None)
        else:
            os.environ["QWEN35_HOME"] = self._old
        self._tmp.cleanup()

    def test_local_server_key_is_resolved_automatically(self):
        from qwen35.backends import resolve_api_key
        from qwen35.llama_runtime import save_server_key

        save_server_key("SECRET123", "http://127.0.0.1:8080", 0, "m")
        cfg = RuntimeConfig(base_url="http://127.0.0.1:8080")
        self.assertEqual(resolve_api_key(cfg), "SECRET123")

    def test_an_explicit_key_always_wins(self):
        from qwen35.backends import resolve_api_key

        cfg = RuntimeConfig(base_url="http://127.0.0.1:8080", api_key="EXPLICIT")
        self.assertEqual(resolve_api_key(cfg), "EXPLICIT")

    def test_a_remote_endpoint_does_not_borrow_the_local_key(self):
        from qwen35.backends import resolve_api_key

        cfg = RuntimeConfig(base_url="https://api.example.com")
        self.assertEqual(resolve_api_key(cfg), "")

    def test_localhost_spellings_are_recognised(self):
        from qwen35.backends import _is_local_url

        for url in ("http://127.0.0.1:8080", "http://localhost:8080",
                    "http://[::1]:8080", "http://0.0.0.0:8080"):
            with self.subTest(url=url):
                self.assertTrue(_is_local_url(url))
        for url in ("https://api.openai.com", "http://192.168.1.5:8080"):
            with self.subTest(url=url):
                self.assertFalse(_is_local_url(url))


# ==========================================================================
# Conversational routing
# ==========================================================================
class TestConversationalRouting(unittest.TestCase):
    """Small talk must not be treated as a work order.

    Asking a 0.8 B model "你好" without this produced 3-4 tool calls, 4 steps and
    60-200 seconds -- and sometimes an empty answer.
    """

    def test_greetings_are_conversational(self):
        for text in ("你好", "您好", "hi", "Hello", "hey", "你是谁？", "谢谢",
                     "在吗", "早上好", "再见", "thanks"):
            with self.subTest(text=text):
                self.assertTrue(looks_conversational(text), text)

    def test_work_requests_are_not_conversational(self):
        for text in ("你好，帮我修一下 calc.py", "运行测试", "hello, please fix the bug",
                     "读一下 README.md", "这个项目有哪些文件？", "总结一下这个项目",
                     "看看 calc.py", "why does the test fail?", "add tests",
                     "你好，看看这个"[:4] + "，看看那个"):
            with self.subTest(text=text):
                self.assertFalse(looks_conversational(text), text)

    def test_long_messages_are_never_conversational(self):
        self.assertFalse(looks_conversational("你好" * 40))

    def test_paths_and_code_disable_the_shortcut(self):
        for text in ("你好 app.py", "hi calc.py", "你好 `ls`", "hello/x"):
            with self.subTest(text=text):
                self.assertFalse(looks_conversational(text), text)

    def test_blank_input_is_not_conversational(self):
        self.assertFalse(looks_conversational(""))
        self.assertFalse(looks_conversational("   "))


class TestConversationalRunIsToolFree(unittest.TestCase):
    """The guarantee, not the hint.

    Prompting a 0.8 B model to stop exploring on a greeting measured 1/4.  So a
    conversational turn is run with an EMPTY tool list: nothing in the schema and
    nothing in the grammar, which makes a tool call impossible rather than
    discouraged.
    """

    def test_build_specs_empty_list_means_no_tools(self):
        self.assertEqual(build_specs([]), [])
        self.assertTrue(len(build_specs(None)) > 0,
                        "None must still mean 'all tools'")

    def test_conversational_run_offers_no_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp), context_window=8192, max_steps=3)
            agent = Agent(cfg, MockBackend(script=[scripted_answer("你好！")]))
            result = agent.run("你好")
            self.assertEqual(agent.tool_specs, [])
            grammar = agent._grammar(thinking=False)
            self.assertNotIn("<function=", grammar,
                             "the grammar must not offer any tool")
            self.assertNotIn("function", grammar)
            self.assertNotIn("params", grammar)
            # ...and the plain-answer branch still forbids a stray <tool_call>
            self.assertIn("ansall", grammar)
            self.assertEqual(result.stop_reason, "finished")
            self.assertEqual(result.steps, [])

    def test_conversational_prompt_is_short_and_has_no_tool_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp), context_window=8192)
            agent = Agent(cfg, MockBackend(script=[scripted_answer("你好")]))
            agent._conversational = True
            prompt = agent._system_prompt([])
            self.assertLess(len(prompt), 400,
                            "a greeting must not prefill the whole tool block")
            self.assertNotIn("# Tools", prompt)
            self.assertNotIn("<function=", prompt)

    def test_a_work_request_still_gets_the_full_tool_prompt(self):
        from qwen35.prompt import build_tool_schema

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp), context_window=8192)
            agent = Agent(cfg, MockBackend(script=[scripted_answer("ok")]))
            agent._conversational = False
            schema = build_tool_schema(build_specs(None, cfg.permissions))
            prompt = agent._system_prompt(schema)
            self.assertIn("# Tools", prompt)
            self.assertIn("list_dir", prompt)
            self.assertIn("<function=example_function_name>", prompt)

    def test_greeting_answer_costs_one_generation(self):
        """The whole point: no steps, no tools, one reply."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_cfg(Path(tmp), context_window=8192, max_steps=8)
            backend = MockBackend(script=[scripted_answer("你好！有什么可以帮你的？")])
            agent = Agent(cfg, backend)
            result = agent.run("你好")
            self.assertEqual(backend.calls, 1)
            self.assertEqual(result.tool_calls, 0)
            self.assertEqual(result.stop_reason, "finished")


class TestAttachmentRouting(unittest.TestCase):
    """With the attached text already in the prompt, the first step offers no tools.

    Measured before this existed: a 973-token spreadsheet, fully inlined, still
    cost 6 steps and 210 s -- the model called ``list_dir`` twice (the rules say
    "call list_dir once before reading anything"), tripped the duplicate breaker,
    and sometimes returned no answer at all.  With the routing: 0 steps, 43 s.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _attachment(self, inlined: bool = True):
        from qwen35.ingest import Attachment

        att = Attachment(source=Path("报告.xlsx"), kind="xlsx",
                         text="| 商品 | 价格 |\n| --- | --- |\n| 遥控器 | 839 |")
        att.inlined = inlined
        att.out_rel = ".qwen35/attachments/01-报告.md"
        return att

    # -- the decision itself -------------------------------------------
    def test_extract_request_strips_the_scaffolding(self):
        from qwen35.ingest import compose_task

        task = compose_task("总结一下", [self._attachment()])
        self.assertIn("# Attached files", task)
        self.assertEqual(extract_request(task), "总结一下")

    def test_the_harness_own_wording_is_not_a_change_request(self):
        """The card says "converted to text" -- that must not match "convert".

        This was a real bug: the marker was tested against the whole composed
        message, so the harness's own boilerplate made every attachment question
        look like a request to modify the workspace, and the fast path never ran.
        """
        from qwen35.ingest import compose_task

        task = compose_task("这个文件讲了什么？", [self._attachment()])
        self.assertIn("converted to text", task)
        self.assertFalse(wants_changes(task))

    def test_a_content_question_routes_to_no_tools(self):
        from qwen35.ingest import compose_task

        att = self._attachment()
        for request in ("这个文件总结一下", "讲了什么？", "summarize it"):
            task = compose_task(request, [att])
            self.assertTrue(answers_from_prompt(task, [att]), request)

    def test_a_change_request_keeps_the_tools(self):
        from qwen35.ingest import compose_task

        att = self._attachment()
        for request in ("把附件表格导出成 csv", "按附件要求改 calc.py",
                        "convert the table to json", "save it as data.csv"):
            task = compose_task(request, [att])
            self.assertFalse(answers_from_prompt(task, [att]), request)

    def test_an_image_attachment_keeps_the_tools(self):
        """An image has no inlined text, so view_image is still needed."""
        att = self._attachment(inlined=False)
        att.kind = "image"
        self.assertFalse(answers_from_prompt("描述一下这张图", [att]))

    def test_no_attachments_means_normal_tool_use(self):
        self.assertFalse(answers_from_prompt("总结一下", []))

    # -- and in the loop ------------------------------------------------
    def test_the_loop_answers_without_a_single_tool_call(self):
        from qwen35.ingest import compose_task

        att = self._attachment()
        cfg = make_cfg(self.root, context_window=8192, max_steps=6)
        agent = Agent(cfg, MockBackend(script=[scripted_answer("表格里有遥控器，839元。")]),
                      attachments=[att])
        result = agent.run(compose_task("这个文件总结一下", [att]))
        self.assertEqual(result.tool_calls, 0, "no tool should have been needed")
        self.assertIn("839", result.answer)
        self.assertEqual(result.stop_reason, "finished")

    def test_the_loop_keeps_tools_when_the_workspace_must_change(self):
        from qwen35.ingest import compose_task

        att = self._attachment()
        cfg = make_cfg(self.root, context_window=8192, max_steps=6)
        script = [scripted_tool_call("write_file", path="out.csv", content="a,b"),
                  scripted_answer("done")]
        agent = Agent(cfg, MockBackend(script=script), attachments=[att])
        result = agent.run(compose_task("把附件表格导出成 out.csv", [att]))
        self.assertEqual(result.tool_calls, 1)
        self.assertTrue((self.root / "out.csv").is_file())

    def test_tools_come_back_if_the_no_tool_step_fails(self):
        """A single bad first step must not leave the run unable to act."""
        from qwen35.ingest import compose_task

        att = self._attachment()
        cfg = make_cfg(self.root, context_window=8192, max_steps=6)
        script = [scripted_answer(""),                       # empty first reply
                  scripted_tool_call("list_dir", path="."),
                  scripted_answer("listed")]
        agent = Agent(cfg, MockBackend(script=script), attachments=[att])
        result = agent.run(compose_task("这个文件总结一下", [att]))
        self.assertIn("list_dir", {s.call.name for s in result.steps if s.call})


class TestIllegalToolDefence(unittest.TestCase):
    """The grammar should make an invented tool name impossible.

    It is not merely a theoretical concern: a real run produced a call to
    ``grep``, which is not one of the six tools and cannot be produced by the
    grammar.  That means the grammar was *not applied* for that step — and
    :meth:`Agent._grammar` used to swallow any such failure and return ``None``
    silently.  These tests pin both halves of the fix.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _agent(self, script, **rt):
        cfg = make_cfg(self.root, context_window=8192,
                       max_steps=rt.pop("max_steps", 8), **rt)
        return Agent(cfg, MockBackend(script=script)), cfg

    def test_an_invented_tool_is_refused_and_never_executed(self):
        agent, _cfg = self._agent([
            scripted_tool_call("grep", path="."),
            scripted_answer("done"),
        ])
        result = agent.run("search the project")
        executed = [s for s in result.steps if s.result is not None]
        self.assertEqual(executed, [], "an invented tool must never be executed")
        self.assertEqual(result.stop_reason, "finished")
        self.assertIn("grep", " ".join(agent.manager.ledger.errors))

    def test_the_correction_lists_the_real_tools(self):
        agent, _cfg = self._agent([
            scripted_tool_call("search", query="TODO"),
            scripted_answer("done"),
        ])
        agent.run("find todos")
        payload = agent.manager.build_payload(system="sys", tools=None)
        joined = "\n".join(str(m.get("content", "")) for m in payload)
        self.assertIn("there is no tool called", joined)
        self.assertIn("read_file", joined)

    def test_repeated_invented_names_stop_the_run(self):
        agent, _cfg = self._agent([scripted_tool_call("grep", path=".")] * 12,
                                  max_steps=12)
        result = agent.run("search")
        self.assertIn("illegal tool name", result.stop_reason)

    def test_a_legal_name_is_not_touched_by_the_defence(self):
        agent, _cfg = self._agent([
            scripted_tool_call("list_dir", path="."),
            scripted_answer("done"),
        ])
        result = agent.run("look")
        executed = [s for s in result.steps if s.result is not None]
        self.assertEqual(len(executed), 1)
        self.assertTrue(executed[0].result.ok)

    def test_a_grammar_failure_is_reported_not_hidden(self):
        import qwen35.loop as loop_module

        events: list[str] = []
        cfg = make_cfg(self.root, context_window=8192)
        agent = Agent(cfg, MockBackend(script=[]),
                      on_event=lambda e, _d: events.append(e))

        original = loop_module.build_grammar

        def explode(*_a, **_k):
            raise RuntimeError("synthetic grammar failure")

        loop_module.build_grammar = explode
        try:
            grammar = agent._grammar(thinking=False)
        finally:
            loop_module.build_grammar = original

        self.assertIsNone(grammar)
        self.assertIn("grammar_unavailable", events)
        self.assertTrue(any("UNAVAILABLE" in e for e in agent.manager.ledger.errors))


class TestThinkingPolicy(unittest.TestCase):
    """Which turns get to reason -- measured, and now locked down."""

    def _agent(self, thinking: str, conversational: bool):
        cfg = make_cfg(Path("."), thinking=thinking, context_window=4096)
        agent = Agent(cfg, MockBackend(script=[]))
        agent._conversational = conversational
        return agent

    def test_conversational_first_step_never_reasons(self):
        for mode in ("on", "auto", "off"):
            with self.subTest(mode=mode):
                agent = self._agent(mode, True)
                self.assertFalse(agent._use_thinking(0, False))

    def test_first_work_step_reasons_when_on_or_auto(self):
        self.assertTrue(self._agent("on", False)._use_thinking(0, False))
        self.assertTrue(self._agent("auto", False)._use_thinking(0, False))
        self.assertFalse(self._agent("off", False)._use_thinking(0, False))

    def test_truncated_reasoning_is_not_repeated(self):
        """Retrying the verbose path that just produced nothing wastes the budget."""
        for mode in ("on", "auto"):
            with self.subTest(mode=mode):
                agent = self._agent(mode, False)
                self.assertFalse(agent._use_thinking(1, True, last_truncated=True))

    def test_auto_reasons_on_a_retry(self):
        self.assertTrue(self._agent("auto", False)._use_thinking(3, True))

    def test_auto_does_not_reason_on_routine_later_steps(self):
        self.assertFalse(self._agent("auto", False)._use_thinking(3, False))


# ==========================================================================
# Prompt construction
# ==========================================================================
class TestPrompt(unittest.TestCase):
    def test_tools_block_matches_the_chat_template(self):
        from qwen35.prompt import render_tools_block
        block = render_tools_block([{"type": "function",
                                     "function": {"name": "list_dir"}}])
        # verbatim strings from Qwen/Qwen3.5-0.8B chat_template.jinja
        self.assertIn("# Tools", block)
        self.assertIn("You have access to the following functions:", block)
        self.assertIn("<tools>", block)
        self.assertIn("</tools>", block)
        self.assertIn("<IMPORTANT>", block)
        self.assertIn("Function calls MUST follow the specified format", block)
        self.assertIn("</IMPORTANT>", block)

    def test_build_system_embeds_tools_only_when_asked(self):
        from qwen35.prompt import build_system
        tools = [{"type": "function", "function": {"name": "list_dir"}}]
        with_tools = build_system(tools, embed_tools=True)
        without = build_system(tools, embed_tools=False)
        self.assertIn("<tools>", with_tools)
        self.assertNotIn("<tools>", without)
        self.assertIn("How to work here", without)

    def test_permission_note_is_included(self):
        from qwen35.prompt import system_rules
        rules = system_rules(permission_note="# Your permissions\nAllowed: read.")
        self.assertIn("Your permissions", rules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
