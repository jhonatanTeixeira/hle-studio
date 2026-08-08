"""The (isa=*, output_language="rust") plugin, built with Cargo. This is the
FIRST concrete plugin, extracted near-verbatim from the regex/anchor logic
that was proven live in jhonatanTeixeira/portal_to_another_world's
tools/agent_study_all.py (the `_fn_name_in`/`_extract_fn_body`/
`register_lookup_entry` functions) - not invented fresh, so it inherits that
code's real fix history (e.g. the `(?:pub\\s+)?` prefix in the extraction
regex exists specifically because an earlier version produced "pub pub fn"
when replacing a function - see the git history of the originating project
if this ever needs revisiting).
"""
from __future__ import annotations

import re
import subprocess

from hle_studio.config import TargetConfig
from hle_studio.plugins.base import BuildRunner, DispatchEditor, FunctionExtractor


class RustMatchTableDispatchEditor(DispatchEditor):
    def __init__(self, config: TargetConfig):
        self.config = config

    def address_already_registered(self, table_content: str, addr_hex: str) -> bool:
        addr = addr_hex.upper().lstrip("0x").lstrip("0X")
        addr = addr.rjust(8, "0")
        # Both sides uppercased consistently - a real bug in the originating
        # project compared a hardcoded lowercase "0x" against an all-upper
        # string and NEVER matched, silently disabling this exact check and
        # letting the same address get registered 3 times. Fixed here from
        # the start.
        return f"0X{addr}" in table_content.upper()

    def list_registered_addresses(self, table_content: str) -> set[str]:
        # A naive `0x(\w+)\s*=>` regex only matches the LAST address right
        # before `=>`, silently missing every address before a `|` in a
        # consolidated multi-pattern arm (`0xA | 0xB => Some(...)`) - a real
        # bug found live 2026-08-07 in the originating project when several
        # near-duplicate stub functions got consolidated into one shared
        # implementation. Expand each `|`-separated group explicitly.
        addrs = set()
        for group in re.findall(r"((?:0x[0-9A-Fa-f]{8}\s*\|\s*)*0x[0-9A-Fa-f]{8})\s*=>\s*Some\(", table_content):
            addrs.update(a.upper() for a in re.findall(r"0x([0-9A-Fa-f]{8})", group))
        return addrs

    def insert_entry(self, table_content: str, addr_hex: str, layer: str, fn_name: str) -> str:
        addr = addr_hex.upper().lstrip("0x").lstrip("0X").rjust(8, "0")
        if self.address_already_registered(table_content, addr):
            raise ValueError(f"0x{addr} already registered - caller must check first")
        anchor = self.config.dispatch_anchor
        idx = table_content.find(anchor)
        if idx == -1:
            raise ValueError(f"dispatch anchor {anchor!r} not found in table content")
        # Insert on the line right AFTER the anchor ends. `anchor` must cover
        # everything through the opening of the actual match block (e.g.
        # "pub fn lookup(addr: u32) -> Option<NativeFn> {\n    match addr {")
        # - NOT just the function signature's own brace, which is itself a
        # "{" and would make a naive "find the next {" match itself.
        insert_at = table_content.find("\n", idx + len(anchor)) + 1
        entry = self.config.dispatch_entry_template.format(
            addr=addr, fn_ref=f"{layer}::{fn_name}"
        )
        new_line = f"        {entry}\n"
        return table_content[:insert_at] + new_line + table_content[insert_at:]


class RustRegexFunctionExtractor(FunctionExtractor):
    def fn_name_in(self, source: str, name: str) -> bool:
        return re.search(rf"\bfn\s+{re.escape(name)}\s*\(", source) is not None

    def extract_fn_name(self, code_snippet: str) -> str | None:
        m = re.search(r"\bfn\s+(\w+)\s*\(", code_snippet)
        return m.group(1) if m else None

    def extract_fn_body(self, source: str, name: str) -> tuple[int, int] | None:
        # (?:pub\s+)? is INSIDE the match on purpose - the match start must
        # land on "pub" when present, or a replace-in-place re-inserts a new
        # "pub fn" right after the old "pub" left behind, producing
        # "pub pub fn ..." (a real bug hit once in the originating project).
        m = re.search(rf"(?:pub\s+)?\bfn\s+{re.escape(name)}\s*\(", source)
        if not m:
            return None
        start = m.start()
        lines_before = source[:start].splitlines(keepends=True)
        i = len(lines_before) - 1
        while i >= 0 and (
            lines_before[i].strip().startswith("///")
            or lines_before[i].strip().startswith("#[")
            or lines_before[i].strip() == ""
        ):
            if lines_before[i].strip() == "" and i > 0 and not (
                lines_before[i - 1].strip().startswith("///")
                or lines_before[i - 1].strip().startswith("#[")
            ):
                break
            i -= 1
        start = sum(len(lines_before[j]) for j in range(i + 1))
        brace_start = source.find("{", m.end())
        if brace_start == -1:
            return None
        depth = 0
        j = brace_start
        while j < len(source):
            if source[j] == "{":
                depth += 1
            elif source[j] == "}":
                depth -= 1
                if depth == 0:
                    return (start, j + 1)
            j += 1
        return None


class CargoBuildRunner(BuildRunner):
    def __init__(self, config: TargetConfig, timeout: int = 180):
        self.config = config
        self.timeout = timeout

    def build(self) -> tuple[bool, str]:
        out = subprocess.run(
            self.config.build_cmd, capture_output=True, text=True,
            timeout=self.timeout, cwd=self.config.build_cwd,
        )
        combined = (out.stdout or "") + (out.stderr or "")
        return out.returncode == 0, "\n".join(combined.splitlines()[-80:])

    def test(self) -> tuple[bool, str]:
        out = subprocess.run(
            self.config.test_cmd, capture_output=True, text=True,
            timeout=self.timeout, cwd=self.config.build_cwd,
        )
        combined = (out.stdout or "") + (out.stderr or "")
        return out.returncode == 0, "\n".join(combined.splitlines()[-100:])
