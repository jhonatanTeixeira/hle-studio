"""Abstract interfaces every (ISA, output-language) plugin must implement.

The core graph/agent code in hle_studio never touches regex or subprocess
argv directly for anything language-specific - it only calls through these
three interfaces. This is the concrete mechanism behind the "35-45% of the
original pipeline needs real rewriting per target, 55-65% is reusable"
finding: everything on THIS side of the interface is the part that changes
per (isa, language); everything calling INTO it (the pydantic-graph nodes)
never changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class DispatchEditor(ABC):
    """Reads/writes the target project's native-function dispatch table
    (e.g. a Rust `match` arm, a C `switch` case, a JSON/TOML table)."""

    @abstractmethod
    def address_already_registered(self, table_content: str, addr_hex: str) -> bool:
        """True if `addr_hex` already has an entry in `table_content`."""

    @abstractmethod
    def insert_entry(self, table_content: str, addr_hex: str, layer: str, fn_name: str) -> str:
        """Returns `table_content` with one new entry inserted. Must be
        idempotent-safe to call only once address_already_registered() is
        False - implementations should still refuse (raise) rather than
        silently duplicate if called anyway, as defense in depth."""

    @abstractmethod
    def list_registered_addresses(self, table_content: str) -> set[str]:
        """Every address (uppercase hex, no 0x prefix) already registered
        in `table_content` - the driver uses this to compute the pending
        queue, deliberately never leaving that choice to the model."""


class FunctionExtractor(ABC):
    """Detects/extracts a named function or test item's source text within
    a layer file, in whatever the target language's syntax is."""

    @abstractmethod
    def fn_name_in(self, source: str, name: str) -> bool:
        """True if `source` already defines an item named `name`."""

    @abstractmethod
    def extract_fn_body(self, source: str, name: str) -> tuple[int, int] | None:
        """Returns (start, end) character offsets of the named item
        (including its doc comment/attributes), or None if not found."""

    @abstractmethod
    def extract_fn_name(self, code_snippet: str) -> str | None:
        """Given a snippet of a single function/test definition, returns
        the name it defines, or None if it doesn't look like one."""


class BuildRunner(ABC):
    """Runs the target project's real build/test commands and reports real
    pass/fail - never trust an agent's own narration, per this framework's
    core verification philosophy."""

    @abstractmethod
    def build(self) -> tuple[bool, str]:
        """Returns (success, real captured output tail)."""

    @abstractmethod
    def test(self) -> tuple[bool, str]:
        """Returns (success, real captured output tail). Only meaningful to
        call after build() succeeds."""


class TraceRanker(ABC):
    """Turns a target's own raw execution-trace format into a ranked list of
    real call-target addresses - the input to the `draft` pipeline
    (draft_graph.py). ISA/target-specific because trace formats differ (a
    Saturn portal_trace.jsonl line looks nothing like an N64/RSP trace), but
    the contract is the same everywhere: every returned address must be one
    the target genuinely executed, ranked by real call frequency - never
    invented, never a static-analysis guess. See plugins/sh2_rust.py's
    Sh2RustTraceRanker, validated against a real 51M-line/several-GB jsonl
    capture (jhonatanTeixeira/portal_to_another_world,
    traces/portal_trace_play.jsonl, 2026-08-07 case study)."""

    @abstractmethod
    def rank_targets(self, trace_path: str, top_n: int) -> list[str]:
        """Returns up to `top_n` address strings (uppercase hex, no 0x
        prefix), most-frequently-called first."""


class MechanicalTranslator(ABC):
    """Turns one real call-target address into ISA-appropriate pseudocode -
    zero LLM calls, purely mechanical (decode + CFG-following walk +
    per-instruction transliteration). This is deliberately separated from
    PolishWithLLM (draft_graph.py): every address that CAN be resolved this
    way should be, cheaply and deterministically, before any model call -
    the LLM's job is to clean up/dedupe this output, never to (re)decode
    opcodes itself. See plugins/sh2_rust.py's Sh2RustMechanicalTranslator,
    which vendors jhonatanTeixeira/portal_to_another_world's proven
    tools/sh2dis.py decoder + tools/study_call.py's CFG-following
    disassemble_function() + tools/sh2_to_rust_pseudo.py's per-opcode
    pseudocode emission, unchanged in logic."""

    @abstractmethod
    def translate(self, addr_hex: str) -> "MechanicalResult":
        """Pure, deterministic, no LLM call. `addr_hex` must be one this
        target's binary/trace data can actually resolve - implementations
        should raise rather than fabricate if it can't."""


@dataclass
class MechanicalResult:
    pseudo: str
    resolved: bool
    note: str = ""
