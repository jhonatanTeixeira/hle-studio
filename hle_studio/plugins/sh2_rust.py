"""The (isa="sh2", output_language="rust") TraceRanker/MechanicalTranslator
pair for the `draft` pipeline - extracted near-verbatim from
jhonatanTeixeira/portal_to_another_world's real 2026-08-07 case study
(sandbox_hle/pipeline.py's `rank_targets` + `MechanicalDisasm` node), where
both were validated against a real 51M-line/several-GB
traces/portal_trace_play.jsonl capture and the project's own extracted
0.BIN/IP.BIN Saturn game binary.
"""
from __future__ import annotations

import subprocess
from collections import Counter

from hle_studio.config import TargetConfig
from hle_studio.plugins.base import MechanicalResult, MechanicalTranslator, TraceRanker
from hle_studio.plugins.sh2 import sh2dis
from hle_studio.plugins.sh2.cfg_walk import disassemble_function
from hle_studio.plugins.sh2.sh2_to_rust_pseudo import compact_pseudo, translate_function


class Sh2RustTraceRanker(TraceRanker):
    """Ranks real call-target addresses by frequency in a raw
    `portal_trace.jsonl`-shaped capture (see
    vendor/yabause/src/portal_trace.c in the originating project for how
    that stream is produced - each `call` event line contains a resolved
    `"target":"XXXXXXXX"` field).

    Uses `grep -o` rather than a per-line `json.loads` loop deliberately:
    this file is routinely 50M+ lines / multi-GB in a real play session,
    and grep's C implementation finishing in under two minutes beats a pure-
    Python full scan by a wide margin for this single extraction (measured:
    ~90s on a real 51M-line/several-GB file)."""

    def rank_targets(self, trace_path: str, top_n: int) -> list[str]:
        out = subprocess.run(
            ["grep", "-o", r'"target":"[0-9A-Fa-f]\{8\}"', trace_path],
            capture_output=True, text=True, timeout=600,
        )
        counts = Counter(line[len('"target":"'):-1].upper() for line in out.stdout.splitlines())
        return [addr for addr, _ in counts.most_common(top_n)]


class Sh2RustMechanicalTranslator(MechanicalTranslator):
    """CFG-following disassembly (cfg_walk.disassemble_function) + per-
    instruction Rust-pseudo emission (sh2_to_rust_pseudo.translate_function)
    for one real, trace-confirmed SH-2 function entry point - zero LLM
    calls, fully deterministic."""

    def __init__(self, config: TargetConfig):
        self.config = config
        self._data = None  # lazy-loaded, see _binary_data()

    def _binary_data(self):
        if self._data is None:
            # [trace].game_binary_path/game_binary_secondary_path - kept
            # deliberately separate from [paths].reference_source_roots
            # (that's C/C++ reference source for the porting agent's own
            # tools, an unrelated thing - see config.py's field comment).
            bin_path = self.config.game_binary_path
            if bin_path is None:
                raise ValueError("Sh2RustMechanicalTranslator needs [trace].game_binary_path set "
                                  "in hle_studio_target.toml")
            ip_bin_path = self.config.game_binary_secondary_path
            if ip_bin_path:
                self._data = sh2dis.load_combined_binary(str(bin_path), str(ip_bin_path))
            else:
                self._data = sh2dis.load_binary(str(bin_path))
        return self._data

    def translate(self, addr_hex: str) -> MechanicalResult:
        addr = int(addr_hex, 16)
        blocks, all_resolved, note = disassemble_function(addr, self._binary_data())
        decs = [d for block in blocks for d in block]
        # compact_pseudo(): this feeds an LLM prompt downstream (draft_graph's
        # PolishWithLLM) - real, measured token cost, not just tidiness.
        pseudo_lines = compact_pseudo(translate_function(decs))
        return MechanicalResult(pseudo="\n".join(pseudo_lines), resolved=all_resolved, note=note)
