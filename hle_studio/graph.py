"""The generic porting pipeline: CheckAlreadyPorted -> RunAgentTurn ->
JudgeReview -> FixAfterJudgeRejection -> VerifyResult, as pydantic-graph
nodes. Ported near-verbatim (control-flow shape unchanged) from
jhonatanTeixeira/portal_to_another_world's tools/agent_study_all.py, where
this exact shape was validated against real Saturn/SH-2/Rust porting runs -
including two real, live-caught bugs this version already has the fix for
(see judge.py's docstring, and the address-registration dedup bug noted in
plugins/rust_cargo.py).

Everything project-specific is reached through `SharedPortContext.config`
and its three plugin objects (dispatch_editor/extractor/build_runner) - this
module itself never imports anything Rust- or SH-2-specific.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from hle_studio.config import TargetConfig
from hle_studio.judge import JudgeVerdict
from hle_studio.plugins.base import BuildRunner, DispatchEditor, FunctionExtractor

import asyncio

MAX_DEPENDENCY_DEPTH = 5
MAX_JUDGE_FIX_ATTEMPTS = 3
MAX_TOOL_REQUESTS_PER_TURN = 20
TURN_USAGE_LIMITS = UsageLimits(request_limit=MAX_TOOL_REQUESTS_PER_TURN)


@dataclass
class SharedPortContext:
    write_lock: asyncio.Lock
    agent: Agent
    judge: Agent
    config: TargetConfig
    dispatch_editor: DispatchEditor
    extractor: FunctionExtractor
    build_runner: BuildRunner
    in_progress: set = field(default_factory=set)


@dataclass
class PortState:
    addr_hex: str
    depth: int
    shared: SharedPortContext
    agent_report: str = ""
    pre_snapshot: dict = field(default_factory=dict)
    judge_verdict: "JudgeVerdict | None" = None
    judge_fix_attempts: int = 0


def addresses_already_ported(shared: SharedPortContext) -> set[str]:
    table_path = shared.config.dispatch_table_file
    if not table_path.exists():
        return set()
    return shared.dispatch_editor.list_registered_addresses(table_path.read_text(encoding="utf-8"))


def next_unported_targets(shared: SharedPortContext) -> list[str]:
    """Every address with a ready study that ISN'T already in the dispatch
    table, in ascending address order - computed HERE, by the driver, so
    the agent is never in a position to preview several and cherry-pick the
    easiest one (a real failure mode caught live: a model given freedom to
    browse studies gravitated toward whichever looked easiest and called
    that 'not skipping')."""
    ported = addresses_already_ported(shared)
    cfg = shared.config
    name_re = re.compile(cfg.study_filename_regex)
    targets = []
    for path in sorted(glob.glob(str(cfg.project_root / cfg.studies_glob))):
        fname = os.path.basename(path)
        m = name_re.match(fname)
        if not m:
            continue
        addr = m.group(1).upper()
        if addr in ported:
            continue
        content = Path(path).read_text(encoding="utf-8", errors="ignore")
        if cfg.study_ready_marker in content:
            targets.append(addr)
    return targets


def _snapshot_files(shared: SharedPortContext) -> dict:
    cfg = shared.config
    snap = {}
    if cfg.native_functions_dir.exists():
        for p in cfg.native_functions_dir.rglob("*"):
            if p.is_file():
                try:
                    snap[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    pass
    if cfg.dispatch_table_file.exists():
        snap[str(cfg.dispatch_table_file)] = cfg.dispatch_table_file.read_text(
            encoding="utf-8", errors="ignore"
        )
    return snap


def _diff_added_lines(pre: dict, post: dict) -> str:
    chunks = []
    for path, new_content in post.items():
        old_lines = set(pre.get(path, "").splitlines())
        added = [line for line in new_content.splitlines() if line not in old_lines]
        if added:
            chunks.append(f"--- {path} (linhas novas) ---\n" + "\n".join(added))
    return "\n\n".join(chunks) if chunks else "(nenhuma linha nova detectada em nenhum arquivo rastreado)"


@dataclass
class CheckAlreadyPorted(BaseNode[PortState]):
    """Cheap first check, no LLM call - bail out if the target is already
    ported, already mid-flight elsewhere in this run, or the dependency
    chain went too deep."""

    async def run(self, ctx: GraphRunContext[PortState]) -> "RunAgentTurn | End[str]":
        addr = ctx.state.addr_hex
        shared = ctx.state.shared
        if addr in addresses_already_ported(shared):
            return End(f"0x{addr}: já portado, nada a fazer")
        if addr in shared.in_progress:
            return End(f"0x{addr}: ciclo de dependência detectado (já em andamento nesta cadeia) - "
                        "não recursar de novo, resolver manualmente")
        if ctx.state.depth > MAX_DEPENDENCY_DEPTH:
            return End(f"0x{addr}: profundidade de dependência > {MAX_DEPENDENCY_DEPTH}, pare e reporte")
        shared.in_progress.add(addr)
        ctx.state.pre_snapshot = _snapshot_files(shared)
        return RunAgentTurn()


@dataclass
class RunAgentTurn(BaseNode[PortState]):
    """The actual LLM turn - one agent.run() call, with the full toolset
    (see agent_tools.py), including a tool that recursively chases a
    dependency through this exact same graph before returning."""

    async def run(self, ctx: GraphRunContext[PortState]) -> "JudgeReview":
        addr = ctx.state.addr_hex
        prompt = (
            f"Your assigned target for this turn is 0x{addr}. Call read_study('{addr}') first, "
            "then follow the process. Do not call read_study on any OTHER address unless you've "
            "determined it's a genuine dependency, in which case use port_dependency_first on it "
            "rather than reading it and faking the rest."
        )
        try:
            result = await ctx.state.shared.agent.run(prompt, deps=ctx.state, usage_limits=TURN_USAGE_LIMITS)
            ctx.state.agent_report = result.output
        except Exception as e:  # noqa: BLE001
            ctx.state.agent_report = f"[agente falhou nesse turno: {e}]"
        return JudgeReview()


@dataclass
class JudgeReview(BaseNode[PortState]):
    """A separate, different-family, cheap agent reviews only the diff -
    no game/domain context. On rejection, routes to FixAfterJudgeRejection
    (which hands the exact findings back to the porting agent) instead of
    just reporting and moving on, up to MAX_JUDGE_FIX_ATTEMPTS rounds."""

    async def run(self, ctx: GraphRunContext[PortState]) -> "FixAfterJudgeRejection | VerifyResult":
        shared = ctx.state.shared
        added = _diff_added_lines(ctx.state.pre_snapshot, _snapshot_files(shared))
        if added.startswith("(nenhuma linha nova"):
            ctx.state.judge_verdict = JudgeVerdict(approved=True, findings=[])
            return VerifyResult()
        try:
            result = await shared.judge.run(
                f"Review this newly added code (address 0x{ctx.state.addr_hex}):\n\n{added}",
                usage_limits=TURN_USAGE_LIMITS,
            )
            ctx.state.judge_verdict = result.output
        except Exception as e:  # noqa: BLE001
            from hle_studio.judge import JudgeFinding
            ctx.state.judge_verdict = JudgeVerdict(approved=False, findings=[
                JudgeFinding(rule="(falha do juiz)", what_is_wrong=f"juiz falhou ao revisar: {e}", snippet="")
            ])
        if ctx.state.judge_verdict.approved:
            return VerifyResult()
        if ctx.state.judge_fix_attempts >= MAX_JUDGE_FIX_ATTEMPTS:
            return VerifyResult()
        return FixAfterJudgeRejection()


@dataclass
class FixAfterJudgeRejection(BaseNode[PortState]):
    """Hands the porting agent the judge's EXACT findings and asks it to
    fix them in place - never by appending a second attempt. Loops back to
    JudgeReview for a fresh review of the fix."""

    async def run(self, ctx: GraphRunContext[PortState]) -> "JudgeReview":
        ctx.state.judge_fix_attempts += 1
        findings = ctx.state.judge_verdict.findings if ctx.state.judge_verdict else []
        findings_text = "\n".join(
            f"- RULE {f.rule}: {f.what_is_wrong}\n  offending snippet: {f.snippet}" for f in findings
        )
        prompt = (
            f"A separate code reviewer checked your changes for target 0x{ctx.state.addr_hex} and "
            f"REJECTED them (fix attempt {ctx.state.judge_fix_attempts}/{MAX_JUDGE_FIX_ATTEMPTS}). "
            "Exact findings, each one a real problem you must fix - read 'what_is_wrong' carefully, "
            "it names the SPECIFIC problem - do not guess which part of the snippet is wrong:\n\n"
            + findings_text
            + "\n\nFix every one of these by editing the offending item in place - never by appending "
              "another attempt next to the rejected code. After fixing, verify build/test yourself "
              "before finishing this turn."
        )
        try:
            result = await ctx.state.shared.agent.run(prompt, deps=ctx.state, usage_limits=TURN_USAGE_LIMITS)
            ctx.state.agent_report += f"\n\n[correção {ctx.state.judge_fix_attempts}]: {result.output}"
        except Exception as e:  # noqa: BLE001
            ctx.state.agent_report += f"\n\n[correção {ctx.state.judge_fix_attempts} falhou: {e}]"
        return JudgeReview()


@dataclass
class VerifyResult(BaseNode[PortState]):
    """Never trust the agent's own narration - re-check for real, under the
    same write_lock so this doesn't race a sibling worker's own
    verification, AND require the judge's approval on top of a clean
    build/test."""

    async def run(self, ctx: GraphRunContext[PortState]) -> End[str]:
        addr = ctx.state.addr_hex
        shared = ctx.state.shared
        async with shared.write_lock:
            registered = addr in addresses_already_ported(shared)
            build_ok, build_tail = shared.build_runner.build()
            test_ok, test_tail = (False, build_tail)
            if build_ok:
                test_ok, test_tail = shared.build_runner.test()
            judge_ok = ctx.state.judge_verdict is not None and ctx.state.judge_verdict.approved
        shared.in_progress.discard(addr)
        status = "CONFIRMADO" if (registered and build_ok and test_ok and judge_ok) else "NÃO CONFIRMADO"
        if ctx.state.judge_verdict is None:
            findings_text = "(juiz não rodou)"
        elif not ctx.state.judge_verdict.findings:
            findings_text = "(nenhum)"
        else:
            findings_text = "\n".join(
                f"- RULE {f.rule}: {f.what_is_wrong} | trecho: {f.snippet}"
                for f in ctx.state.judge_verdict.findings
            )
        return End(
            f"0x{addr}: {status} (registrado={registered}, build={build_ok}, test={test_ok}, "
            f"juiz_aprovou={judge_ok}, tentativas_de_correção={ctx.state.judge_fix_attempts})\n"
            f"relato do agente: {ctx.state.agent_report}\n"
            f"achados finais do juiz:\n{findings_text}\n"
            f"--- build/test (real, agora) ---\n{test_tail}"
        )


graph = Graph(nodes=[CheckAlreadyPorted, RunAgentTurn, JudgeReview, FixAfterJudgeRejection, VerifyResult])
