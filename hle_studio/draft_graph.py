"""The `draft` pipeline: raw execution trace -> mechanical pseudocode ->
semantic clustering -> LLM polish -> a first-draft native-function file +
dispatch-table snippet. Generalizes jhonatanTeixeira/portal_to_another_world's
sandbox_hle/pipeline.py (a real, twice-iterated case study run 2026-08-07
against a local july-engine server, Gemma-4-E4B then Qwen3.5-4B) into
target-agnostic pydantic-graph nodes, the same shape as graph.py's porting
graph: everything ISA/language-specific goes through a plugin
(TraceRanker/MechanicalTranslator from plugins/base.py), everything else is
config-driven.

This produces DRAFTS, not verified ports - there is no judge, no build/test
gate here (contrast with graph.py's CheckAlreadyPorted->...->VerifyResult).
The intended flow is `draft` first (cheap, can run against a small local
model), then feed the resulting per-address studies into the real `port`
pipeline for judge review + real build/test verification. Never wire
`draft`'s output directly into a dispatch table without that second pass.

Two real, live-observed failure modes from the case study that shaped this
module's design (not hypothetical - see that project's conversation history
for exact repro):

1. Reasoning models on small/quantized local backends can spend their ENTIRE
   turn re-litigating ambiguous cases inside a code comment ("wait, let me
   reconsider...") and get cut off by the context window before ever closing
   the JSON - looks like a parse bug, is actually unbounded rumination. The
   system prompt below explicitly tells the model it's allowed to say a
   cluster looks wrong and split it (see `split_reason`) SPECIFICALLY so
   indecision has a concrete, terminable action instead of spiraling.
2. A model asked to invent a dispatch-table entry format from prose will
   drift (missing the `Some(...)` wrapper an `Option<NativeFn>` return type
   requires, wrong module path, etc.) - the system prompt now interpolates
   `config.dispatch_entry_template` VERBATIM instead of describing it, so
   there's no format for the model to reconstruct from memory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import httpx
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from hle_studio.config import TargetConfig
from hle_studio.plugins.base import MechanicalTranslator, TraceRanker


@dataclass
class DraftSharedContext:
    config: TargetConfig
    trace_ranker: TraceRanker
    translator: MechanicalTranslator
    llm_base_url: str
    llm_model_alias: str
    llm_api_key: str = "placeholder"
    top_n: int = 20


@dataclass
class DraftState:
    shared: DraftSharedContext
    targets: list[str] = field(default_factory=list)
    mechanical: dict[str, dict] = field(default_factory=dict)  # addr -> {"pseudo","resolved","note"}
    clusters: list[list[str]] = field(default_factory=list)
    polished: dict[str, dict] = field(default_factory=dict)


@dataclass
class SelectTargets(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "MechanicalDisasm":
        shared = ctx.state.shared
        if not shared.config.raw_trace_path:
            raise ValueError("draft needs [trace].raw_trace_path set in hle_studio_target.toml")
        ctx.state.targets = shared.trace_ranker.rank_targets(str(shared.config.raw_trace_path), shared.top_n)
        print(f"[SelectTargets] {len(ctx.state.targets)} alvos reais selecionados por frequência real de chamada")
        return MechanicalDisasm()


@dataclass
class MechanicalDisasm(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "ClusterPseudocode":
        translator = ctx.state.shared.translator
        for addr in ctx.state.targets:
            result = translator.translate(addr)
            ctx.state.mechanical[addr] = {"pseudo": result.pseudo, "resolved": result.resolved, "note": result.note}
        n_resolved = sum(1 for v in ctx.state.mechanical.values() if v["resolved"])
        print(f"[MechanicalDisasm] {len(ctx.state.mechanical)} funções desmontadas mecanicamente "
              f"({n_resolved} completas, {len(ctx.state.mechanical) - n_resolved} parciais) - 0 chamadas de LLM")
        return ClusterPseudocode()


@dataclass
class ClusterPseudocode(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "PolishWithLLM":
        cfg = ctx.state.shared.config
        payload = {addr: v["pseudo"] for addr, v in ctx.state.mechanical.items() if v["pseudo"].strip()}
        try:
            from hle_studio.semantic_clustering import cluster_texts
            result = cluster_texts(payload, model_id=cfg.cluster_model, threshold=cfg.cluster_threshold)
            ctx.state.clusters = result.clusters
            print(f"[ClusterPseudocode] {len(payload)} propostas -> {len(result.clusters)} clusters "
                  f"(limiar={cfg.cluster_threshold})")
        except ImportError as e:
            # Real failure, visible, not silently papered over - fall back to
            # one cluster per address so the pipeline can still finish.
            print(f"[ClusterPseudocode] clustering indisponível ({e}) - 1 cluster por endereço")
            ctx.state.clusters = [[a] for a in payload.keys()]
        return PolishWithLLM()


DRAFT_SYSTEM_PROMPT_TEMPLATE = """You are helping port {isa} code to {output_language} native code.

You receive mechanical pseudocode (NOT valid {output_language} - produced by an automatic,
instruction-by-instruction transliterator with no real control-flow reconstruction) for one or more
addresses that a semantic-duplicate detector (embedding similarity) grouped as likely the same real
implementation.

IMPORTANT - the automatic grouping CAN BE WRONG: embedding similarity is a statistical signal, not
proof. Before writing anything, read every address's pseudocode yourself and decide the REAL subgroups
(same observable input/output, same memory/hardware effect) - do not force a merge just because the
grouping suggested one, and do not spiral into unresolved back-and-forth in a comment if a case looks
ambiguous: pick your best-supported reading, note the uncertainty in ONE line, and move on.

Your task:
1. Re-derive the real subgroups (1 subgroup covering everyone, or up to N subgroups if none share logic).
2. For each real subgroup, write ONE {output_language} function capturing ONLY what the pseudocode
   actually shows - never invent logic, and never invent methods/APIs that weren't in the pseudocode.
3. For each subgroup, the dispatch-table entry in EXACTLY this format (substitute the real address(es)
   and function reference, change nothing else about the shape): `{dispatch_entry_template}`
4. Respond with strict JSON, a list of subgroups even if there's only one:
   {{"groups": [{{"addrs": ["ADDR1","ADDR2"], "code": "...", "dispatch_entry": "...", "fn_name": "...",
                 "split_reason": "only fill in if you split the suggested grouping - say why"}}]}}
   No markdown, no ```, JSON only.
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


@dataclass
class PolishWithLLM(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "WriteOutput":
        shared = ctx.state.shared
        cfg = shared.config
        system_prompt = DRAFT_SYSTEM_PROMPT_TEMPLATE.format(
            isa=cfg.isa, output_language=cfg.output_language,
            dispatch_entry_template=cfg.dispatch_entry_template.format(addr="ADDR1 | 0xADDR2", fn_ref="module::fn_name"),
        )
        headers = {"Authorization": f"Bearer {shared.llm_api_key}"} if shared.llm_api_key else {}
        async with httpx.AsyncClient(timeout=300) as client:
            for i, cluster in enumerate(ctx.state.clusters):
                bodies = []
                for addr in cluster:
                    m = ctx.state.mechanical.get(addr)
                    if not m:
                        continue
                    bodies.append(f"--- {addr} ({'complete' if m['resolved'] else 'PARTIAL: ' + m['note']}) ---\n{m['pseudo']}")
                if not bodies:
                    continue
                user_prompt = (
                    f"Suggested group - {len(cluster)} address(es): {', '.join(cluster)}\n\n" + "\n\n".join(bodies)
                )
                try:
                    resp = await client.post(
                        f"{shared.llm_base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": shared.llm_model_alias,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            # Deliberately no max_tokens: reasoning models on some
                            # local backends burn real completion tokens on a
                            # "Thinking Process:" preamble before `content` - a cap
                            # here risks truncation mid-reasoning, before the model
                            # ever reaches the answer (finish_reason="length",
                            # content=""). See module docstring point 1 for why the
                            # prompt's split_reason escape hatch matters more than
                            # a token budget for the OTHER failure mode this causes.
                            "temperature": 0.1,
                        },
                    )
                    resp.raise_for_status()
                    resp_json = resp.json()
                    usage = resp_json.get("usage", {})
                    content = resp_json["choices"][0]["message"]["content"]
                    groups = _extract_json(content).get("groups", [])
                except Exception as e:  # noqa: BLE001 - a failed cluster is reported, not silently dropped
                    groups = [{"addrs": cluster, "code": f"// LLM FAILED: {e}", "dispatch_entry": "", "fn_name": "error"}]
                    usage = {}
                ctx.state.polished[f"cluster_{i}"] = {"input_cluster": cluster, "groups": groups, "usage": usage}
                split_note = f" -> MODEL SPLIT INTO {len(groups)} SUBGROUPS" if len(groups) > 1 else ""
                print(f"[PolishWithLLM] cluster {i} ({len(cluster)} addrs, "
                      f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
                      f"completion_tokens={usage.get('completion_tokens', '?')}){split_note}")
        return WriteOutput()


@dataclass
class WriteOutput(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> End[dict]:
        total_prompt_tok, total_completion_tok, n_splits = 0, 0, 0
        code_blocks, dispatch_entries = [], []
        for key, p in ctx.state.polished.items():
            usage = p.get("usage", {})
            total_prompt_tok += usage.get("prompt_tokens", 0)
            total_completion_tok += usage.get("completion_tokens", 0)
            groups = p.get("groups", [])
            if len(groups) > 1:
                n_splits += 1
            for g in groups:
                note = f" (model split the cluster: {g['split_reason']})" if g.get("split_reason") else ""
                code_blocks.append(f"// {key} - addrs: {', '.join(g.get('addrs', []))}{note}\n{g.get('code', '')}")
                dispatch_entries.append(g.get("dispatch_entry", ""))

        summary = {
            "n_targets": len(ctx.state.targets),
            "n_mechanical": len(ctx.state.mechanical),
            "n_clusters": len(ctx.state.clusters),
            "n_llm_calls": len(ctx.state.polished),
            "n_model_splits": n_splits,
            "prompt_tokens": total_prompt_tok,
            "completion_tokens": total_completion_tok,
            "code": "\n\n".join(code_blocks),
            "dispatch_entries": dispatch_entries,
            "mechanical": ctx.state.mechanical,
            "clusters": ctx.state.clusters,
            "polished": ctx.state.polished,
        }
        print(f"[WriteOutput] {summary['n_targets']} alvos -> {summary['n_mechanical']} desmontados -> "
              f"{summary['n_clusters']} clusters -> {summary['n_llm_calls']} chamadas ao LLM "
              f"({summary['n_model_splits']} split(s)). "
              f"TOKENS: {total_prompt_tok} entrada / {total_completion_tok} saída.")
        return End(summary)


draft_graph = Graph(nodes=[SelectTargets, MechanicalDisasm, ClusterPseudocode, PolishWithLLM, WriteOutput])
