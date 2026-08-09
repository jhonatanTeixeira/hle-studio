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

import asyncio
import json
import re
from dataclasses import dataclass, field

import httpx
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from hle_studio.config import TargetConfig
from hle_studio.plugins.base import BuildRunner, DispatchEditor, FunctionExtractor, MechanicalTranslator, TraceRanker


@dataclass
class DraftSharedContext:
    config: TargetConfig
    trace_ranker: TraceRanker
    translator: MechanicalTranslator
    llm_base_url: str
    llm_model_alias: str
    llm_api_key: str = "placeholder"
    top_n: int = 20
    # Only needed if `write_and_verify` is True (see WriteAndVerify's own
    # docstring for why this is opt-in, never automatic) - the SAME plugin
    # trio the `port` graph uses, reused rather than reforked.
    dispatch_editor: DispatchEditor | None = None
    extractor: FunctionExtractor | None = None
    build_runner: BuildRunner | None = None
    write_and_verify: bool = False


@dataclass
class DraftState:
    shared: DraftSharedContext
    targets: list[str] = field(default_factory=list)
    mechanical: dict[str, dict] = field(default_factory=dict)  # addr -> {"pseudo","resolved","note"}
    dependencies: dict[str, set[str]] = field(default_factory=dict)  # addr -> {addr, ...} - IN this batch only
    external_dependencies: dict[str, set[str]] = field(default_factory=dict)  # addr -> {addr, ...} - outside this batch
    clusters: list[list[str]] = field(default_factory=list)
    polished: dict[str, dict] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


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
    async def run(self, ctx: GraphRunContext[DraftState]) -> "BuildDependencyGraph":
        translator = ctx.state.shared.translator
        for addr in ctx.state.targets:
            result = translator.translate(addr)
            ctx.state.mechanical[addr] = {"pseudo": result.pseudo, "resolved": result.resolved, "note": result.note}
        n_resolved = sum(1 for v in ctx.state.mechanical.values() if v["resolved"])
        print(f"[MechanicalDisasm] {len(ctx.state.mechanical)} funções desmontadas mecanicamente "
              f"({n_resolved} completas, {len(ctx.state.mechanical) - n_resolved} parciais) - 0 chamadas de LLM")
        return BuildDependencyGraph()


# Matches how translate_one_rust (sh2_to_rust_pseudo.py, the sh2_rust plugin's
# emission backend) renders a call target - a static BSR/JSR-to-a-resolved-
# address becomes `call_L<addr>()`, and a register-indirect JSR/BRAF/BSRF that
# compact_pseudo's local constant substitution managed to resolve to a literal
# becomes `call_dynamic(0x<addr>)`. Both forms carry the real target address
# in the text itself - no need to touch MechanicalTranslator's own interface
# to get dependency edges out of it; this is deliberately just a regex pass
# over pseudocode that's already produced, same "don't invent a new capture
# mechanism when the data already exists in another form" lesson as this
# project's branch_targets_by_pc.json (see docs/ in the originating project).
_CALL_TARGET_RE = re.compile(r"call_L([0-9A-Fa-f]{8})\(\)|call_dynamic\(0x([0-9A-Fa-f]{8})\)")


def extract_call_targets(pseudo: str) -> set[str]:
    """Every address this pseudocode calls, uppercase hex, deduplicated -
    from the two regex forms above. A register-indirect call that compact
    constant-propagation could NOT resolve to a literal is invisible here
    (renders as `call_dynamic(rN)`) - a real, honest gap, not silently
    guessed at; see MechanicalResult.note for whether the source function
    itself was fully resolved."""
    targets = set()
    for m in _CALL_TARGET_RE.finditer(pseudo):
        targets.add((m.group(1) or m.group(2)).upper())
    return targets


@dataclass
class BuildDependencyGraph(BaseNode[DraftState]):
    """Real call edges between the addresses in THIS batch, extracted from
    the mechanical pseudocode already produced (zero extra disassembly, zero
    LLM calls) - not a discovery mechanism like the originating project's
    tools/expand_call_graph.py (which finds NEW addresses to study), but an
    ORDERING one: which of the addresses we're ABOUT to draft call which
    others we're ALSO about to draft, so ClusterPseudocode can process
    callees before their callers and PolishWithLLM can tell the model
    "this calls X, already drafted in this run as fn_x - reuse that name"
    instead of the model inventing an unrelated one."""

    async def run(self, ctx: GraphRunContext[DraftState]) -> "ClusterPseudocode":
        in_batch = set(ctx.state.mechanical.keys())
        for addr, m in ctx.state.mechanical.items():
            called = extract_call_targets(m["pseudo"])
            ctx.state.dependencies[addr] = called & in_batch
            ctx.state.external_dependencies[addr] = called - in_batch
        n_edges = sum(len(v) for v in ctx.state.dependencies.values())
        n_external = sum(len(v) for v in ctx.state.external_dependencies.values())
        print(f"[BuildDependencyGraph] {n_edges} dependências reais dentro do lote, "
              f"{n_external} apontam pra fora do lote selecionado")

        # Fan-in: addresses many OTHERS in this batch depend on - the "shared
        # foundation" the docs/native_api_foundation.md doc talks about.
        # Purely informational here (the topo order already processes these
        # first regardless) - printed so a human watching the run sees which
        # addresses are worth double-checking once drafted, since a mistake
        # in one of these propagates into everything that reuses it.
        fan_in: dict[str, int] = {}
        for deps in ctx.state.dependencies.values():
            for dep in deps:
                fan_in[dep] = fan_in.get(dep, 0) + 1
        shared = sorted((c, a) for a, c in fan_in.items() if c >= 2)
        if shared:
            top = ", ".join(f"{a} ({c}x)" for c, a in reversed(shared[-5:]))
            print(f"[BuildDependencyGraph] endereços chamados por >=2 outros do lote (fundação "
                  f"compartilhada, processados primeiro): {top}")
        return ClusterPseudocode()


def _topo_order_clusters(clusters: list[list[str]], dependencies: dict[str, set[str]]) -> list[list[str]]:
    """Kahn's algorithm at cluster granularity: cluster A must be processed
    after cluster B if any address in A calls any address in B. Real code
    has real cycles (mutual/indirect recursion) - a cycle can't be perfectly
    serialized, so any cluster still blocked once nothing else is ready gets
    appended in original order with a printed note, rather than silently
    dropped or crashing."""
    addr_to_cluster = {a: i for i, c in enumerate(clusters) for a in c}
    cluster_deps: list[set[int]] = [set() for _ in clusters]
    for i, cluster in enumerate(clusters):
        for a in cluster:
            for dep in dependencies.get(a, ()):
                j = addr_to_cluster.get(dep)
                if j is not None and j != i:
                    cluster_deps[i].add(j)

    order: list[int] = []
    done = [False] * len(clusters)
    remaining = set(range(len(clusters)))
    while remaining:
        ready = [i for i in remaining if cluster_deps[i] <= set(order)]
        if not ready:
            # Real cycle - break it by taking whatever's left in original
            # order rather than guessing which edge to cut.
            print(f"[ClusterPseudocode] {len(remaining)} cluster(s) em dependência cíclica - "
                  f"processando na ordem original, sem tentar quebrar o ciclo")
            ready = sorted(remaining)
        for i in sorted(ready):
            order.append(i)
            remaining.discard(i)
    return [clusters[i] for i in order]


def _topo_waves(clusters: list[list[str]], dependencies: dict[str, set[str]]) -> list[list[int]]:
    """Same Kahn's-algorithm dependency graph as `_topo_order_clusters`, but
    returns the *waves* (each a list of cluster-indices with no dependency
    on each other, or on anything not already in an earlier wave) instead of
    flattening them into one order. This is what makes concurrent
    `PolishWithLLM` calls safe: every cluster within one wave is provably
    independent of every other cluster in that same wave, so firing them all
    at once can never race a caller against a callee it needs the drafted
    name of - `ctx.state.polished` is only ever read from PRIOR, fully-
    completed waves, never from wave-mates still in flight. Real cycles get
    the same fallback as `_topo_order_clusters`: whatever's left becomes one
    final wave, processed concurrently since there's no ordering left to
    respect among them anyway."""
    addr_to_cluster = {a: i for i, c in enumerate(clusters) for a in c}
    cluster_deps: list[set[int]] = [set() for _ in clusters]
    for i, cluster in enumerate(clusters):
        for a in cluster:
            for dep in dependencies.get(a, ()):
                j = addr_to_cluster.get(dep)
                if j is not None and j != i:
                    cluster_deps[i].add(j)

    waves: list[list[int]] = []
    done: set[int] = set()
    remaining = set(range(len(clusters)))
    while remaining:
        ready = [i for i in remaining if cluster_deps[i] <= done]
        if not ready:
            print(f"[PolishWithLLM] {len(remaining)} cluster(s) em dependência cíclica - "
                  f"tratando como uma onda final só, sem tentar quebrar o ciclo")
            ready = sorted(remaining)
        waves.append(sorted(ready))
        done.update(ready)
        remaining.difference_update(ready)
    return waves


@dataclass
class ClusterPseudocode(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "PolishWithLLM":
        cfg = ctx.state.shared.config
        payload = {addr: v["pseudo"] for addr, v in ctx.state.mechanical.items() if v["pseudo"].strip()}
        try:
            from hle_studio.semantic_clustering import cluster_texts
            result = cluster_texts(payload, model_id=cfg.cluster_model, threshold=cfg.cluster_threshold)
            clusters = result.clusters
            print(f"[ClusterPseudocode] {len(payload)} propostas -> {len(clusters)} clusters "
                  f"(limiar={cfg.cluster_threshold})")
        except ImportError as e:
            # Real failure, visible, not silently papered over - fall back to
            # one cluster per address so the pipeline can still finish.
            print(f"[ClusterPseudocode] clustering indisponível ({e}) - 1 cluster por endereço")
            clusters = [[a] for a in payload.keys()]
        ctx.state.clusters = _topo_order_clusters(clusters, ctx.state.dependencies)
        return PolishWithLLM()


DRAFT_SYSTEM_PROMPT_TEMPLATE = """You are helping port {isa} code to {output_language} native code.
{api_foundation_block}
You receive mechanical pseudocode (NOT valid {output_language} - produced by an automatic,
instruction-by-instruction transliterator with no real control-flow reconstruction) for one or more
addresses that a semantic-duplicate detector (embedding similarity) grouped as likely the same real
implementation.

IMPORTANT - the automatic grouping CAN BE WRONG: embedding similarity is a statistical signal, not
proof. Before writing anything, read every address's pseudocode yourself and decide the REAL subgroups
(same observable input/output, same memory/hardware effect) - do not force a merge just because the
grouping suggested one, and do not spiral into unresolved back-and-forth in a comment if a case looks
ambiguous: pick your best-supported reading, note the uncertainty in ONE line, and move on.

This project organizes native functions into fixed layers/modules - one file per layer, real ones
already exist in the codebase, you're placing new code into one of them, never inventing your own
module path or writing a bare unqualified reference:
{layers_block}

IMPORTANT - some pseudocode contains a line like `// TODO sh2_to_rust_pseudo: mnemônico não
traduzido: UNKNOWN XXXX` or a header `// pendências: opcode desconhecido em ADDR`. This marks a point
the mechanical disassembler could not statically resolve - confirmed (opcode-by-opcode, against the
real reference decoder) to be a genuine dead end: on real hardware AND in the reference emulator, the
CPU fetching an opcode with no defined meaning does NOT continue silently - it raises an illegal-
instruction exception (saves SR/PC, jumps to the exception vector). Do the equivalent: emit an
explicit, loud failure at that exact point (e.g. `unreachable!("...")` or `panic!("...")` describing
what was unresolved) - never invent what the logic "should" do past that point, and never quietly omit
it either. This does NOT mean skip the function: the address it belongs to is a real, trace-confirmed
call target (the game genuinely calls it) - implement everything you DO have solid evidence for
faithfully, and only the specific unresolved branch gets the explicit trap.

Your task:
1. Re-derive the real subgroups (1 subgroup covering everyone, or up to N subgroups if none share logic).
2. For each real subgroup, write ONE {output_language} function capturing ONLY what the pseudocode
   actually shows - never invent logic, and never invent methods/APIs that weren't in the pseudocode.
3. For each subgroup, pick the SINGLE best-fitting `layer` from the list above by its exact key (e.g.
   "hardware_io") - never a path you invent, never "crate::...", never a bare function with no layer.
4. Respond with strict JSON, a list of subgroups even if there's only one:
   {{"groups": [{{"addrs": ["ADDR1","ADDR2"], "code": "...", "fn_name": "...", "layer": "hardware_io",
                 "split_reason": "only fill in if you split the suggested grouping - say why"}}]}}
   No markdown, no ```, JSON only. Do NOT include a "dispatch_entry" field - the exact dispatch-table
   line is assembled mechanically from your `layer` + `fn_name`, never from text you write, so it can
   never come out malformed.
"""


def _find_drafted(polished: dict, addr: str) -> dict | None:
    """Looks up whether `addr` was already resolved by an earlier-processed
    cluster in THIS run (dependency-ordered by _topo_order_clusters) -
    returns that group's dict (fn_name/dispatch_entry) or None if it hasn't
    been drafted yet (a forward/external/cyclic dependency - legitimate,
    not an error; the model is simply not told about it)."""
    for p in polished.values():
        for g in p.get("groups", []):
            if addr in g.get("addrs", []):
                return g
    return None


def _compute_dispatch_entry(cfg: TargetConfig, group: dict) -> str:
    """Assembles the real dispatch-table line mechanically from the model's
    `layer` + `fn_name` choice - never from a `dispatch_entry` string the
    model wrote itself. A real, twice-observed failure mode this replaces:
    asked to write the dispatch line as free text, the model produced FOUR
    different module-path conventions across one 15-address run
    (`hardware_io::fn`, `crate::native_functions::fn`, `native_functions::fn`,
    and bare `fn` with no module at all) - none of which is guaranteed to
    even compile, since draft doesn't know which module a function will
    really live in. Constraining the model to pick a `layer` NAME from a
    fixed list (a discrete choice, not free text) and computing the path
    ourselves makes a malformed dispatch entry structurally impossible."""
    layer = group.get("layer", "")
    if layer not in cfg.layer_names():
        # Doesn't guess a different layer - flags the entry as needing a
        # real decision instead of silently picking the first one.
        group["layer_warning"] = f"model returned unknown layer {layer!r}, not one of {cfg.layer_names()}"
        layer = layer or "UNKNOWN_LAYER"
    addrs = group.get("addrs", [])
    # dispatch_entry_template already has a literal "0x" before {addr} (e.g.
    # "0x{addr} => Some({fn_ref}),") - meant for a single bare hex address.
    # For a multi-address match arm the real Rust shape is
    # `0xA | 0xB | 0xC => ...`, each with its own "0x" - so the FIRST
    # address stays bare (the template's fixed "0x" covers it) and every
    # address after gets its own explicit "0x" prefix, or the template
    # would silently duplicate/omit a prefix.
    addr = " | ".join([addrs[0]] + [f"0x{a}" for a in addrs[1:]]) if addrs else "00000000"
    fn_name = group.get("fn_name", "unnamed")
    return cfg.dispatch_entry_template.format(addr=addr, fn_ref=f"{layer}::{fn_name}")


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text)


# Bounded, not unlimited: DeepInfra (or any provider) has real rate limits,
# and an unbounded `asyncio.gather` over potentially hundreds of clusters
# would fire them all at once. 8 concurrent requests was picked as a
# reasonable default matching the same number floated when this concurrency
# support was requested (2026-08-08) - not measured/tuned beyond that yet.
POLISH_CONCURRENCY = 8


async def _polish_one_cluster(
    client: httpx.AsyncClient, headers: dict, shared: "DraftSharedContext", cfg: "TargetConfig",
    system_prompt: str, i: int, cluster: list[str], mechanical: dict, dependencies: dict,
    polished_so_far: dict,
) -> tuple[str, dict] | None:
    """One cluster's full request/response cycle, as a standalone coroutine
    so `PolishWithLLM` can run several of these concurrently via
    `asyncio.gather` within one dependency-topological wave (see
    `_topo_waves`). `polished_so_far` is a snapshot of `ctx.state.polished`
    taken before the wave started - safe to read from concurrently since
    nothing in the CURRENT wave writes to it until the whole wave finishes
    (every cluster in one wave is, by construction, independent of every
    other cluster in that same wave)."""
    bodies = []
    dep_notes = []
    for addr in cluster:
        m = mechanical.get(addr)
        if not m:
            continue
        bodies.append(f"--- {addr} ({'complete' if m['resolved'] else 'PARTIAL: ' + m['note']}) ---\n{m['pseudo']}")
        for dep in sorted(dependencies.get(addr, ())):
            drafted = _find_drafted(polished_so_far, dep)
            if drafted:
                dep_notes.append(f"- {addr} calls {dep}, already drafted earlier in THIS run as "
                                  f"`{drafted['fn_name']}` ({drafted['dispatch_entry']}) - reuse that "
                                  f"name/reference, don't invent a new one for the same address.")
    if not bodies:
        return None
    dep_block = ("\n\nKnown dependencies already drafted in this run:\n" + "\n".join(dep_notes)
                 if dep_notes else "")
    user_prompt = (
        f"Suggested group - {len(cluster)} address(es): {', '.join(cluster)}\n\n"
        + "\n\n".join(bodies) + dep_block
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
        for g in groups:
            g["dispatch_entry"] = _compute_dispatch_entry(cfg, g)
    except Exception as e:  # noqa: BLE001 - a failed cluster is reported, not silently dropped
        groups = [{"addrs": cluster, "code": f"// LLM FAILED: {e}", "dispatch_entry": "", "fn_name": "error"}]
        usage = {}
    split_note = f" -> MODEL SPLIT INTO {len(groups)} SUBGROUPS" if len(groups) > 1 else ""
    cost = usage.get("estimated_cost")
    cost_note = f", cost=${cost:.6f}" if cost is not None else ""
    print(f"[PolishWithLLM] cluster {i} ({len(cluster)} addrs, "
          f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
          f"completion_tokens={usage.get('completion_tokens', '?')}{cost_note}){split_note}")
    return f"cluster_{i}", {"input_cluster": cluster, "groups": groups, "usage": usage}


@dataclass
class PolishWithLLM(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "WriteOutput":
        shared = ctx.state.shared
        cfg = shared.config
        api_foundation_block = ""
        if cfg.api_foundation_doc and cfg.api_foundation_doc.exists():
            api_foundation_block = (
                "\n=== REAL target API contract - copy these signatures EXACTLY, never invent your own ===\n"
                + cfg.api_foundation_doc.read_text(encoding="utf-8")
                + "\n=== end of real API contract ===\n"
            )
        layers_block = "\n".join(f"- {l.name}: {l.title}" for l in cfg.layers) or "- (no layers configured)"
        system_prompt = DRAFT_SYSTEM_PROMPT_TEMPLATE.format(
            isa=cfg.isa, output_language=cfg.output_language,
            api_foundation_block=api_foundation_block,
            layers_block=layers_block,
        )
        headers = {"Authorization": f"Bearer {shared.llm_api_key}"} if shared.llm_api_key else {}
        waves = _topo_waves(ctx.state.clusters, ctx.state.dependencies)
        print(f"[PolishWithLLM] {len(ctx.state.clusters)} clusters em {len(waves)} onda(s) "
              f"topológica(s), até {POLISH_CONCURRENCY} chamadas concorrentes por onda")
        semaphore = asyncio.Semaphore(POLISH_CONCURRENCY)

        async def bounded(i: int, cluster: list[str], polished_snapshot: dict):
            async with semaphore:
                return await _polish_one_cluster(
                    client, headers, shared, cfg, system_prompt, i, cluster,
                    ctx.state.mechanical, ctx.state.dependencies, polished_snapshot,
                )

        async with httpx.AsyncClient(timeout=300) as client:
            for wave in waves:
                # Snapshot BEFORE firing the wave's concurrent requests - every
                # cluster in this wave only ever needs to see PRIOR waves'
                # results (that's what makes them independent enough to run
                # concurrently in the first place), never each other's
                # still-in-flight ones.
                polished_snapshot = dict(ctx.state.polished)
                results = await asyncio.gather(
                    *(bounded(i, ctx.state.clusters[i], polished_snapshot) for i in wave)
                )
                for result in results:
                    if result is None:
                        continue
                    key, value = result
                    ctx.state.polished[key] = value
        return WriteOutput()


@dataclass
class WriteOutput(BaseNode[DraftState]):
    async def run(self, ctx: GraphRunContext[DraftState]) -> "WriteAndVerify | End[dict]":
        total_prompt_tok, total_completion_tok, n_splits = 0, 0, 0
        total_cost = 0.0
        have_cost = False  # tracks whether ANY call actually reported a real
                            # cost, so a provider that never sends this field
                            # (usage.estimated_cost is DeepInfra-specific, not
                            # a standard OpenAI response field) reports "n/a"
                            # instead of a misleading "$0.00".
        code_blocks, dispatch_entries = [], []
        for key, p in ctx.state.polished.items():
            usage = p.get("usage", {})
            total_prompt_tok += usage.get("prompt_tokens", 0)
            total_completion_tok += usage.get("completion_tokens", 0)
            if usage.get("estimated_cost") is not None:
                total_cost += usage["estimated_cost"]
                have_cost = True
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
            "estimated_cost_usd": total_cost if have_cost else None,
            "code": "\n\n".join(code_blocks),
            "dispatch_entries": dispatch_entries,
            "mechanical": ctx.state.mechanical,
            "clusters": ctx.state.clusters,
            "polished": ctx.state.polished,
            "dependencies": {a: sorted(d) for a, d in ctx.state.dependencies.items() if d},
            "external_dependencies": {a: sorted(d) for a, d in ctx.state.external_dependencies.items() if d},
        }
        cost_line = f"${total_cost:.6f} (real, via usage.estimated_cost)" if have_cost else "n/a (provider didn't report it)"
        print(f"[WriteOutput] {summary['n_targets']} alvos -> {summary['n_mechanical']} desmontados -> "
              f"{summary['n_clusters']} clusters -> {summary['n_llm_calls']} chamadas ao LLM "
              f"({summary['n_model_splits']} split(s)). "
              f"TOKENS: {total_prompt_tok} entrada / {total_completion_tok} saída. CUSTO: {cost_line}.")
        ctx.state.summary = summary
        if ctx.state.shared.write_and_verify:
            return WriteAndVerify()
        return End(summary)


@dataclass
class WriteAndVerify(BaseNode[DraftState]):
    """Opt-in only (DraftSharedContext.write_and_verify, CLI `--write`) -
    NEVER runs by default. Writes each drafted group into its real layer
    file + inserts its dispatch entry, using the SAME DispatchEditor/
    FunctionExtractor the `port` graph already trusts (not reforked), then
    runs a real `cargo build` + `cargo test` and reports pass/fail.

    This is NOT the judge review `port` does (no separate model checking
    conventions/gotchas) - it's the one gate that's non-negotiable regardless:
    a real, live-observed case (2026-08-08) had `draft` output that compiled
    clean but pushed the WRONG value to the stack (a shadowed `let` binding
    silently overwrote a saved register before it was used) - Rust's own
    compiler doesn't catch that, only a real test with real register values
    would. Skipping straight from `draft` to a trusted dispatch table
    without at least this gate would let exactly that class of bug in
    silently."""

    async def run(self, ctx: GraphRunContext[DraftState]) -> End[dict]:
        shared = ctx.state.shared
        cfg = shared.config
        if not (shared.dispatch_editor and shared.extractor and shared.build_runner):
            ctx.state.summary["write_error"] = "write_and_verify=True but dispatch_editor/extractor/build_runner not set"
            print(f"[WriteAndVerify] {ctx.state.summary['write_error']} - nada escrito")
            return End(ctx.state.summary)

        table_path = cfg.dispatch_table_file
        table_content = table_path.read_text(encoding="utf-8") if table_path.exists() else ""
        written_files, skipped, dispatch_inserted = [], [], []

        for key, p in ctx.state.polished.items():
            for g in p.get("groups", []):
                fn_name = g.get("fn_name")
                layer = g.get("layer")
                code = g.get("code", "")
                if not fn_name or layer not in cfg.layer_names() or not code.strip():
                    skipped.append({"key": key, "reason": g.get("layer_warning") or "missing fn_name/layer/code"})
                    continue
                layer_path = cfg.native_functions_dir / f"{layer}.{cfg.layer_file_extension}"
                existing = layer_path.read_text(encoding="utf-8") if layer_path.exists() else ""
                if shared.extractor.fn_name_in(existing, fn_name):
                    skipped.append({"key": key, "reason": f"{fn_name} já existe em {layer_path.name} - não sobrescrito"})
                    continue
                addrs = g.get("addrs") or ["00000000"]
                # insert_entry only takes ONE address per call - a multi-
                # address group (`0xA | 0xB => Some(fn)`) needs one call per
                # address, not just the first (a real bug caught before this
                # ever ran: the first version silently dropped every address
                # after addrs[0] from the dispatch table).
                already = [a for a in addrs if shared.dispatch_editor.address_already_registered(table_content, a)]
                if already:
                    skipped.append({"key": key, "reason": f"endereço(s) {already} já registrado(s) na tabela de dispatch"})
                    continue
                new_content = existing.rstrip() + f"\n\n// {key} - drafted by hle-studio, not yet judge-reviewed\n{code}\n"
                layer_path.write_text(new_content, encoding="utf-8")
                written_files.append(str(layer_path))
                # NOTE: this produces N separate single-address match arms
                # for a multi-address group, not one consolidated
                # `0xA | 0xB => Some(fn)` arm (DispatchEditor.insert_entry's
                # real signature only takes one address per call) - still
                # correct Rust, just not the terser style the codebase
                # prefers for genuine near-duplicates; consolidating is
                # real follow-up work for DispatchEditor itself, not solved
                # here.
                for a in addrs:
                    table_content = shared.dispatch_editor.insert_entry(table_content, a, layer, fn_name)
                dispatch_inserted.append(g["dispatch_entry"])

        if dispatch_inserted:
            table_path.write_text(table_content, encoding="utf-8")
            written_files.append(str(table_path))

        build_ok, build_tail = shared.build_runner.build()
        test_ok, test_tail = (False, "(build falhou, teste não rodou)")
        if build_ok:
            test_ok, test_tail = shared.build_runner.test()

        ctx.state.summary["write_and_verify"] = {
            "written_files": written_files,
            "skipped": skipped,
            "dispatch_inserted": dispatch_inserted,
            "build_ok": build_ok,
            "test_ok": test_ok,
            "build_tail": build_tail,
            "test_tail": test_tail,
        }
        print(f"[WriteAndVerify] {len(dispatch_inserted)} função(ões) escrita(s) em {len(written_files)} arquivo(s), "
              f"{len(skipped)} pulada(s). build={'OK' if build_ok else 'FALHOU'}, "
              f"test={'OK' if test_ok else 'FALHOU'}")
        if not build_ok:
            print(f"[WriteAndVerify] cauda do build:\n{build_tail}")
        elif not test_ok:
            print(f"[WriteAndVerify] cauda do teste:\n{test_tail}")
        return End(ctx.state.summary)


draft_graph = Graph(nodes=[SelectTargets, MechanicalDisasm, BuildDependencyGraph, ClusterPseudocode, PolishWithLLM,
                            WriteOutput, WriteAndVerify])
