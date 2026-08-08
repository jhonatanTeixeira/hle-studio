"""CLI entrypoint. `hle-studio port <hle_studio_target.toml> [--limit N] [--workers N]`

Concurrency model unchanged from the originating project: N workers run the
network-bound agent turns in parallel, but every file-mutating tool call and
every real build/test verification serializes on one `write_lock` (Group
Commit shape) - see docs in graph.py/agent_tools.py.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from hle_studio.agent_tools import build_agent
from hle_studio.config import TargetConfig
from hle_studio.graph import CheckAlreadyPorted, PortState, SharedPortContext, graph, next_unported_targets
from hle_studio.judge import build_judge_agent
from hle_studio.plugins.rust_cargo import CargoBuildRunner, RustMatchTableDispatchEditor, RustRegexFunctionExtractor

MAX_WORKERS = 5

PLUGIN_REGISTRY = {
    ("rust-match-table", "rust-regex", "rust"): (
        RustMatchTableDispatchEditor, RustRegexFunctionExtractor, CargoBuildRunner,
    ),
}

# (isa, output_language) -> (TraceRanker, MechanicalTranslator) for `draft`.
# Separate registry from PLUGIN_REGISTRY above: `draft` only needs these two,
# never DispatchEditor/FunctionExtractor/BuildRunner (no build/test gate in
# the draft pipeline - see draft_graph.py's module docstring for why).
DRAFT_PLUGIN_REGISTRY = {}


def _register_draft_plugins():
    # Imported lazily, not at module load: importing plugins.sh2_rust pulls
    # in the vendored sh2dis/cfg_walk/sh2_to_rust_pseudo modules, which is
    # unnecessary weight for anyone only ever running `hle-studio port`.
    from hle_studio.plugins.sh2_rust import Sh2RustMechanicalTranslator, Sh2RustTraceRanker
    DRAFT_PLUGIN_REGISTRY[("sh2", "rust")] = (Sh2RustTraceRanker, Sh2RustMechanicalTranslator)


def _resolve_plugins(config: TargetConfig):
    key = (config.dispatch_strategy, config.extractor_strategy, config.output_language)
    if key not in PLUGIN_REGISTRY:
        raise ValueError(
            f"no plugin registered for (dispatch={config.dispatch_strategy!r}, "
            f"extractor={config.extractor_strategy!r}, language={config.output_language!r}) - "
            f"known combinations: {list(PLUGIN_REGISTRY)}"
        )
    dispatch_cls, extractor_cls, build_cls = PLUGIN_REGISTRY[key]
    return dispatch_cls(config), extractor_cls(), build_cls(config)


async def _worker(name: int, queue: asyncio.Queue, shared: SharedPortContext, results: list):
    while True:
        try:
            target = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        print(f"\n[worker {name}] === alvo 0x{target} ===")
        state = PortState(addr_hex=target, depth=0, shared=shared)
        end = await graph.run(CheckAlreadyPorted(), state=state)
        print(f"[worker {name}] {end.output}")
        results.append((target, end.output.startswith(f"0x{target}: CONFIRMADO")))
        queue.task_done()


async def _run_port(args):
    config = TargetConfig.load(args.target_config)
    print(f"target: {config.name}  isa: {config.isa}  language: {config.output_language}")
    print(f"endpoint: {args.base_url}  model: {args.model}  juiz: {args.judge_model}  "
          f"workers: {min(args.workers, MAX_WORKERS)}")

    dispatch_editor, extractor, build_runner = _resolve_plugins(config)
    agent = build_agent(args.model, args.base_url, args.api_key, config, args.reasoning_effort)
    project_rules = ""
    if config.judge_project_rules_file and config.judge_project_rules_file.exists():
        project_rules = config.judge_project_rules_file.read_text(encoding="utf-8")
    judge = build_judge_agent(
        args.judge_model, args.judge_base_url, args.judge_api_key,
        config.isa, config.output_language, project_rules,
    )
    shared = SharedPortContext(
        write_lock=asyncio.Lock(), agent=agent, judge=judge, config=config,
        dispatch_editor=dispatch_editor, extractor=extractor, build_runner=build_runner,
    )
    n_workers = min(args.workers, MAX_WORKERS)
    results: list[tuple[str, bool]] = []

    while True:
        if args.limit and len(results) >= args.limit:
            print(f"\nlimite de {args.limit} atingido, parando.")
            break
        pending = next_unported_targets(shared)
        if not pending:
            print("\nnenhum estudo pendente restante.")
            break
        round_size = n_workers
        if args.limit:
            round_size = min(round_size, args.limit - len(results))
        batch = pending[:round_size]
        print(f"\n=== rodada: {len(batch)} alvo(s) em paralelo: {', '.join('0x' + a for a in batch)} ===")
        queue: asyncio.Queue = asyncio.Queue()
        for t in batch:
            queue.put_nowait(t)
        workers = [asyncio.create_task(_worker(i, queue, shared, results)) for i in range(len(batch))]
        await queue.join()
        for w in workers:
            w.cancel()

    ok_count = sum(1 for _, ok in results if ok)
    print(f"\nfeito: {ok_count}/{len(results)} alvos confirmados portados de verdade")


def _resolve_draft_plugins(config: TargetConfig):
    _register_draft_plugins()
    key = (config.isa, config.output_language)
    if key not in DRAFT_PLUGIN_REGISTRY:
        raise ValueError(f"no draft plugin registered for (isa={config.isa!r}, "
                          f"language={config.output_language!r}) - known: {list(DRAFT_PLUGIN_REGISTRY)}")
    ranker_cls, translator_cls = DRAFT_PLUGIN_REGISTRY[key]
    return ranker_cls(), translator_cls(config)


async def _run_draft(args):
    from hle_studio.draft_graph import DraftSharedContext, DraftState, SelectTargets, draft_graph

    config = TargetConfig.load(args.target_config)
    print(f"target: {config.name}  isa: {config.isa}  language: {config.output_language}  top: {args.top}")
    print(f"endpoint: {args.base_url}  model: {args.model}  clustering: {config.cluster_model} "
          f"(threshold={config.cluster_threshold})")
    trace_ranker, translator = _resolve_draft_plugins(config)
    shared = DraftSharedContext(
        config=config, trace_ranker=trace_ranker, translator=translator,
        llm_base_url=args.base_url, llm_model_alias=args.model, llm_api_key=args.api_key, top_n=args.top,
    )
    state = DraftState(shared=shared)
    end = await draft_graph.run(SelectTargets(), state=state)
    result = end.output

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "draft_code.txt"), "w", encoding="utf-8") as f:
            f.write(result["code"] + "\n")
        with open(os.path.join(args.out_dir, "draft_dispatch_entries.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result["dispatch_entries"]) + "\n")
        print(f"\ndraft escrito em {args.out_dir}/")


def main():
    parser = argparse.ArgumentParser(prog="hle-studio")
    sub = parser.add_subparsers(dest="command", required=True)

    port_p = sub.add_parser("port", help="port already-studied functions into real target-language code")
    port_p.add_argument("target_config", help="path to this target's hle_studio_target.toml")
    port_p.add_argument("--limit", type=int, default=None)
    port_p.add_argument("--workers", type=int, default=5)
    port_p.add_argument("--model", type=str, default=os.environ.get("HLE_MODEL", "porting-agent-model"))
    port_p.add_argument("--base-url", type=str, default=os.environ.get("HLE_BASE_URL", "http://localhost:8080/v1"))
    port_p.add_argument("--api-key", type=str, default=os.environ.get("HLE_API_KEY", "placeholder"))
    port_p.add_argument("--judge-model", type=str, default=os.environ.get("HLE_JUDGE_MODEL", "judge-model"))
    port_p.add_argument("--judge-base-url", type=str, default=os.environ.get("HLE_JUDGE_BASE_URL", "http://localhost:8080/v1"))
    port_p.add_argument("--judge-api-key", type=str, default=os.environ.get("HLE_JUDGE_API_KEY", "placeholder"))
    port_p.add_argument("--reasoning-effort", type=str, default=None,
                         choices=["none", "low", "medium", "high", "max"])

    draft_p = sub.add_parser("draft", help="raw trace -> mechanical pseudocode -> semantic clustering -> "
                                            "LLM-polished first-draft code (no judge, no build/test gate)")
    draft_p.add_argument("target_config", help="path to this target's hle_studio_target.toml (needs [trace])")
    draft_p.add_argument("--top", type=int, default=20, help="how many top-frequency call targets to draft")
    draft_p.add_argument("--model", type=str, default=os.environ.get("HLE_DRAFT_MODEL", "porting-agent-model"))
    draft_p.add_argument("--base-url", type=str, default=os.environ.get("HLE_DRAFT_BASE_URL", "http://localhost:8080/v1"))
    draft_p.add_argument("--api-key", type=str, default=os.environ.get("HLE_DRAFT_API_KEY", "placeholder"))
    draft_p.add_argument("--out-dir", type=str, default=None)

    args = parser.parse_args()
    if args.command == "port":
        asyncio.run(_run_port(args))
    elif args.command == "draft":
        asyncio.run(_run_draft(args))
    else:  # pragma: no cover
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
