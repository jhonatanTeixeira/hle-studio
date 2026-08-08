"""Loads a target project's `hle_studio_target.toml` - the ONE file that
turns the generic graph/agent/judge machinery in this package into something
that actually knows how to port functions for a specific (ISA, output
language) pair, in a specific project's directory layout.

Design intent (see docs/decompilation_architecture-equivalent decisions in
the originating project): every Saturn/SH-2/Rust-specific value that used to
be a Python module-level constant in `tools/agent_study_all.py` lives here
as config instead - the graph/agent code in this package must never import
anything Saturn-specific directly.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - this package requires 3.11+ per pyproject.toml
    import tomli as tomllib  # type: ignore[import-not-found]


@dataclass
class LayerSpec:
    name: str
    title: str = ""


@dataclass
class TargetConfig:
    """Everything a run of hle-studio needs to know about ONE target
    project, loaded from that project's own `hle_studio_target.toml`."""

    # [target]
    name: str
    isa: str
    output_language: str
    description: str = ""

    # [paths] - all resolved to absolute paths against project_root
    project_root: Path = field(default_factory=Path)
    native_functions_dir: Path = field(default_factory=Path)
    layer_file_extension: str = "rs"
    test_file: Path = field(default_factory=Path)
    studies_glob: str = "traces/calls/*_study.md"
    study_ready_marker: str = "## Interpretação"
    study_filename_regex: str = r"^([0-9A-Fa-f]{8})_study\.md$"
    reference_source_roots: list[Path] = field(default_factory=list)

    # [layers]
    layers: list[LayerSpec] = field(default_factory=list)

    # [build]
    build_cmd: list[str] = field(default_factory=lambda: ["true"])
    test_cmd: list[str] = field(default_factory=lambda: ["true"])
    build_cwd: Path = field(default_factory=Path)

    # [dispatch]
    dispatch_strategy: str = "rust-match-table"
    # For the "rust-match-table" strategy: must cover the FULL text through
    # the opening of the actual match block, e.g.
    # "pub fn lookup(addr: u32) -> Option<NativeFn> {\n    match addr {" -
    # NOT just the function signature's own brace. A signature-only anchor
    # ends in "{" itself, so a naive "find the next {" search matches that
    # same brace and inserts BEFORE the match block instead of inside it
    # (a real bug caught while porting this exact class - see
    # RustMatchTableDispatchEditor.insert_entry's comment).
    dispatch_anchor: str = ""
    dispatch_entry_template: str = "0x{addr} => Some({fn_ref}),"
    dispatch_table_file: Path = field(default_factory=Path)

    # [extractor]
    extractor_strategy: str = "rust-regex"

    # [judge]
    judge_base_ruleset: str = "generic"
    judge_project_rules_file: Path | None = None

    # [docs]
    idiom_table_doc: Path | None = None
    memory_map_doc: Path | None = None
    playbook_file: Path | None = None

    # [llm] - agentgateway is the intended transport; these are just the
    # aliases/URL this project expects, never a provider-specific key. Also
    # doubles as the endpoint for `draft`'s PolishWithLLM step
    # (draft_alias, defaulting to porting_model_alias) - the SAME endpoint
    # works unchanged whether it's an agentgateway route, a raw provider, or
    # a local OpenAI-compatible server (e.g. july-engine) - validated live
    # against the latter (2026-08-07 case study, see draft_graph.py).
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key_env: str = "AGENTGATEWAY_API_KEY"
    porting_model_alias: str = "porting-agent-model"
    judge_model_alias: str = "judge-model"
    draft_model_alias: str = ""  # empty -> falls back to porting_model_alias, see load()

    # [trace] - the `draft` pipeline's input: a raw, target-specific
    # execution trace (see plugins.base.TraceRanker). Optional - only needed
    # for `hle-studio draft`, not for `hle-studio port`.
    raw_trace_path: Path | None = None
    trace_ranker_strategy: str = ""
    mechanical_translator_strategy: str = ""
    # The actual target binary/binaries MechanicalTranslator decodes opcodes
    # from - deliberately separate from [paths].reference_source_roots
    # (that's C/C++ REFERENCE source for the porting agent's own tools, a
    # different thing entirely; reusing it for the binary path was a real
    # naming collision caught before this ever shipped).
    game_binary_path: Path | None = None
    game_binary_secondary_path: Path | None = None  # e.g. Saturn's IP.BIN - optional

    # [clustering] - see semantic_clustering.py for why these specific
    # defaults (not arbitrary - measured against a real corpus).
    cluster_model: str = "microsoft/graphcodebert-base"
    cluster_threshold: float = 0.55

    @classmethod
    def load(cls, toml_path: str | Path) -> "TargetConfig":
        toml_path = Path(toml_path).resolve()
        root = toml_path.parent
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        target = data.get("target", {})
        paths = data.get("paths", {})
        layers = data.get("layers", {})
        build = data.get("build", {})
        dispatch = data.get("dispatch", {})
        extractor = data.get("extractor", {})
        judge = data.get("judge", {})
        docs = data.get("docs", {})
        llm = data.get("llm", {})
        trace = data.get("trace", {})
        clustering = data.get("clustering", {})

        layer_order = layers.get("order", [])
        layer_titles = layers.get("titles", {})
        layer_specs = [LayerSpec(name=n, title=layer_titles.get(n, n)) for n in layer_order]

        def _p(rel: str | None) -> Path | None:
            return (root / rel).resolve() if rel else None

        return cls(
            name=target.get("name", "unnamed-target"),
            isa=target.get("isa", "unknown"),
            output_language=target.get("output_language", "unknown"),
            description=target.get("description", ""),
            project_root=root,
            native_functions_dir=_p(paths.get("native_functions_dir")) or root,
            layer_file_extension=paths.get("layer_file_extension", "rs"),
            test_file=_p(paths.get("test_file")) or root,
            studies_glob=paths.get("studies_glob", "traces/calls/*_study.md"),
            study_ready_marker=paths.get("study_ready_marker", "## Interpretação"),
            study_filename_regex=paths.get("study_filename_regex", r"^([0-9A-Fa-f]{8})_study\.md$"),
            reference_source_roots=[p for p in (
                _p(r) for r in paths.get("reference_source_roots", [])
            ) if p is not None],
            layers=layer_specs,
            build_cmd=build.get("build_cmd", ["true"]),
            test_cmd=build.get("test_cmd", ["true"]),
            build_cwd=_p(build.get("build_cwd", ".")) or root,
            dispatch_strategy=dispatch.get("strategy", "rust-match-table"),
            dispatch_anchor=dispatch.get("anchor_string", ""),
            dispatch_entry_template=dispatch.get("entry_template", "0x{addr} => Some({fn_ref}),"),
            dispatch_table_file=_p(dispatch.get("table_file")) or root,
            extractor_strategy=extractor.get("strategy", "rust-regex"),
            judge_base_ruleset=judge.get("base_ruleset", "generic"),
            judge_project_rules_file=_p(judge.get("project_rules_file")),
            idiom_table_doc=_p(docs.get("idiom_table_doc")),
            memory_map_doc=_p(docs.get("memory_map_doc")),
            playbook_file=_p(docs.get("playbook_file")),
            llm_base_url=llm.get("agentgateway_base_url", "http://localhost:8080/v1"),
            llm_api_key_env=llm.get("api_key_env", "AGENTGATEWAY_API_KEY"),
            porting_model_alias=llm.get("porting_model_alias", "porting-agent-model"),
            judge_model_alias=llm.get("judge_model_alias", "judge-model"),
            draft_model_alias=llm.get("draft_model_alias", "") or llm.get("porting_model_alias", "porting-agent-model"),
            raw_trace_path=_p(trace.get("raw_trace_path")),
            trace_ranker_strategy=trace.get("ranker_strategy", ""),
            mechanical_translator_strategy=trace.get("mechanical_translator_strategy", ""),
            game_binary_path=_p(trace.get("game_binary_path")),
            game_binary_secondary_path=_p(trace.get("game_binary_secondary_path")),
            cluster_model=clustering.get("model", "microsoft/graphcodebert-base"),
            cluster_threshold=clustering.get("threshold", 0.55),
        )

    def layer_names(self) -> list[str]:
        return [l.name for l in self.layers]

    def layer_title(self, name: str) -> str:
        for l in self.layers:
            if l.name == name:
                return l.title
        return name
