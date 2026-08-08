"""Builds the porting agent with its full toolset. Every tool here reaches
project specifics ONLY through `PortState.shared.config`/`.dispatch_editor`/
`.extractor`/`.build_runner` - never a hardcoded path or a language-specific
regex directly. This is the mechanical guarantee that a new (isa, language)
plugin is the only thing that has to change to retarget this whole file.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess

from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
import httpx

from hle_studio.config import TargetConfig
from hle_studio.graph import PortState, addresses_already_ported, graph, CheckAlreadyPorted


def _read_playbook(config: TargetConfig) -> str:
    if config.playbook_file and config.playbook_file.exists():
        return config.playbook_file.read_text(encoding="utf-8")
    return "(no playbook file configured for this target - proceed with generic care)"


def _read_docs(config: TargetConfig) -> str:
    parts = []
    for doc in (config.idiom_table_doc, config.memory_map_doc):
        if doc and doc.exists():
            parts.append(f"--- {doc.name} ---\n{doc.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


SYSTEM_PROMPT_TEMPLATE = """\
# Role
You port ONE already-studied {isa} function into real, compiling, tested \
{output_language} code, for the "{target_name}" static recompilation \
project. You are given exactly one assigned target address per turn by the \
process driving you - you do not choose which one, and you do not browse \
other studies to find an easier one instead. If the assigned target \
genuinely depends on a function that isn't ported yet, use \
port_dependency_first(addr_hex) to actually go port that one first, then \
come back and finish the assigned target.

# The process you must follow (verbatim, this project's own playbook)

{playbook}

# Reference docs for this target's ISA/memory model

{docs}

# Tools available to you
- read_study(addr_hex): read a specific study's full content, by hex \
address (no "0x" prefix). Use this for your assigned target AND for any \
dependency you need to chase.
- list_imported_addresses(): every address already in the dispatch table.
- read_layer_file(layer): read one of the layer files in full. Valid \
layers: {layers}.
- read_test_file(): read the current test file in full - ALWAYS do this \
before appending a new test, to avoid duplicating a name.
- grep_native_functions(pattern): grep across the native-functions dir for \
`pattern` - find a similar already-ported function to model your \
translation on.
- append_function_to_layer(layer, code): append one complete function item \
(with its doc comment) to the given layer file. Refuses if a function with \
that name already exists - use replace_function_in_layer instead.
- replace_function_in_layer(layer, function_name, new_code): replace an \
existing function in place, matched by name - use this to FIX a mistake \
you already wrote, never by appending a second attempt.
- register_dispatch_entry(addr_hex, layer, function_name): register the \
address in the dispatch table. Refuses if already registered.
- append_test(code): append one complete test item to the test file. \
Refuses if a test with that name already exists.
- replace_test(function_name, new_code): replace an existing test in \
place, matched by name.
- port_dependency_first(addr_hex): if your assigned target depends on a \
function that isn't ported yet, call this - it recursively runs this \
exact same process for the dependency and returns a report. Refuses (with \
a clear reason) instead of recursing into a cycle or exceeding max depth - \
in that case, stop and report the situation, don't fake the call.
- grep_reference_source(pattern): grep the target's reference/ground-truth \
source tree(s) for `pattern` to confirm real hardware/ISA semantics.
- run_build(): runs the real build command, returns real output.
- run_test(): runs the real test command, returns real output. Run this \
before declaring anything done.

# What "done" means for this turn
1. The function(s) needed are written via append_function_to_layer (or \
fixed in place via replace_function_in_layer, never a second attempt next \
to a broken first one).
2. Each is registered via register_dispatch_entry.
3. Each has a real test via append_test/replace_test - or, if you \
genuinely cannot compute an expected value safely by hand, say so \
explicitly in your final report instead of silently skipping the test.
4. run_build() and run_test() both come back clean - if either doesn't, \
fix the problem in place before finishing, don't report success anyway.
5. Your final message is a short report: which address(es) you ported, \
which dependency (if any) you chased and what that returned, and the real \
tail of the test output you just ran.

If, after genuinely trying, the assigned target really cannot be ported \
this turn, say so explicitly and name exactly why - do not silently move \
on, and do not fabricate a working-looking implementation that doesn't \
really do what the study says.
"""


def _layer_path(config: TargetConfig, layer: str) -> str:
    if layer not in config.layer_names():
        raise ValueError(f"unknown layer {layer!r} - must be one of {config.layer_names()}")
    return str(config.native_functions_dir / f"{layer}.{config.layer_file_extension}")


def build_agent(model: str, base_url: str, api_key: str, config: TargetConfig,
                 reasoning_effort: str | None = None) -> Agent:
    REQUEST_TIMEOUT = httpx.Timeout(connect=30.0, read=1800.0, write=60.0, pool=30.0)
    client = AsyncOpenAI(
        base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT,
        http_client=httpx.AsyncClient(timeout=REQUEST_TIMEOUT),
    )
    llm = OpenAIChatModel(model, provider=OpenAIProvider(openai_client=client))
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        isa=config.isa, output_language=config.output_language, target_name=config.name,
        playbook=_read_playbook(config), docs=_read_docs(config),
        layers=", ".join(config.layer_names()),
    )
    agent_kwargs = dict(deps_type=PortState, system_prompt=system_prompt, retries=2, tool_timeout=180.0)
    if reasoning_effort is not None:
        agent_kwargs["model_settings"] = OpenAIChatModelSettings(openai_reasoning_effort=reasoning_effort)
    agent = Agent(llm, **agent_kwargs)

    def _study_path(addr_hex: str) -> str | None:
        addr = addr_hex.upper()
        for path in glob.glob(str(config.project_root / config.studies_glob)):
            m = re.match(config.study_filename_regex, os.path.basename(path))
            if m and m.group(1).upper() == addr:
                return path
        return None

    @agent.tool
    def read_study(ctx: RunContext[PortState], addr_hex: str) -> str:
        """Read a study's full content by hex address (no 0x prefix)."""
        path = _study_path(addr_hex)
        if not path:
            return f"(no study file found for 0x{addr_hex.upper()})"
        return open(path, encoding="utf-8").read()

    @agent.tool
    def list_imported_addresses(ctx: RunContext[PortState]) -> str:
        """Every address currently registered in the dispatch table, one per line."""
        addrs = sorted(addresses_already_ported(ctx.deps.shared))
        return "\n".join(addrs) if addrs else "(none found)"

    @agent.tool
    def read_layer_file(ctx: RunContext[PortState], layer: str) -> str:
        """Read one of the layer files in full."""
        try:
            path = _layer_path(config, layer)
        except ValueError as e:
            return str(e)
        return open(path, encoding="utf-8").read() if os.path.exists(path) else "(empty)"

    @agent.tool
    def read_test_file(ctx: RunContext[PortState]) -> str:
        """Read the current test file in full - do this before appending a new test."""
        return config.test_file.read_text(encoding="utf-8") if config.test_file.exists() else "(empty)"

    @agent.tool
    def grep_native_functions(ctx: RunContext[PortState], pattern: str) -> str:
        """Grep across the native-functions dir for `pattern`."""
        try:
            out = subprocess.run(["grep", "-rn", pattern, str(config.native_functions_dir)],
                                  capture_output=True, text=True, timeout=15)
            lines = (out.stdout or "").splitlines()[:60]
            return "\n".join(lines) if lines else f"(no results for {pattern!r})"
        except Exception as e:  # noqa: BLE001
            return f"(error running grep: {e})"

    @agent.tool
    async def append_function_to_layer(ctx: RunContext[PortState], layer: str, code: str) -> str:
        """Append one complete function item (with doc comment) to the given
        layer file. Refuses if a function with that name already exists."""
        try:
            path = _layer_path(config, layer)
        except ValueError as e:
            return str(e)
        fn_name = ctx.deps.shared.extractor.extract_fn_name(code)
        if not fn_name:
            return "couldn't find a function name in code - is this a complete item?"
        async with ctx.deps.shared.write_lock:
            existing = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
            if ctx.deps.shared.extractor.fn_name_in(existing, fn_name):
                return (f"a function named {fn_name!r} already exists in {layer} - use "
                         "replace_function_in_layer instead of appending another attempt")
            with open(path, "a", encoding="utf-8") as f:
                if existing and not code.startswith("\n"):
                    f.write("\n")
                f.write(code.rstrip() + "\n")
        return f"appended {fn_name} to {layer} - now run_build() to check it compiles"

    @agent.tool
    async def replace_function_in_layer(ctx: RunContext[PortState], layer: str, function_name: str, new_code: str) -> str:
        """Replace an existing function in place, matched by name."""
        try:
            path = _layer_path(config, layer)
        except ValueError as e:
            return str(e)
        async with ctx.deps.shared.write_lock:
            content = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
            span = ctx.deps.shared.extractor.extract_fn_body(content, function_name)
            if span is None:
                return f"{function_name!r} not found in {layer} - use append_function_to_layer if this is new"
            start, end = span
            content = content[:start] + new_code.strip() + "\n" + content[end:]
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return f"replaced {function_name} in {layer} - now run_build() to check it compiles"

    @agent.tool
    async def register_dispatch_entry(ctx: RunContext[PortState], addr_hex: str, layer: str, function_name: str) -> str:
        """Register the address in the dispatch table. Refuses if already registered."""
        if layer not in config.layer_names():
            return f"unknown layer {layer!r} - must be one of {config.layer_names()}"
        table_path = config.dispatch_table_file
        async with ctx.deps.shared.write_lock:
            content = table_path.read_text(encoding="utf-8") if table_path.exists() else ""
            editor = ctx.deps.shared.dispatch_editor
            if editor.address_already_registered(content, addr_hex):
                return f"0x{addr_hex.upper()} already registered - check list_imported_addresses(), don't double-register"
            content = editor.insert_entry(content, addr_hex, layer, function_name)
            table_path.write_text(content, encoding="utf-8")
        return f"registered 0x{addr_hex.upper()} -> {layer}::{function_name}"

    @agent.tool
    async def append_test(ctx: RunContext[PortState], code: str) -> str:
        """Append one complete test item to the test file. Refuses if a
        test with that name already exists."""
        fn_name = ctx.deps.shared.extractor.extract_fn_name(code)
        if not fn_name:
            return "couldn't find a test function name in code"
        async with ctx.deps.shared.write_lock:
            existing = config.test_file.read_text(encoding="utf-8") if config.test_file.exists() else ""
            if ctx.deps.shared.extractor.fn_name_in(existing, fn_name):
                return f"a test named {fn_name!r} already exists - use replace_test instead"
            with open(config.test_file, "a", encoding="utf-8") as f:
                if existing and not code.startswith("\n"):
                    f.write("\n")
                f.write(code.rstrip() + "\n")
        return f"appended test {fn_name} - now run_test() to check it passes"

    @agent.tool
    async def replace_test(ctx: RunContext[PortState], function_name: str, new_code: str) -> str:
        """Replace an existing test in place, matched by name."""
        async with ctx.deps.shared.write_lock:
            content = config.test_file.read_text(encoding="utf-8") if config.test_file.exists() else ""
            span = ctx.deps.shared.extractor.extract_fn_body(content, function_name)
            if span is None:
                return f"{function_name!r} not found in test file - use append_test if this is new"
            start, end = span
            content = content[:start] + new_code.strip() + "\n" + content[end:]
            config.test_file.write_text(content, encoding="utf-8")
        return f"replaced test {function_name} - now run_test() to check it passes"

    @agent.tool
    async def port_dependency_first(ctx: RunContext[PortState], addr_hex: str) -> str:
        """Port a dependency FIRST, recursively, before continuing your own target."""
        dep = addr_hex.upper()
        shared = ctx.deps.shared
        if dep in addresses_already_ported(shared):
            return f"0x{dep} is already ported - just call it directly, no need to recurse"
        if dep in shared.in_progress:
            return (f"0x{dep} is already being ported somewhere in this same chain - this is a "
                     "cycle (or duplicate chase) - do NOT recurse again, report this instead")
        from hle_studio.graph import MAX_DEPENDENCY_DEPTH
        if ctx.deps.depth + 1 > MAX_DEPENDENCY_DEPTH:
            return f"dependency depth would exceed {MAX_DEPENDENCY_DEPTH} - stop, report the chain so far"
        sub_state = PortState(addr_hex=dep, depth=ctx.deps.depth + 1, shared=shared)
        end = await graph.run(CheckAlreadyPorted(), state=sub_state)
        return f"port_dependency_first(0x{dep}) result:\n{end.output}"

    @agent.tool
    def grep_reference_source(ctx: RunContext[PortState], pattern: str) -> str:
        """Grep the target's reference/ground-truth source tree(s)."""
        for root in config.reference_source_roots:
            if not root.exists():
                continue
            out = subprocess.run(["grep", "-rn", pattern, str(root)], capture_output=True, text=True, timeout=15)
            lines = (out.stdout or "").splitlines()[:30]
            if lines:
                return "\n".join(lines)
        return f"(no results for {pattern!r} in any reference source root)"

    @agent.tool
    async def run_build(ctx: RunContext[PortState]) -> str:
        """Run the real build command."""
        async with ctx.deps.shared.write_lock:
            ok, out = ctx.deps.shared.build_runner.build()
        return out

    @agent.tool
    async def run_test(ctx: RunContext[PortState]) -> str:
        """Run the real test command."""
        async with ctx.deps.shared.write_lock:
            ok, out = ctx.deps.shared.build_runner.test()
        return out

    return agent
