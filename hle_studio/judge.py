"""The separate, cheap, different-model-family judge - reviews only a diff
of newly added code against a fixed checklist, with zero project/domain
context. Deliberately not the same model/agent doing the porting (checking
your own homework catches less than a second, less-correlated pair of
eyes).

`JudgeFinding` is structured on purpose (rule + what_is_wrong + snippet as
THREE separate fields, not one free-text string) - a real run of the
originating pipeline found the fixing agent misreading a bare quoted-line
finding as "the doc comment is wrong" when the actual violation was the
function NAME in that same line, and it "fixed" the wrong thing three times
in a row without ever touching the real problem. Separating "which rule"
from "the snippet" removes that specific ambiguity.
"""
from __future__ import annotations

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

GENERIC_JUDGE_RULES = """\
1. RAW POINTERS: `as *mut`/`as *const` on what looks like a target-console \
memory address, or `.write_volatile(`/`.read_volatile(` anywhere. Forbidden \
- must go through the project's own memory abstraction (a method taking a \
plain address value, never a real host pointer).
2. FAKE DELEGATION: a function whose doc comment/surrounding text claims it \
calls or delegates to another function, but the body doesn't actually call \
anything and just returns a plausible-looking value instead.
3. SELF-CONSISTENT TESTS: a test whose expected value is computed by \
literally re-deriving the SAME formula/algorithm the function under test \
implements, instead of an independently pre-computed constant with the \
arithmetic explained in a plain comment. Such a test would still pass even \
if the implementation had a consistent-but-wrong bug - that's the finding.
4. TESTS THAT DON'T ASSERT ANYTHING MEANINGFUL: a test item with no real \
assertion at all, or one that only checks the function didn't crash.
5. SILENT FALLBACK HIDING A REAL ERROR: a default-value fallback on a \
read/write where a genuine failure (bad address, corrupt state) should be a \
loud error/panic instead of silently substituting a default that's \
indistinguishable from a real result. Only flag a NEW/worse instance in the \
code you're reviewing, not a blanket objection if this pattern is already a \
widely-used, accepted convention elsewhere in the same codebase.
6. DUPLICATE DEFINITIONS: the SAME function or test name defined more than \
once in the blob you were given.
"""

JUDGE_SYSTEM_PROMPT_TEMPLATE = """\
You are a mechanical code reviewer for a {isa}-to-{output_language} static \
recompilation project. You do NOT need to understand what the target \
program does, and you should NOT comment on style, naming taste, or \
whether the logic seems plausible - you have no context for that and must \
not guess. You are checking ONLY for a fixed list of mistakes.

{ruleset}

For each finding, fill in all three fields SEPARATELY - a downstream fixing \
agent only sees these fields, not your reasoning, so be precise about WHAT \
is wrong, not just WHERE:
- rule: which numbered rule above, e.g. "1 (raw pointer)".
- what_is_wrong: one specific sentence naming exactly what's wrong - never \
something vague like "this line has a problem".
- snippet: the exact offending line, quoted as-is.

If you find NONE of the above, set approved=true with an empty findings \
list. Do not invent a finding to have something to say.
"""


class JudgeFinding(BaseModel):
    rule: str = Field(description="Which checklist rule, by number and short name.")
    what_is_wrong: str = Field(description="One specific sentence: exactly what violates the rule.")
    snippet: str = Field(description="The exact offending line/snippet.")


class JudgeVerdict(BaseModel):
    approved: bool = Field(description="True only if NONE of the checklist items were found.")
    findings: list[JudgeFinding] = Field(default_factory=list)


def build_judge_agent(
    model: str, base_url: str, api_key: str,
    isa: str, output_language: str,
    project_rules: str = "",
) -> Agent:
    """project_rules: extra, project-specific checklist rules appended after
    the generic ones (e.g. Saturn's "address embedded in function name")."""
    ruleset = GENERIC_JUDGE_RULES
    if project_rules.strip():
        ruleset += "\n" + project_rules.strip() + "\n"
    system_prompt = JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
        isa=isa, output_language=output_language, ruleset=ruleset,
    )
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=httpx.Timeout(120.0))
    llm = OpenAIChatModel(model, provider=OpenAIProvider(openai_client=client))
    return Agent(llm, output_type=JudgeVerdict, system_prompt=system_prompt, retries=2)
