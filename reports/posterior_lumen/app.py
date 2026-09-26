"""Shared pieces of the two Lumen apps: the local LLM, the warehouse source and agent prompts."""

import difflib
import functools
import inspect
import os
import re
from typing import Annotated, Literal

import param

import duckdb
from pydantic import BeforeValidator
import lumen.ai as lmai
from lumen.ai.agents import AnalysisAgent, ChatAgent, SQLAgent, TableListAgent, ValidationAgent, VegaLiteAgent
from lumen.sources.duckdb import DuckDBSource

from .analyses import ANALYSES
from .catalog import METADATA, WAREHOUSE
from lumen.ai.agents import sql as lumen_sql
from lumen.ai.tools import FunctionTool

# gemma4:e2b-it-qat is the smallest Gemma 4 (4.3 GB) and fits beside the app in 8 GB of RAM.
# gemma4:e4b-it-qat (6.1 GB) follows instructions noticeably better if you have the memory.
DEFAULT_MODEL = "gemma4:e4b-it-qat"
OLLAMA_URL = "http://localhost:11434/v1"


class LocalOllama(lmai.llm.Ollama):
    """Ollama with Gemma's thinking switched off by default.

    Gemma 4 writes a long hidden reasoning trace before every answer. For Lumen's short,
    structured requests it adds 8-10x latency (about 8 s instead of 0.6 s per call on an
    M-series laptop) without better answers, so every request asks for no reasoning.
    """

    think = param.Boolean(default=False, doc="Let the model reason before answering.")

    async def run_client(self, model_spec, messages, **kwargs):
        if not self.think:
            kwargs.setdefault("reasoning_effort", "none")
        return await super().run_client(model_spec, messages, **kwargs)

    async def _run_tool_calls(self, tool_instances, tool_calls, tool_contexts, messages):
        """Run tool calls with forgiving arguments (see ``forgiving``); restore the tools after."""
        originals = {}
        for name, tool in tool_instances.items():
            fn = getattr(tool, "function", None)
            if callable(fn):
                originals[name] = fn
                tool.function = forgiving(fn, name)
        try:
            return await super()._run_tool_calls(tool_instances, tool_calls, tool_contexts, messages)
        finally:
            for name, fn in originals.items():
                tool_instances[name].function = fn


TOPICS = {
    "E33": "budget marketing mmm ads advertising spend customer customers meta google facebook channel market "
           "denmark sweden norway plan recommended cost euro eur",
    "E36": "mrp opinion state states agree agreement survey abortion employers insurance majority wyoming poll public",
    "E37": "heat heatwave seattle portland temperature hot hotter climate warming summer summers 100f 108f fahrenheit "
           "celsius record return period attribution",
    "E40": "election clinton trump electoral vote votes forecast polls 2016 president swing pennsylvania michigan "
           "wisconsin florida",
    "E41": "bmi growth child children boy boys centile percentile weight height age chart dutch",
}
_WORD = re.compile(r"[a-z0-9]+")


def _words(text):
    return set(_WORD.findall(text.lower().replace("°", "")))


def look_up_answer(question: str) -> str:
    """Answer a question about the example models' findings (a chance, how much, how many, which is
    cheapest or highest) from answers computed in advance from the posterior draws. Returns the matching
    model's headline probabilities and key facts; quote them, do not recompute them."""
    from .report import election_facts, growth_facts, heat_facts, mmm_facts, mrp_facts

    words = _words(question)
    scores = {m: len(words & _words(t)) for m, t in TOPICS.items()}
    for m in TOPICS:  # "E33" in the question wins outright
        if m.lower() in words:
            scores[m] += 10
    model_id = max(scores, key=scores.get)
    if scores[model_id] == 0:
        return ("No model matched. The models are: E33 marketing budget, E36 state opinion (MRP), "
                "E37 Seattle heatwave, E40 2016 election forecast, E41 child BMI growth charts.")
    facts_fn = {"E33": mmm_facts, "E36": mrp_facts, "E37": heat_facts, "E40": election_facts,
                "E41": growth_facts}[model_id]
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        facts = facts_fn(con)
        probs = con.execute("SELECT question, in_words, probability FROM key_probabilities WHERE model_id = ?",
                            [model_id]).fetchall()
        title, caveat = con.execute("SELECT title, caveat FROM models WHERE model_id = ?", [model_id]).fetchone()
    rank = lambda text: -len(words & _words(text))  # noqa: E731  most overlapping first
    lines = [f"Model {model_id}: {title}", "Probabilities (from the posterior draws):"]
    lines += [f"- {q} {w} ({p:.2f})" for q, w, p in sorted(probs, key=lambda r: rank(r[0]))]
    lines += ["Key facts:"] + [f"- {f}" for f in sorted(facts, key=rank)]
    lines.append(f"Caveat: {caveat}")
    return "\n".join(lines)


def _forgiving_table_model(sources):
    """Lumen's list of allowed table names, but a near-miss snaps to the closest real table.

    The SQL agent must name the tables its query uses from a fixed list. A 2B model often
    writes the name of its *output* there instead (``posterior_draw_analysis``), fails
    validation three times and gives up. The name is only bookkeeping (the SQL itself is what
    runs), so the closest real name is a safe correction.
    """
    tables = sorted({table for _, table in sources})

    def closest(value):
        if value in tables or not isinstance(value, str):
            return value
        bare = value.split("/")[-1].split(".")[-1].strip('"')
        match = difflib.get_close_matches(bare, tables, n=1, cutoff=0.0)
        return match[0] if match else value

    return Annotated[Literal[tuple(tables)], BeforeValidator(closest)]


lumen_sql.make_table_model = _forgiving_table_model


class StepTool(FunctionTool):
    """FunctionTool usable as a planner step: Lumen 1.3 passes the step's own keywords
    (``step_title``) to the function, which then fails; keep only the function's fields."""

    async def respond(self, messages, context, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in self._model.model_fields}
        return await super().respond(messages, context, **kwargs)


class PosteriorPlanner(lmai.Planner):
    """Lumen's Planner without the up-front "is this question ambiguous?" check by default.

    A 2B model answers that check "yes" even for specific questions and then waits for the
    reader to answer a clarifying question. POSTERIOR_CLARIFY=1 turns the check back on.
    """

    allow_clarification = param.Boolean(default=os.environ.get("POSTERIOR_CLARIFY", "0") == "1")

    async def _check_clarification_needed(self, messages, context):
        if not self.allow_clarification:
            return False
        return await super()._check_clarification_needed(messages, context)


def forgiving(fn, tool_name):
    """Wrap a tool function so a small model's near-miss call still works or fails gracefully.

    Small models call tools with invented argument names (``table_slug_list`` for
    ``table_slugs``) or a string where a list is expected. Lumen passes the arguments
    straight through, the call raises TypeError and, after three retries, the whole question
    fails. Here close-enough names are mapped onto the real parameters, lone strings are
    wrapped in lists, and any remaining error is returned to the model as text it can act on.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def fix(arguments):
        fixed = {k: v for k, v in arguments.items() if k in params or takes_kwargs}
        missing = [n for n, p in params.items() if n not in fixed and p.default is inspect.Parameter.empty
                   and p.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)]
        for key, value in arguments.items():
            if key in fixed or not missing:
                continue
            match = difflib.get_close_matches(key, missing, n=1, cutoff=0.5)
            target = match[0] if match else (missing[0] if len(missing) == 1 else None)
            if target:
                fixed[target] = value
                missing.remove(target)
        for n, p in params.items():  # a lone string where the parameter wants a list
            if n in fixed and isinstance(fixed[n], str) and "list" in str(p.annotation).lower():
                fixed[n] = [fixed[n]]
        return fixed

    def explain(exc):
        return (f"Tool error in {tool_name}: {exc}. Call it again with these arguments: "
                f"{', '.join(params)}.")

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def wrapper(**arguments):
            try:
                return await fn(**fix(arguments))
            except Exception as exc:
                return explain(exc)
    else:
        @functools.wraps(fn)
        def wrapper(**arguments):
            try:
                return fn(**fix(arguments))
            except Exception as exc:
                return explain(exc)
    return wrapper


def make_llm(model=None, endpoint=None):
    """An Ollama-served model for every Lumen task (Lumen uses Ollama's OpenAI-compatible API)."""
    model = model or os.environ.get("POSTERIOR_LLM", DEFAULT_MODEL)
    return LocalOllama(
        think=os.environ.get("POSTERIOR_THINK", "0") == "1",
        endpoint=endpoint or os.environ.get("OLLAMA_URL", OLLAMA_URL),
        api_key="ollama",  # required by the OpenAI client, ignored by Ollama
        model_kwargs={"default": {"model": model}},
    )


_TABLE_REF = re.compile(r'(\b(?:FROM|JOIN)\s+)("?)([A-Za-z_][\w]*)("?)', re.IGNORECASE)
_CTE_NAME = re.compile(r'(?:\bWITH\s+|,\s*)([A-Za-z_]\w*)\s+AS\s*\(', re.IGNORECASE)


def fix_table_typos(sql, known):
    """Correct near-miss table names after FROM/JOIN (``posterior_drawds`` -> ``posterior_draws``).

    A 2B model repeats the same misspelled table name on every retry. Names that are real,
    defined in a WITH clause, or not close to any real table (similarity < 0.85) are left alone.
    """
    ctes = {m.lower() for m in _CTE_NAME.findall(sql)}

    def swap(m):
        name = m.group(3)
        if name in known or name.lower() in ctes:
            return m.group(0)
        match = difflib.get_close_matches(name, known, n=1, cutoff=0.85)
        return f"{m.group(1)}{m.group(2)}{match[0]}{m.group(4)}" if match else m.group(0)

    return _TABLE_REF.sub(swap, sql)


class ForgivingDuckDBSource(DuckDBSource):
    """DuckDBSource that fixes near-miss table names in the SQL it is given."""

    def _known(self):
        return list(METADATA)

    def execute(self, sql_query, *args, **kwargs):
        return super().execute(fix_table_typos(sql_query, self._known()), *args, **kwargs)

    def create_sql_expr_source(self, tables, *args, **kwargs):
        tables = {name: fix_table_typos(sql, self._known()) for name, sql in tables.items()}
        return super().create_sql_expr_source(tables, *args, **kwargs)


def make_source():
    if not WAREHOUSE.exists():
        raise FileNotFoundError(f"{WAREHOUSE} not found: run `python reports/build_warehouse.py` first")
    return ForgivingDuckDBSource(uri=str(WAREHOUSE), tables=list(METADATA), metadata=METADATA,
                                 read_only=True, name="posteriors")


# What a small model most needs to be told: how to turn draws into probabilities,
# and which summaries must never be combined.
POSTERIOR_RULES = """
{{ super() }}

These tables hold the results of Bayesian models. Rules for posteriors:
- A probability is a share of posterior draws: SELECT AVG(CASE WHEN value > 0.5 THEN 1.0 ELSE 0.0 END)
  FROM posterior_draws WHERE model_id = '...' AND quantity = '...' AND group_name = '...'.
- For "what is the chance" questions, first try key_probabilities: filter it ONLY by model_id
  (never by the question text) and return question, probability and in_words for all its rows.
- Look up the exact quantity name in the table `quantities` before filtering posterior_draws.
- A 90% credible interval is q05 to q95. Never add, average or subtract q05/q95 columns across rows;
  percentiles do not add up. For sums or differences, compute them per draw from posterior_draws
  (join on model_id, quantity and draw) and summarise afterwards.
- Use only columns that the table's schema lists, and filter only on words the user said.
  Every model has one fixed date or scenario already, so do not add date filters.
- Keep the columns quantity, group_name, draw and value when returning draws, so the
  posterior analyses can use them.
"""

# A 2B model plans badly unless the usual plan shapes are spelled out.
PLANNER_RULES = """
{{ super() }}

Plans for this posterior database (keep plans to two steps):
- A question about a finding (what is the chance, how likely, how much, how many, which is
  cheapest or highest): step 1 is the look_up_answer tool with the user's question, step 2 is
  ChatAgent quoting its answer.
- A request to show, list, plot or explore data: step 1 is SQLAgent, which creates the table
  the next step needs.
- Step 2 is ChatAgent to answer in words. This is the default, including for "what is the
  chance", "how likely" and "which ... most" questions.
- Use AnalysisAgent as step 2 only when the user asks to show, plot or explore draws,
  distributions, intervals, curves or parameters, or names an analysis (PosteriorProbability,
  JointProbability, PosteriorDistribution, DrawCurves, CredibleIntervals, UncertaintyFan,
  BudgetWhatIf, ParameterDiagnostics). Use VegaLiteAgent for any other chart.
- Questions about which data or models exist: TableListAgent alone.
"""

# (No quote characters inside Jinja expressions here: Lumen HTML-escapes overrides.)
EXPLAIN_RULES = """
{{ super() }}
{%- if memory.posterior_answer is defined %}

The look_up_answer tool found these answers, computed from the posterior draws. Answer the
user's question from them in two or three sentences, quoting the numbers and ranges exactly:
{{ memory.posterior_answer }}
{%- endif %}

You explain results of Bayesian models to non-specialists:
- Give the best guess and the range, e.g. "about 43% (likely between 40% and 46%)".
- Say probabilities as "about N in 100", using exactly the N you were given (0.98 is 98 in 100,
  not 9 in 10). Never call a probability below 99 in 100 certain.
- Only report numbers present in the data you were given; if a number is missing, say so.
- Mention the model's caveat (table models) when the answer depends on it.
"""


# Worked question -> SQL pairs: small models copy a pattern far more reliably than they follow rules.
SQL_EXAMPLES = """
{{ super() }}

Examples for this database:
- "What is the chance the recommended budget beats the current plan?" ->
  SELECT question, probability, in_words FROM key_probabilities WHERE model_id = 'E33'
- "How likely was Clinton to win Pennsylvania?" ->
  SELECT state, p_clinton_win, q05, median, q95 FROM election_states WHERE state = 'Pennsylvania'
- "Show the posterior draws of Clinton's share in Pennsylvania, Michigan and Wisconsin" ->
  SELECT quantity, group_name, draw, value FROM posterior_draws
  WHERE model_id = 'E40' AND quantity = 'clinton_state_share'
  AND group_name IN ('Pennsylvania', 'Michigan', 'Wisconsin')
- "Which states had the widest 90% intervals?" ->
  SELECT state, q05, median, q95, q95 - q05 AS width FROM election_states ORDER BY width DESC LIMIT 10
- "Chance that Clinton's national share is above 52%?" ->
  SELECT AVG(CASE WHEN value > 0.52 THEN 1.0 ELSE 0.0 END) AS probability FROM posterior_draws
  WHERE model_id = 'E40' AND quantity = 'clinton_national_share'
- "How much hotter is a once-in-100-years heatwave now than in the past?" ->
  SELECT climate, return_period_years, q05, median, q95 FROM heatwave_return_levels
  WHERE return_period_years BETWEEN 90 AND 110
- "Where is the next new customer cheapest?" ->
  SELECT market, channel, marginal_cost_q05, marginal_cost_median, marginal_cost_q95
  FROM mmm_budget_cells ORDER BY marginal_cost_median
- "Show the heatwave return levels" -> SELECT * FROM heatwave_return_levels
- "Plot the posterior distribution of the marginal cost per market" ->
  SELECT quantity, group_name, draw, value FROM posterior_draws
  WHERE model_id = 'E33' AND quantity = 'marginal_cost_eur'
- "Show the growth centile curves draw by draw" ->
  SELECT quantity, group_name, x, draw, value FROM posterior_draws
  WHERE model_id = 'E41' AND quantity = 'bmi_centile'
- "Did the E40 sampler converge?" ->
  SELECT variable, coord, r_hat, ess_bulk, ess_tail FROM param_summary WHERE model_id = 'E40' ORDER BY r_hat DESC
- "Show the trace plots of the E37 parameters" ->
  SELECT variable, coord, chain, draw, value FROM posterior_params WHERE model_id = 'E37'
- "Show the budget response curves" -> SELECT * FROM mmm_response_curves
"""


def make_agents():
    return [
        SQLAgent(template_overrides={"main": {"instructions": POSTERIOR_RULES, "examples": SQL_EXAMPLES}}),
        ChatAgent(template_overrides={"main": {"instructions": EXPLAIN_RULES}}),
        AnalysisAgent(analyses=ANALYSES),
    ]


def make_explorer(model=None):
    """Chat with the posterior warehouse; custom posterior analyses are one click away."""
    return lmai.ExplorerUI(
        data=make_source(),
        llm=make_llm(model),
        agents=make_agents(),
        # fewer agents = fewer ways for a small model to route a question wrongly
        default_agents=[TableListAgent, ChatAgent, SQLAgent, VegaLiteAgent, ValidationAgent],
        coordinator=PosteriorPlanner,
        tools=[StepTool(look_up_answer, provides=["posterior_answer"], purpose=(
            "look_up_answer: answers questions about the models' findings (chances, how much, how many, "
            "cheapest, highest) from answers computed in advance from the posterior draws."))],
        coordinator_params={"template_overrides": {"main": {"instructions": PLANNER_RULES}}},
        title="Posterior Explorer",
        suggestions=[
            ("casino", "What is the chance the recommended E33 budget beats the current plan?"),
            ("how_to_vote", "Which states had the widest 90% intervals in the 2016 forecast?"),
            ("thermostat", "How much more likely did warming make the 2021 Seattle heatwave?"),
            ("child_care", "Show the uncertainty of the 97th BMI centile line by age."),
        ],
        log_level="INFO",
    )
