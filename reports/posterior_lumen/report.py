"""A Lumen Report over the posterior warehouse: one section per model.

Each section is a fixed recipe the author controls (SQL + posterior analyses with preset
widgets) opened by a short summary. Readers change the widgets, edit the SQL, re-run a
section, or export the report to a notebook.

Summaries are "facts first": Python computes every number from the posterior draws, and
the local LLM only turns that fact sheet into two or three plain sentences. A 2B model asked
to read a table itself mixes up percentiles and invents averages; asked to phrase given
facts, it is reliable. The fact sheet is shown under the summary so readers can check it.
Set ``captions=False`` (or ``POSTERIOR_LLM=none``) to show the facts without any LLM.
"""

import duckdb
import numpy as np
import panel as pn
import param

from lumen.ai.actions import SQLQuery
from lumen.ai.report import Action, Report, Section
from lumen.pipeline import Pipeline

from .analyses import (
    BudgetWhatIf, CredibleIntervals, DrawCurves, JointProbability, ParameterDiagnostics, PosteriorDistribution,
    PosteriorProbability, UncertaintyFan,
)
from .app import make_llm, make_source
from .stories import STORIES
from .catalog import WAREHOUSE

NARRATOR = (
    "You write for a non-specialist reader of a statistics report. Using ONLY the facts given, "
    "write two or three short plain sentences that answer the question. Keep every number exactly "
    "as written in the facts, including its range. Do not add numbers, causes or advice that are "
    "not in the facts. No headings, no bullet points."
)


def pct(p):
    """A probability as 'about N in 100'."""
    if p >= 0.995:
        return "over 99 in 100"
    if p <= 0.005:
        return "under 1 in 100"
    return f"about {round(100 * p)} in 100"


def draws(con, model_id, quantity, group="all"):
    return con.execute(
        "SELECT value FROM posterior_draws WHERE model_id = ? AND quantity = ? AND group_name = ? ORDER BY draw",
        [model_id, quantity, group]).df()["value"].to_numpy(float)  # NULL -> NaN


def rng(v, fmt="{:,.0f}"):
    v = v[~np.isnan(v)]
    lo, mid, hi = np.quantile(v, [0.05, 0.5, 0.95])
    return f"about {fmt.format(mid)} (likely {fmt.format(lo)} to {fmt.format(hi)})"


class Narrate(Action):
    """Compute a fact sheet from the warehouse and have the LLM phrase it."""

    question = param.String(doc="The question the summary answers.")

    facts = param.Callable(doc="Function (duckdb connection) -> list of fact strings.")

    async def _execute(self, context, **kwargs):
        with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
            facts = self.facts(con)
        sheet = "\n".join(f"- {f}" for f in facts)
        outputs = []
        if self.llm is not None:
            reply = await self.llm.invoke(
                [{"role": "user", "content": f"Question: {self.question}\n\nFacts:\n{sheet}"}],
                system=NARRATOR)
            # without a response_model, invoke returns the raw ChatCompletion
            text = reply.choices[0].message.content if hasattr(reply, "choices") else str(reply)
            outputs.append(pn.pane.Markdown(text.strip(), sizing_mode="stretch_width",
                                            styles={"font-size": "1.05em"}))
        outputs.append(pn.Card(pn.pane.Markdown(sheet), title="Numbers behind this summary (computed, not generated)",
                               collapsed=self.llm is not None, sizing_mode="stretch_width"))
        return outputs, {}


def mmm_facts(con):
    cells = con.execute("SELECT * FROM mmm_budget_cells ORDER BY marginal_cost_median").df()
    out = [f"{r.market} {r.channel}: the next new customer costs about EUR {r.marginal_cost_median:.0f} "
           f"(likely EUR {r.marginal_cost_q05:.0f} to {r.marginal_cost_q95:.0f})" for r in cells.itertuples()]
    gain = draws(con, "E33", "plan_gain")
    out.append(f"Re-splitting the same total budget as the model recommends: {rng(gain)} extra new customers "
               f"a year; chance it beats today's plan: {pct((gain > 0).mean())}")
    out.append("The model has no experiment behind it; test before moving large sums")
    return out


def mrp_facts(con):
    states = con.execute("SELECT * FROM mrp_state_opinion ORDER BY median DESC").df()
    hi, lo = states.iloc[0], states.iloc[-1]
    small = states.sort_values("n_survey").iloc[0]
    inside = ((states.benchmark_share >= states.q05) & (states.benchmark_share <= states.q95)).mean()
    return [
        f"Nationally, {rng(100 * draws(con, 'E36', 'national_share_agree'), '{:.0f}%')} of adults agree",
        f"Highest: {hi.state}, about {100 * hi['median']:.0f}% (likely {100 * hi.q05:.0f}% to {100 * hi.q95:.0f}%)",
        f"Lowest: {lo.state}, about {100 * lo['median']:.0f}% (likely {100 * lo.q05:.0f}% to {100 * lo.q95:.0f}%)",
        f"{small.state} had only {small.n_survey} people in the survey, yet the model estimates about "
        f"{100 * small['median']:.0f}% (likely {100 * small.q05:.0f}% to {100 * small.q95:.0f}%) by borrowing "
        f"strength from similar people and states",
        f"Checked against a 55,000-person survey, the likely range contained the check value in "
        f"{100 * inside:.0f} of 100 states",
    ]


def heat_facts(con):
    out = []
    for climate, words in [("past climate (1.2 C cooler)", "a world 1.2 C cooler than 2021"),
                           ("2021 climate", "the 2021 climate"), ("2025 climate", "the 2025 climate")]:
        p = draws(con, "E37", "p_100F_per_summer", climate)
        out.append(f"In {words}, a 100 F day comes in {rng(100 * p, '{:.0f}')} summers out of 100")
    out.append(f"A heatwave of the same rarity is now {rng(draws(con, 'E37', 'intensity_change_c'), '{:.1f}')} "
               f"degrees C hotter than in the cooler world")
    pr = draws(con, "E37", "probability_ratio_108F")
    out.append(f"In {pct(np.isnan(pr).mean())} of the model's explanations, the 108 F day of 28 June 2021 "
               f"was impossible in the cooler world")
    return out


def election_facts(con):
    ev = draws(con, "E40", "clinton_electoral_votes")
    states = con.execute("SELECT * FROM election_states").df()
    wide = {s: draws(con, "E40", "clinton_state_share", s) for s in ["Pennsylvania", "Michigan", "Wisconsin"]}
    wins = np.column_stack([v > 0.5 for v in wide.values()])
    return [
        f"On 7 November 2016 the model gave Clinton a win chance of {pct((ev >= 270).mean())}, "
        f"with {rng(ev)} electoral votes",
        "She actually won 233 electoral votes and lost",
        f"The actual state result fell inside the model's 9-in-10 range in only "
        f"{100 * states.inside_90.mean():.0f} of 100 states, so the ranges were too narrow",
        f"The model gave Clinton a chance of {pct(wins.all(axis=1).mean())} to win Pennsylvania, Michigan and "
        f"Wisconsin together (she lost all three)",
    ]


def growth_facts(con):
    out = []
    for group in ["age 7, BMI 18.9", "age 15, BMI 24.2"]:
        v = draws(con, "E41", "pct_boys_heavier", group)
        age, bmi = group.replace("age ", "").split(", BMI ")
        out.append(f"For an example {age}-year-old boy with BMI {bmi}, {rng(v, '{:.1f}')} in 100 boys his age "
                   f"have a higher BMI")
    out.append("The centile lines are estimates too: the chart shows their uncertainty as bands")
    return out


class PosteriorView(Action):
    """Run a posterior analysis on the result of a SQL query."""

    analysis = param.Parameter(doc="PosteriorAnalysis subclass to run.")

    preset = param.Dict(default={}, doc="Starting widget values for the analysis.")

    source = param.Parameter(doc="Source to query.")

    sql_expr = param.String(doc="SQL selecting the rows the analysis needs.")

    table = param.String(doc="Name for the query result.")

    async def _execute(self, context, **kwargs):
        source = self.source.create_sql_expr_source({self.table: self.sql_expr})
        pipeline = Pipeline(source=source, table=self.table)
        view = self.analysis.instance(preset=self.preset)(pipeline, context)
        return [view], {}


class Story(Action):
    """The example's story for a lay reader: bespoke maps, fan charts and dot charts."""

    model_id = param.String()

    async def _execute(self, context, **kwargs):
        return [STORIES[self.model_id]()], {}


def _story(model_id):
    return Story(model_id=model_id, title="")


def _query(source, table, sql, title):
    return SQLQuery(source=source, table=table, sql_expr=sql, title=title, generate_caption=False)


def _narrate(llm, question, facts):
    return Narrate(llm=llm, question=question, facts=facts, title="In short")


def _view(source, analysis, table, sql, title, **preset):
    return PosteriorView(source=source, analysis=analysis, table=table, sql_expr=sql,
                         title=title, preset=preset)


def draws_sql(model_id, quantity):
    return (f"SELECT quantity, group_name, draw, value FROM posterior_draws "
            f"WHERE model_id = '{model_id}' AND quantity = '{quantity}'")


def curves_sql(model_id, quantity):
    return (f"SELECT d.quantity, d.group_name, d.x, q.x_name, d.draw, d.value FROM posterior_draws d "
            f"JOIN quantities q USING (model_id, quantity) "
            f"WHERE d.model_id = '{model_id}' AND d.quantity = '{quantity}'")


def params_sql(model_id, max_elements=3):
    """Raw draws of the model's small parameters (hyperparameters and scalars): the ones
    convergence checks look at first. Vector effects stay in posterior_params for the explorer."""
    return (f"SELECT variable, coord, chain, draw, value FROM posterior_params WHERE model_id = '{model_id}' "
            f"AND variable IN (SELECT variable FROM param_summary WHERE model_id = '{model_id}' "
            f"GROUP BY variable HAVING COUNT(*) <= {max_elements}) ORDER BY variable, coord, chain, draw")


def has_full_posterior(model_id):
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        return "posterior_params" in tables and bool(con.execute(
            "SELECT COUNT(*) FROM posterior_params WHERE model_id = ?", [model_id]).fetchone()[0])


def full_posterior(source, model_id, *views):
    """The 'full posterior' tail of a section: extra uncertainty views plus sampler diagnostics."""
    if not has_full_posterior(model_id):
        return []
    return [*views, _view(source, ParameterDiagnostics, f"{model_id.lower()}_params", params_sql(model_id),
                          "Full posterior · the fitted hyperparameters, chain by chain")]


def make_report(captions=True, model=None, auto_execute=True):
    source = make_source()
    llm = make_llm(model) if captions else None

    mmm = Section(
        _story("E33"),
        _narrate(llm, "Where should next year's marketing budget go, and how sure are we?", mmm_facts),
        _query(source, "mmm_cells",
               "SELECT market, channel, spend_today_eur_per_week, current_plan_eur_per_year, "
               "recommended_plan_eur_per_year, marginal_cost_q05, marginal_cost_median, marginal_cost_q95, "
               "p_next_1000_beats_average FROM mmm_budget_cells ORDER BY marginal_cost_median",
               "Where does the next euro work hardest?"),
        _view(source, BudgetWhatIf, "mmm_curves", "SELECT * FROM mmm_response_curves",
              "What if we change the weekly budget?", plan="Model's recommended plan"),
        _view(source, PosteriorProbability, "mmm_draws",
              "SELECT quantity, group_name, draw, value FROM posterior_draws WHERE model_id = 'E33'",
              "How sure are we that the recommended plan helps?",
              quantity="plan_gain", threshold=0.0, direction="above"),
        *full_posterior(
            source, "E33",
            _view(source, PosteriorDistribution, "mmm_cost_dist", draws_sql("E33", "marginal_cost_eur"),
                  "Full posterior · the cost of the next customer, cell by cell"),
            _view(source, DrawCurves, "mmm_curve_draws", curves_sql("E33", "customers_per_week"),
                  "Full posterior · response curves, draw by draw",
                  groups=["Denmark|Google", "Denmark|Meta"]),
        ),
        title="E33 · Marketing budget across three markets",
    )

    mrp = Section(
        _story("E36"),
        _narrate(llm, "What does each US state think, from one survey of 5,000 people?", mrp_facts),
        _query(source, "mrp_states",
               "SELECT state, n_survey, raw_survey_share, q05, median, q95, p_majority_agree "
               "FROM mrp_state_opinion ORDER BY median DESC",
               "Every state from one survey of 5,000 people"),
        _view(source, CredibleIntervals, "mrp_intervals", "SELECT * FROM mrp_state_opinion",
              "State estimates against the 55,000-person benchmark", reference="benchmark_share", top=50),
        _view(source, JointProbability, "mrp_joint", draws_sql("E36", "state_share_agree"),
              "Chance that a majority agrees in several states at once",
              groups=["West Virginia", "Alabama", "Mississippi"], threshold=0.5),
        *full_posterior(
            source, "E36",
            _view(source, PosteriorDistribution, "mrp_dist", draws_sql("E36", "state_share_agree"),
                  "Full posterior · big and small states side by side",
                  groups=["California", "Texas", "Wyoming", "Vermont", "Alaska", "Delaware"]),
        ),
        title="E36 · Public opinion by state (MRP)",
    )

    heat = Section(
        _story("E37"),
        _narrate(llm, "How much did warming change Seattle's hottest days?", heat_facts),
        _query(source, "heat_summary",
               "SELECT quantity, group_name, median, q05, q95, n_draws FROM posterior_summary "
               "WHERE model_id = 'E37' AND quantity IN ('p_100F_per_summer', 'intensity_change_c')",
               "What warming did to Seattle's hottest days"),
        _view(source, UncertaintyFan, "heat_levels", "SELECT * FROM heatwave_return_levels",
              "How hot is a once-in-N-years day? (the X marks 28 June 2021)",
              logx=True, mark_x=100.0, mark_y=42.2),
        _view(source, PosteriorProbability, "heat_draws", draws_sql("E37", "p_100F_per_summer"),
              "Chance of a 100 °F day in a summer",
              threshold=0.1, direction="above"),
        *full_posterior(
            source, "E37",
            _view(source, DrawCurves, "heat_curve_draws", curves_sql("E37", "return_level_c"),
                  "Full posterior · return-level curves, draw by draw", logx=True),
            _view(source, PosteriorDistribution, "heat_dist", draws_sql("E37", "intensity_change_c"),
                  "Full posterior · how much hotter the same rarity has become"),
        ),
        title="E37 · The 2021 Pacific Northwest heatwave",
    )

    election = Section(
        _story("E40"),
        _narrate(llm, "What did the poll model expect in 2016, and how did it do?", election_facts),
        _query(source, "election_timeline", "SELECT * FROM election_timeline ORDER BY as_of",
               "How the 2016 forecast moved"),
        _view(source, CredibleIntervals, "election_intervals",
              "SELECT * FROM election_states WHERE n_polls > 0",
              "State forecasts against the result", reference="actual_share", top=51),
        _view(source, JointProbability, "election_joint", draws_sql("E40", "clinton_state_share"),
              "The 'blue wall': did the model know these states move together?",
              groups=["Pennsylvania", "Michigan", "Wisconsin"], threshold=0.5),
        *full_posterior(
            source, "E40",
            _view(source, PosteriorDistribution, "election_ev_dist", draws_sql("E40", "clinton_electoral_votes"),
                  "Full posterior · Clinton's electoral votes as equally likely elections"),
            _view(source, PosteriorDistribution, "election_state_dist", draws_sql("E40", "clinton_state_share"),
                  "Full posterior · swing-state vote shares",
                  groups=["Florida", "Pennsylvania", "Michigan", "Wisconsin", "North Carolina", "Ohio"]),
        ),
        title="E40 · Forecasting the 2016 US election",
    )

    growth = Section(
        _story("E41"),
        _narrate(llm, "How unusual are these boys' BMIs, and how sure is the chart?", growth_facts),
        _view(source, UncertaintyFan, "growth_fan", "SELECT * FROM growth_centiles WHERE centile IN (3, 50, 97)",
              "BMI centile lines for Dutch boys, with the uncertainty of the lines themselves",
              mark_x=7.0, mark_y=18.9),
        _view(source, PosteriorProbability, "growth_draws", draws_sql("E41", "pct_boys_heavier"),
              "How unusual are the example boys?", threshold=5.0, direction="below"),
        *full_posterior(
            source, "E41",
            _view(source, DrawCurves, "growth_curve_draws", curves_sql("E41", "bmi_centile"),
                  "Full posterior · centile lines, draw by draw",
                  groups=["centile 97", "centile 50"], logx=False),
        ),
        title="E41 · Growth charts",
    )

    return Report(mmm, mrp, heat, election, growth, title="Posterior Report", auto_execute=auto_execute)
