"""Descriptions of every table in the posterior warehouse.

Lumen hands these to the LLM with the table schemas, so they are written for a
small model: say what one row is, what each column means and its units, and
which question the table answers. They are also stored in the warehouse
itself (table ``table_guide``) so the descriptions travel with the data.
"""

from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[1]
WAREHOUSE = REPORTS_DIR / "posteriors.duckdb"
RESULTS_DIR = REPORTS_DIR / "results"

Q_COLS = {
    "q05": "5th percentile of the posterior (lower end of the 90% credible interval)",
    "q25": "25th percentile of the posterior",
    "median": "posterior median (the single best guess)",
    "q75": "75th percentile of the posterior",
    "q95": "95th percentile of the posterior (upper end of the 90% credible interval)",
}

METADATA = {
    "key_probabilities": {
        "description": (
            "START HERE for 'what is the chance / how likely / how sure' questions. One row per "
            "headline question about a model, answered with a posterior probability computed "
            "from the draws. Filter by model_id and read question, probability and in_words."
        ),
        "columns": {
            "model_id": "model id: E33 marketing budget, E36 state opinion, E37 heatwave, E40 2016 election, E41 child growth",
            "question": "the yes/no question, in words",
            "probability": "posterior probability that the answer is yes, 0-1",
            "in_words": "the probability said as 'about N in 100'",
            "computed_with_sql": "the SQL over posterior_draws that computed the probability",
        },
    },
    "models": {
        "description": (
            "One row per Bayesian model (example notebook). Use it to find the model_id, "
            "the question each model answers, its data and its main caveat."
        ),
        "columns": {
            "model_id": "short id of the model and its notebook, e.g. E36",
            "notebook": "notebook file the posterior comes from",
            "title": "the question the model answers, in plain words",
            "topic": "subject area",
            "data": "data the model was fitted to",
            "method": "statistical model used",
            "headline": "the main finding in one sentence",
            "caveat": "the most important limitation",
        },
    },
    "posterior_draws": {
        "description": (
            "Long table of equally likely posterior draws for every model. One row is one "
            "draw of one quantity (optionally for one group). The chance that a quantity "
            "exceeds a threshold t is AVG(CASE WHEN value > t THEN 1 ELSE 0 END) over its "
            "draws. Draws with the same model_id, quantity and draw number belong to the "
            "same joint posterior sample, so joint events (e.g. two states both won) can be "
            "computed by joining on draw."
        ),
        "columns": {
            "model_id": "model id, see table models",
            "quantity": "name of the posterior quantity, see table quantities",
            "group_name": "group the draw belongs to (a state, a market|channel cell, ...); 'all' when the quantity has no groups",
            "x": "for curves only (see quantities.x_name): the x value, e.g. age or return period; NULL otherwise",
            "draw": "draw number (0-based); same number = same joint sample",
            "value": "value of the quantity in this draw, in the unit of table quantities",
        },
    },
    "posterior_params": {
        "description": (
            "Every chain and draw of the fitted model parameters (the raw posterior from the sampler), "
            "one row per model_id, variable, coordinate, chain and draw. Parameters with more than "
            "200 values per draw are only in param_summary. Use it for trace plots, rank plots, pair "
            "plots and convergence checks."
        ),
        "columns": {
            "model_id": "model id",
            "variable": "parameter name as in the PyMC model",
            "coord": "which element of the parameter, e.g. 'market=Denmark, channel=Google'; 'all' for scalars",
            "chain": "MCMC chain number",
            "draw": "draw number within the chain",
            "value": "parameter value",
        },
    },
    "param_summary": {
        "description": (
            "One row per model parameter element: posterior mean, sd, percentiles and convergence "
            "diagnostics (R-hat near 1.00 and ESS in the hundreds or more mean the sampler converged)."
        ),
        "columns": {
            "model_id": "model id",
            "variable": "parameter name as in the PyMC model",
            "coord": "which element of the parameter; 'all' for scalars",
            "mean": "posterior mean",
            "sd": "posterior standard deviation",
            **Q_COLS,
            "r_hat": "rank-normalised split R-hat; above 1.01 suggests the chains disagree",
            "ess_bulk": "effective sample size for the centre of the distribution",
            "ess_tail": "effective sample size for the tails (5% and 95% quantiles)",
            "n_draws": "number of draws (chains x draws)",
        },
    },
    "quantities": {
        "description": (
            "Dictionary of every quantity in posterior_draws: what it means and its unit. "
            "Look here first to find the right quantity name."
        ),
        "columns": {
            "model_id": "model id",
            "quantity": "quantity name used in posterior_draws",
            "unit": "unit of the value column",
            "meaning": "what the quantity means in plain words",
            "x_name": "for curves: what the x column of posterior_draws holds (e.g. age_years); NULL otherwise",
            "n_groups": "number of groups the quantity has",
            "n_draws": "number of posterior draws per group",
        },
    },
    "posterior_summary": {
        "description": (
            "Summary of posterior_draws: one row per model_id, quantity and group_name, with "
            "mean, sd and percentiles. Use it for point estimates and 90% credible intervals "
            "(q05 to q95)."
        ),
        "columns": {
            "model_id": "model id",
            "quantity": "quantity name",
            "group_name": "group, 'all' if none",
            "x": "x value for curves, NULL otherwise",
            "mean": "posterior mean",
            "sd": "posterior standard deviation",
            **Q_COLS,
            "p_positive": "posterior probability the quantity is above zero",
            "n_draws": "number of draws summarised",
        },
    },
    "mmm_budget_cells": {
        "description": (
            "E33 hierarchical marketing-mix model of a Scandinavian apparel brand. One row per "
            "market and ad channel: spend today, the current and model-recommended yearly "
            "budgets, and the posterior of the marginal cost of one extra new customer."
        ),
        "columns": {
            "market": "country: Denmark, Sweden or Norway",
            "channel": "ad channel: Google or Meta",
            "spend_today_eur_per_week": "average weekly spend today, EUR",
            "current_plan_eur_per_year": "current yearly budget, EUR",
            "recommended_plan_eur_per_year": "budget the model recommends (same total), EUR",
            "cautious_plan_eur_per_year": "a more cautious re-allocation, EUR",
            "marginal_cost_q05": "5th percentile of EUR per extra new customer at today's spend",
            "marginal_cost_median": "median EUR per extra new customer at today's spend",
            "marginal_cost_q95": "95th percentile of EUR per extra new customer",
            "p_next_1000_beats_average": "probability the next EUR 1,000 here buys customers more cheaply than this channel's average",
        },
    },
    "mmm_response_curves": {
        "description": (
            "E33 response curves: expected new customers per week at a sustained weekly spend, "
            "per market and channel, with posterior percentiles. Diminishing returns show as "
            "the curve flattening."
        ),
        "columns": {
            "market": "Denmark, Sweden or Norway",
            "channel": "Google or Meta",
            "spend_eur_per_week": "sustained weekly spend, EUR",
            "spend_today_eur_per_week": "average weekly spend today, EUR",
            "recommended_pct_of_today": "the recommended plan's budget as % of the current budget",
            "cautious_pct_of_today": "the cautious plan's budget as % of the current budget",
            **{k: f"{v} of new customers per week" for k, v in Q_COLS.items()},
        },
    },
    "mrp_state_opinion": {
        "description": (
            "E36 multilevel regression and post-stratification (MRP): estimated share of adults "
            "in each US state agreeing that employers may decline to cover abortion in "
            "insurance plans (CCES 2018, 5,000 respondents). One row per state."
        ),
        "columns": {
            "state": "state name",
            "abbr": "two-letter state code",
            "n_survey": "respondents from this state in the 5,000-person sample",
            "raw_survey_share": "share agreeing among those respondents (no model)",
            **{k: f"{v}; share of adults agreeing, 0-1" for k, v in Q_COLS.items()},
            "p_majority_agree": "posterior probability that more than half of the state's adults agree",
            "benchmark_share": "MRP estimate from the full 55,000-person survey, used as a check",
            "no_state_predictor_median": "median from a model without the state Republican vote share predictor",
        },
    },
    "mrp_state_by_age": {
        "description": "E36 MRP estimates by state and age group. One row per state and age group.",
        "columns": {
            "state": "state name",
            "age_group": "age band, e.g. 18-29",
            "q05": "5th percentile of the share agreeing",
            "median": "median share agreeing",
            "q95": "95th percentile of the share agreeing",
        },
    },
    "heatwave_return_levels": {
        "description": (
            "E37 extreme-event attribution for the June 2021 Pacific Northwest heatwave at "
            "Seattle-Tacoma airport (non-stationary GEV). One row per return period and "
            "climate: the hottest day expected once every return_period_years, with "
            "posterior percentiles."
        ),
        "columns": {
            "climate": "'2021 climate' or 'past climate (1.2 C cooler)'",
            "return_period_years": "the day is expected once in this many years",
            "q05": "5th percentile of the return level, degrees C",
            "median": "median return level, degrees C",
            "q95": "95th percentile of the return level, degrees C",
        },
    },
    "heatwave_observed": {
        "description": "E37 observed hottest day of each year at Seattle-Tacoma airport.",
        "columns": {"year": "calendar year", "annual_max_c": "hottest daily maximum that year, degrees C"},
    },
    "election_states": {
        "description": (
            "E40 poll-aggregation forecast of the 2016 US presidential election made on "
            "2016-11-07. One row per state (and DC): Clinton's share of the two-party vote with "
            "percentiles, her win probability, and the actual result."
        ),
        "columns": {
            "state": "state name",
            "abbr": "two-letter code",
            "electoral_votes": "electoral votes of the state",
            "p_clinton_win": "posterior probability Clinton wins the state",
            "p_tipping_point": "probability the state casts the decisive electoral vote",
            **{k: f"{v}; Clinton two-party vote share, 0-1" for k, v in Q_COLS.items()},
            "actual_share": "Clinton's actual two-party vote share",
            "n_polls": "number of polls of the state",
            "inside_90": "true if the actual share fell inside the 90% interval",
        },
    },
    "election_timeline": {
        "description": "E40 how the forecast moved through 2016. One row per forecast date.",
        "columns": {
            "as_of": "forecast date",
            "p_clinton_win": "probability Clinton wins the electoral college",
            "ev_q05": "5th percentile of Clinton electoral votes",
            "ev_median": "median Clinton electoral votes",
            "ev_q95": "95th percentile of Clinton electoral votes",
            "state_cover90": "share of states whose actual result fell inside the 90% interval",
        },
    },
    "growth_centiles": {
        "description": (
            "E41 Bayesian BMI growth charts (GAMLSS / Box-Cox t) for Dutch boys aged 0-21. One "
            "row per age and centile line (3rd, 10th, 50th, 90th, 97th): the BMI of that centile "
            "with posterior percentiles showing how uncertain the line itself is."
        ),
        "columns": {
            "age_years": "age in years",
            "centile": "centile line, e.g. 97 means 97% of boys that age are below it",
            **{k: f"{v}; BMI of the centile line, kg/m^2" for k, v in Q_COLS.items()},
        },
    },
}

# Quantity dictionary: (model_id, quantity) -> (unit, meaning)
QUANTITIES = {
    ("E33", "plan_gain"): ("new customers per year", "extra new customers next year from the recommended budget split versus the current one (same total spend)"),
    ("E33", "move_10k_norway_meta_to_sweden_google"): ("new customers per year", "extra new customers from moving EUR 10,000 a year from Norway Meta to Sweden Google"),
    ("E33", "marginal_cost_eur"): ("EUR per new customer", "cost of one extra new customer at today's spend, per market|channel"),
    ("E36", "national_share_agree"): ("share 0-1", "share of all US adults agreeing"),
    ("E36", "state_share_agree"): ("share 0-1", "share of adults agreeing, per state"),
    ("E36", "n_majority_states"): ("states", "number of states where a majority agrees"),
    ("E37", "p_108F_per_summer"): ("probability per summer", "chance a summer has a day of 108 F (42.2 C) or hotter, per climate"),
    ("E37", "p_100F_per_summer"): ("probability per summer", "chance a summer has a day of 100 F (37.8 C) or hotter, per climate"),
    ("E37", "probability_ratio_108F"): ("ratio", "how many times more likely a 108 F day is in 2021 than in the past climate; NULL where it was impossible in the past climate (infinite ratio)"),
    ("E37", "intensity_change_c"): ("degrees C", "how much hotter a heatwave of the same rarity is in 2021 than in the past climate"),
    ("E37", "ceiling_c"): ("degrees C", "hottest possible day under the fitted distribution (GEV upper bound), per climate"),
    ("E40", "clinton_electoral_votes"): ("electoral votes", "Clinton's electoral votes out of 538 (270 wins)"),
    ("E40", "clinton_national_share"): ("share 0-1", "Clinton share of the national two-party vote"),
    ("E40", "clinton_state_share"): ("share 0-1", "Clinton two-party share per state; draws are joint across states"),
    ("E41", "pct_boys_heavier"): ("percent", "percent of boys the same age with a higher BMI than the example boy"),
    ("E41", "tail_df"): ("degrees of freedom", "tail heaviness of the Box-Cox t distribution (small = heavy tails)"),
}
