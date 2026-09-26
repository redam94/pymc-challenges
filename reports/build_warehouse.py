"""Build the posterior warehouse that the Lumen apps explore.

Reads what the example notebooks export and writes tidy tables into
``reports/posteriors.duckdb``. The full posterior comes from ``reports/posteriors/<id>.nc``
(written by each notebook's ``save_posterior`` cell: all chains and draws of the parameters,
and the derived quantities over all their draws). The page exports ``reports/results/*.json``
(copied from ``.scratch/artifact/``) supply the summary tables, and the draws of any model
whose netCDF file is missing.

* ``posterior_draws``   long table of equally likely draws (model, quantity, group, x, draw, value)
* ``posterior_summary`` mean / sd / percentiles per quantity and group
* ``posterior_params``  every chain and draw of the fitted model parameters
* ``param_summary``     per parameter element: mean, sd, percentiles, R-hat, ESS
* ``quantities``        dictionary of the quantities, with units
* ``models``            one row per model: question, data, method, headline, caveat
* one wide table per model for the things a reader asks about most

    uv run --extra reports python reports/build_warehouse.py            # from reports/results
    uv run --extra reports python reports/build_warehouse.py --refresh  # re-copy notebook exports first
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import arviz as az
import duckdb
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posterior_lumen.catalog import METADATA, QUANTITIES, RESULTS_DIR, WAREHOUSE  # noqa: E402

MODELS = ["E33", "E36", "E37", "E40", "E41"]
POSTERIORS_DIR = RESULTS_DIR.parent / "posteriors"
PARAM_DRAWS_MAX = 200  # parameters with at most this many values per draw keep every draw
NOTEBOOK_EXPORTS = Path(__file__).resolve().parents[1] / ".scratch" / "artifact"


def load(model_id):
    return json.loads((RESULTS_DIR / f"{model_id}.json").read_text())


def draws_rows(model_id, quantity, values, group="all"):
    """Rows of posterior_draws for one quantity and group; None becomes NULL."""
    values = [np.nan if v is None else float(v) for v in values]
    return pd.DataFrame({
        "model_id": model_id, "quantity": quantity, "group_name": group, "x": np.nan,
        "draw": np.arange(len(values)), "value": values,
    })


def split_cell(cell):
    market, channel = cell.split("|")
    return market, channel


def mmm(d):
    """E33 hierarchical geo MMM."""
    draws = [
        draws_rows("E33", "plan_gain", d["plan_gain"]["draws"]),
        draws_rows("E33", "move_10k_norway_meta_to_sweden_google", d["move_example"]["draws"]),
    ]
    draws += [draws_rows("E33", "marginal_cost_eur", v, cell)
              for cell, v in d["marginal_cost_eur"]["draws"].items()]
    mc = d["marginal_cost_eur"]
    rc = d["response_curves"]
    cells, curves = [], []
    for cell in d["cells"]:
        market, channel = split_cell(cell)
        cells.append({
            "market": market, "channel": channel,
            "spend_today_eur_per_week": rc["spend_today_eur_per_week"][cell],
            "current_plan_eur_per_year": d["current_plan_eur_per_year"][cell],
            "recommended_plan_eur_per_year": d["recommended_plan_eur_per_year"][cell],
            "cautious_plan_eur_per_year": d["cautious_plan_eur_per_year"][cell],
            "marginal_cost_q05": mc["q05"][cell], "marginal_cost_median": mc["q50"][cell],
            "marginal_cost_q95": mc["q95"][cell],
            "p_next_1000_beats_average": d["p_next_1000_beats_average"][cell],
        })
        current = d["current_plan_eur_per_year"][cell]
        curve = pd.DataFrame({
            "market": market, "channel": channel, "spend_eur_per_week": rc["spend_eur_per_week"][cell],
            "spend_today_eur_per_week": rc["spend_today_eur_per_week"][cell],
            "recommended_pct_of_today": 100 * d["recommended_plan_eur_per_year"][cell] / current,
            "cautious_pct_of_today": 100 * d["cautious_plan_eur_per_year"][cell] / current,
        })
        for q, col in [("q05", "q05"), ("q25", "q25"), ("q50", "median"), ("q75", "q75"), ("q95", "q95")]:
            curve[col] = rc["quantiles"][q][cell]
        curves.append(curve)
    model = {
        "model_id": "E33", "notebook": "E33_hierarchical_geo_mmm.ipynb",
        "title": d["title"], "topic": "marketing", "data": f"{d['brand']}, {d['period']}",
        "method": "hierarchical Bayesian marketing-mix model with adstock and saturation, partial pooling across markets",
        "headline": d["headlines"]["plan_gain"], "caveat": d["headlines"]["caveat"],
    }
    tables = {"mmm_budget_cells": pd.DataFrame(cells), "mmm_response_curves": pd.concat(curves)}
    return model, draws, tables


def mrp(d):
    """E36 multilevel regression and post-stratification."""
    draws = [draws_rows("E36", "national_share_agree", d["national"]["draws"]),
             draws_rows("E36", "n_majority_states", d["majority_states_draws"])]
    states, by_age = [], []
    for s in d["states"]:
        draws.append(draws_rows("E36", "state_share_agree", s["draws"], s["name"]))
        states.append({
            "state": s["name"], "abbr": s["abbr"], "n_survey": s["n_survey"],
            "raw_survey_share": s["raw_survey"],
            "q05": s["q05"], "q25": s["q25"], "median": s["median"], "q75": s["q75"], "q95": s["q95"],
            "p_majority_agree": s["p_majority"], "benchmark_share": s["benchmark_mrp_55k"],
            "no_state_predictor_median": s["no_state_predictor_median"],
        })
        a = s["by_age"]
        by_age.append(pd.DataFrame({"state": s["name"], "age_group": a["groups"],
                                    "q05": a["q05"], "median": a["median"], "q95": a["q95"]}))
    model = {
        "model_id": "E36", "notebook": "E36_mrp_poststratification.ipynb", "title": d["title"],
        "topic": "public opinion", "data": "CCES 2018 (5,000-person subsample) post-stratified to ACS census cells",
        "method": "multilevel logistic regression + post-stratification (MRP)",
        "headline": d["headlines"]["national"] + " " + d["headlines"]["states"],
        "caveat": d["headlines"]["failure"],
    }
    return model, draws, {"mrp_state_opinion": pd.DataFrame(states), "mrp_state_by_age": pd.concat(by_age)}


def heatwave(d):
    """E37 extreme-event attribution."""
    dr = d["draws"]
    climates = {"past": "past climate (1.2 C cooler)", "2021": "2021 climate", "2025": "2025 climate"}
    draws = []
    for key, label in climates.items():
        draws.append(draws_rows("E37", "p_108F_per_summer", dr[f"p108_{key}"], label))
        draws.append(draws_rows("E37", "p_100F_per_summer", dr[f"p100_{key}"], label))
    draws.append(draws_rows("E37", "probability_ratio_108F", dr["pr_108"]))
    draws.append(draws_rows("E37", "intensity_change_c", dr["intensity_change_c"]))
    draws.append(draws_rows("E37", "ceiling_c", dr["ceiling_past_c"], climates["past"]))
    draws.append(draws_rows("E37", "ceiling_c", dr["ceiling_2021_c"], climates["2021"]))
    rl = d["return_levels"]
    levels = []
    for key, label in [("now_2021_c", climates["2021"]), ("past_c", climates["past"])]:
        q05, q50, q95 = rl[key]
        levels.append(pd.DataFrame({"climate": label, "return_period_years": rl["return_period_years"],
                                    "q05": q05, "median": q50, "q95": q95}))
    observed = pd.DataFrame(d["observed"])
    model = {
        "model_id": "E37", "notebook": "E37_extreme_event_attribution.ipynb", "title": d["title"],
        "topic": "climate", "data": d["station"] + " annual maximum temperatures + GISTEMP global temperature",
        "method": d["model"],
        "headline": d["headlines"]["probability_ratio"] + " " + d["headlines"]["intensity"],
        "caveat": d["headlines"]["impossible_past"],
    }
    return model, draws, {"heatwave_return_levels": pd.concat(levels), "heatwave_observed": observed}


def election(d):
    """E40 poll aggregation forecast (2016)."""
    draws = [draws_rows("E40", "clinton_electoral_votes", d["ev_draws"]),
             draws_rows("E40", "clinton_national_share", d["national_share_draws"])]
    names = {s["abbr"]: s["name"] for s in d["states"]}
    joint = np.asarray(d["map_draws"])  # draws x states, in state_order
    for j, abbr in enumerate(d["state_order"]):
        draws.append(draws_rows("E40", "clinton_state_share", joint[:, j], names[abbr]))
    states = pd.DataFrame([{
        "state": s["name"], "abbr": s["abbr"], "electoral_votes": s["ev"],
        "p_clinton_win": s["p_clinton"], "p_tipping_point": s["p_tipping_point"],
        "q05": s["q05"], "q25": s["q25"], "median": s["median"], "q75": s["q75"], "q95": s["q95"],
        "actual_share": s["actual"], "n_polls": s["n_polls"],
        "inside_90": s["q05"] <= s["actual"] <= s["q95"],
    } for s in d["states"]])
    timeline = pd.DataFrame(d["timeline"]).rename(columns={"p_clinton": "p_clinton_win"})
    timeline = timeline[["as_of", "p_clinton_win", "ev_q05", "ev_median", "ev_q95", "state_cover90"]]
    timeline["as_of"] = pd.to_datetime(timeline["as_of"])
    model = {
        "model_id": "E40", "notebook": "E40_poll_aggregation_forecasting.ipynb", "title": d["title"],
        "topic": "elections", "data": "2016 US presidential state and national polls (Economist / Heidemanns et al.)",
        "method": "dynamic Bayesian poll aggregation (Linzer / Economist) with correlated state errors",
        "headline": d["headlines"]["win_probability"], "caveat": d["headlines"]["polling_error"],
    }
    return model, draws, {"election_states": states, "election_timeline": timeline}


def growth(d):
    """E41 distributional regression growth charts."""
    draws = [draws_rows("E41", "tail_df", d["tail_df_draws"])]
    for b in d["example_boys"]:
        draws.append(draws_rows("E41", "pct_boys_heavier", b["pct_boys_heavier_draws"],
                                f"age {b['age']:g}, BMI {b['bmi']:g}"))
    g = d["grid"]
    cols = {0.05: "q05", 0.25: "q25", 0.5: "median", 0.75: "q75", 0.95: "q95"}
    rows = []
    for ci, centile in enumerate(g["centiles"]):
        frame = pd.DataFrame({"age_years": g["age"], "centile": int(centile)})
        for qi, q in enumerate(g["quantiles"]):
            frame[cols[q]] = g["values"][ci][qi]
        rows.append(frame)
    model = {
        "model_id": "E41", "notebook": "E41_distributional_regression_growth.ipynb", "title": d["title"],
        "topic": "health", "data": d["data"], "method": d["model"],
        "headline": d["headlines"]["chart"], "caveat": d["headlines"]["naive"],
    }
    return model, draws, {"growth_centiles": pd.concat(rows)}


# Headline questions answered straight from the draws. Each SQL returns one probability; it is
# stored next to the answer so readers (and the LLM) can see how the number was computed.
def draws_of(model_id, quantity, group="all"):
    return (f"SELECT draw, value FROM posterior_draws WHERE model_id = '{model_id}' "
            f"AND quantity = '{quantity}' AND group_name = '{group}'")


def all_above(model_id, quantity, groups, t):
    """P(every group above t), computed draw by draw."""
    names = ", ".join(f"'{g}'" for g in groups)
    return (f"SELECT AVG(ok) FROM (SELECT draw, MIN(CASE WHEN value > {t} THEN 1.0 ELSE 0.0 END) AS ok "
            f"FROM posterior_draws WHERE model_id = '{model_id}' AND quantity = '{quantity}' "
            f"AND group_name IN ({names}) GROUP BY draw)")


PAST, NOW = "past climate (1.2 C cooler)", "2021 climate"
KEY_QUESTIONS = [
    ("E33", "Does the recommended budget split beat the current plan?",
     f"SELECT AVG(CASE WHEN value > 0 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E33', 'plan_gain')})"),
    ("E33", "Does moving EUR 10,000 a year from Norway Meta to Sweden Google add customers?",
     f"SELECT AVG(CASE WHEN value > 0 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E33', 'move_10k_norway_meta_to_sweden_google')})"),
    ("E33", "Does the recommended split add more than 2,000 new customers a year?",
     f"SELECT AVG(CASE WHEN value > 2000 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E33', 'plan_gain')})"),
    ("E36", "Does more than 45% of US adults agree?",
     f"SELECT AVG(CASE WHEN value > 0.45 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E36', 'national_share_agree')})"),
    ("E36", "Does a majority agree in at least 5 states?",
     f"SELECT AVG(CASE WHEN value >= 5 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E36', 'n_majority_states')})"),
    ("E36", "Does a majority agree in West Virginia, Kentucky and Oklahoma all at once?",
     all_above("E36", "state_share_agree", ["West Virginia", "Kentucky", "Oklahoma"], 0.5)),
    ("E37", "Is a 100 F summer more likely in the 2021 climate than in the past climate?",
     f"SELECT AVG(CASE WHEN a.value > b.value THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E37', 'p_100F_per_summer', NOW)}) a "
     f"JOIN ({draws_of('E37', 'p_100F_per_summer', PAST)}) b USING (draw)"),
    ("E37", "Was the 28 June 2021 heat (108 F) impossible in the past climate but possible in 2021?",
     f"SELECT AVG(CASE WHEN b.value = 0 AND a.value > 0 THEN 1.0 ELSE 0.0 END) "
     f"FROM ({draws_of('E37', 'p_108F_per_summer', NOW)}) a JOIN ({draws_of('E37', 'p_108F_per_summer', PAST)}) b USING (draw)"),
    ("E37", "Is a heatwave of the same rarity at least 2 C hotter now than in the past climate?",
     f"SELECT AVG(CASE WHEN value >= 2 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E37', 'intensity_change_c')})"),
    ("E40", "Does Clinton win the electoral college (270 or more votes)?",
     f"SELECT AVG(CASE WHEN value >= 270 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E40', 'clinton_electoral_votes')})"),
    ("E40", "Does Clinton win the national two-party vote?",
     f"SELECT AVG(CASE WHEN value > 0.5 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E40', 'clinton_national_share')})"),
    ("E40", "Does Clinton win Pennsylvania, Michigan and Wisconsin (the blue wall) together?",
     all_above("E40", "clinton_state_share", ["Pennsylvania", "Michigan", "Wisconsin"], 0.5)),
    ("E41", "Is the example 7-year-old (BMI 18.9) above the 95th centile (fewer than 5 in 100 boys heavier)?",
     f"SELECT AVG(CASE WHEN value < 5 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E41', 'pct_boys_heavier', 'age 7, BMI 18.9')})"),
    ("E41", "Is the example 15-year-old (BMI 24.2) above the 95th centile (fewer than 5 in 100 boys heavier)?",
     f"SELECT AVG(CASE WHEN value < 5 THEN 1.0 ELSE 0.0 END) FROM ({draws_of('E41', 'pct_boys_heavier', 'age 15, BMI 24.2')})"),
]


def in_words(p):
    if p >= 0.995:
        return "almost certain (over 99 in 100)"
    if p <= 0.005:
        return "almost impossible (under 1 in 100)"
    return f"about {round(100 * p)} in 100"


def key_probabilities(con):
    rows = []
    for model_id, question, sql in KEY_QUESTIONS:
        p = float(con.execute(sql).fetchone()[0])
        rows.append({"model_id": model_id, "question": question, "probability": round(p, 3),
                     "in_words": in_words(p), "computed_with_sql": sql})
    return pd.DataFrame(rows)


def sql_string(text):
    """COMMENT ON does not take bound parameters: quote the literal by hand."""
    return "'" + text.replace("'", "''") + "'"


def _labels(df, dims, style="value"):
    """One string per row naming its coordinates: 'Denmark|Google', 'centile 97', 'state=Ohio'."""
    if not dims:
        return pd.Series("all", index=df.index)
    parts = []
    for d in dims:
        v = df[d]
        if style == "dim=value":
            parts.append(d + "=" + v.astype(str))
        elif pd.api.types.is_numeric_dtype(v):
            parts.append(d + " " + v.map("{:g}".format))
        else:
            parts.append(v.astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + ("|" if style == "value" else ", ") + p
    return out


def derived_draws(model_id, tree):
    """posterior_draws rows and quantity descriptions from a notebook's derived/ groups."""
    x_dims = set(filter(None, tree["derived"].attrs.get("x_dims", "").split(",")))
    frames, info = [], []
    for name, node in tree["derived"].children.items():
        da = node.to_dataset()[name].load()
        other = [d for d in da.dims if d != "sample"]
        xd = [d for d in other if d in x_dims]
        gd = [d for d in other if d not in x_dims]
        df = da.to_dataframe(name="value").reset_index()
        df["value"] = df["value"].where(np.isfinite(df["value"]))
        frames.append(pd.DataFrame({
            "model_id": model_id, "quantity": name, "group_name": _labels(df, gd).values,
            "x": df[xd[0]].astype(float).values if xd else np.nan,
            "draw": df["sample"].values, "value": df["value"].values}))
        info.append({"model_id": model_id, "quantity": name, "unit": da.attrs.get("unit", ""),
                     "meaning": da.attrs.get("meaning", ""), "x_name": xd[0] if xd else None})
    return frames, info


def parameters(model_id, tree):
    """Long draws of the small parameters, and a summary with diagnostics for all of them."""
    post = tree["posterior"].to_dataset().load()
    rhat, ess_bulk, ess_tail = az.rhat(post), az.ess(post, method="bulk"), az.ess(post, method="tail")
    draws, summaries = [], []
    for var in post.data_vars:
        da = post[var]
        dims = [d for d in da.dims if d not in ("chain", "draw")]
        stats = xr.Dataset({"r_hat": rhat[var], "ess_bulk": ess_bulk[var], "ess_tail": ess_tail[var],
                            "mean": da.mean(("chain", "draw")), "sd": da.std(("chain", "draw")),
                            **{k: da.quantile(q, ("chain", "draw")).drop_vars("quantile")
                               for k, q in [("q05", .05), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95)]}})
        st = stats.to_dataframe().reset_index() if dims else pd.DataFrame([{k: float(v) for k, v in stats.items()}])
        st.insert(0, "coord", _labels(st, dims, "dim=value").values)
        st.insert(0, "variable", var)
        st.insert(0, "model_id", model_id)
        st["n_draws"] = da.sizes["chain"] * da.sizes["draw"]
        summaries.append(st[["model_id", "variable", "coord", "mean", "sd", "q05", "q25", "median", "q75", "q95",
                             "r_hat", "ess_bulk", "ess_tail", "n_draws"]])
        if int(np.prod([da.sizes[d] for d in dims])) <= PARAM_DRAWS_MAX:
            df = da.to_dataframe(name="value").reset_index()
            draws.append(pd.DataFrame({"model_id": model_id, "variable": var,
                                       "coord": _labels(df, dims, "dim=value").values,
                                       "chain": df["chain"].values, "draw": df["draw"].values,
                                       "value": df["value"].astype(float).values}))
    return draws, summaries


SUMMARY_SQL = """
SELECT model_id, quantity, group_name, x,
       AVG(value) AS mean, STDDEV(value) AS sd,
       QUANTILE_CONT(value, 0.05) AS q05, QUANTILE_CONT(value, 0.25) AS q25, MEDIAN(value) AS median,
       QUANTILE_CONT(value, 0.75) AS q75, QUANTILE_CONT(value, 0.95) AS q95,
       AVG(CASE WHEN value > 0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE value IS NOT NULL) AS p_positive,
       COUNT(value) AS n_draws
FROM posterior_draws GROUP BY ALL ORDER BY model_id, quantity, group_name, x
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help="copy the latest notebook exports from .scratch/artifact first")
    args = parser.parse_args()
    if args.refresh:
        for m in MODELS:
            shutil.copy(NOTEBOOK_EXPORTS / f"{m}.json", RESULTS_DIR / f"{m}.json")

    builders = {"E33": mmm, "E36": mrp, "E37": heatwave, "E40": election, "E41": growth}
    models, draws, tables, info, param_draws, param_summaries = [], [], {}, [], [], []
    for m in MODELS:
        model, json_draws, t = builders[m](load(m))
        models.append(model)
        tables.update(t)
        path = POSTERIORS_DIR / f"{m}.nc"
        if path.exists():
            tree = xr.open_datatree(path, engine="h5netcdf")
            full, full_info = derived_draws(m, tree)
            p_draws, p_summ = parameters(m, tree)
            param_draws += p_draws
            param_summaries += p_summ
            tree.close()
            have = {q["quantity"] for q in full_info}
            json_draws = [d for d in json_draws if d["quantity"].iloc[0] not in have]
            draws += full
            info += full_info
            print(f"{m}: full posterior from {path.name} ({len(have)} derived quantities, {len(p_summ)} parameters)")
        else:
            print(f"{m}: no {path.name}; using the {len(json_draws[0])}-draw page export")
        draws += json_draws
        info += [{"model_id": m, "quantity": d["quantity"].iloc[0], "unit": QUANTITIES[(m, d["quantity"].iloc[0])][0],
                  "meaning": QUANTITIES[(m, d["quantity"].iloc[0])][1], "x_name": None} for d in json_draws]
    draws = pd.concat(draws, ignore_index=True)
    scratch = duckdb.connect()
    scratch.register("posterior_draws", draws)
    summary = scratch.execute(SUMMARY_SQL).df()
    counts = scratch.execute("SELECT model_id, quantity, COUNT(DISTINCT group_name) AS n_groups, "
                             "COUNT(DISTINCT draw) AS n_draws FROM posterior_draws GROUP BY ALL").df()
    quantities = pd.DataFrame(info).merge(counts, on=["model_id", "quantity"])
    quantities = quantities[["model_id", "quantity", "unit", "meaning", "x_name", "n_groups", "n_draws"]]

    tables = {"models": pd.DataFrame(models), "quantities": quantities,
              "posterior_draws": draws, "posterior_summary": summary, **tables}
    if param_summaries:
        tables["param_summary"] = pd.concat(param_summaries, ignore_index=True)
        tables["posterior_params"] = pd.concat(param_draws, ignore_index=True)
    # compute the key probabilities with the very SQL we store
    tables = {"key_probabilities": key_probabilities(scratch), **tables}
    scratch.close()
    guide = pd.DataFrame([{"table_name": t, "description": METADATA[t]["description"]} for t in tables])
    missing = set(tables) - set(METADATA)
    assert not missing, f"tables without a description in catalog.py: {missing}"

    WAREHOUSE.unlink(missing_ok=True)
    con = duckdb.connect(str(WAREHOUSE))
    for name, frame in {**tables, "table_guide": guide}.items():
        con.register("frame", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
        con.unregister("frame")
        if name in METADATA:  # DuckDB comments, so any SQL client sees the descriptions too
            con.execute(f"COMMENT ON TABLE {name} IS {sql_string(METADATA[name]['description'])}")
            for col, text in METADATA[name]["columns"].items():
                con.execute(f'COMMENT ON COLUMN {name}."{col}" IS {sql_string(text)}')
        print(f"{name:24s} {len(frame):9,d} rows")
    con.close()
    print(f"wrote {WAREHOUSE}")


if __name__ == "__main__":
    main()
