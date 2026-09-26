"""Deterministic posterior displays that Lumen can run on any table with the right columns.

An LLM is good at turning a question into SQL and picking a display; it is not good at
probability arithmetic. So the numbers a reader acts on (a probability, an interval, a
what-if) are computed here in plain NumPy, and the LLM only chooses *which* analysis to
run on *which* rows. Every analysis returns its own widgets, so the reader keeps
exploring without asking the model again.

The analyses apply by column names (``Analysis.columns``):

* posterior draws (``quantity, group_name, draw, value``): ``PosteriorProbability``,
  ``JointProbability``, ``PosteriorDistribution``; with an ``x`` column: ``DrawCurves``
* raw sampler output (``variable, coord, chain, draw, value``): ``ParameterDiagnostics``
* summaries with ``q05, median, q95``: ``CredibleIntervals``, ``UncertaintyFan``
* E33 response curves: ``BudgetWhatIf``
"""

import arviz as az
import holoviews as hv
import hvplot.pandas  # noqa: F401  (registers .hvplot)
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import panel as pn
import param

import lumen.ai as lmai
from lumen.ai.utils import get_data

hv.extension("bokeh", logo=False)

# One qualitative palette for groups, one colour for "the model" and one for "reality".
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]
BAND = "#4C78A8"
TRUTH = "#E45756"

LABEL_COLUMNS = ["state", "group_name", "coord", "market", "channel", "climate", "age_group", "as_of",
                 "quantity", "variable"]
X_COLUMNS = ["spend_eur_per_week", "age_years", "return_period_years", "as_of", "year"]


def _pane(plot, height):
    """A plot with a fixed height that fills the width. Height set only through HoloViews
    options is lost inside Panel layouts (plots collapsed to a sliver in the served report)."""
    return pn.pane.HoloViews(plot, height=height, sizing_mode="stretch_width", linked_axes=False)


def _in_words(p):
    """A probability the way a forecaster would say it."""
    if p is None or np.isnan(p):
        return "undefined"
    if p >= 0.995:
        return "almost certain (over 99 in 100)"
    if p <= 0.005:
        return "almost impossible (under 1 in 100)"
    return f"about {round(p * 100)} in 100"


def _label_column(df):
    return next((c for c in LABEL_COLUMNS if c in df.columns and df[c].nunique() > 1),
                next((c for c in LABEL_COLUMNS if c in df.columns), None))


class PosteriorAnalysis(lmai.Analysis):
    """Base class: ``preset`` gives the widgets their starting values (used by the report)."""

    preset = param.Dict(default={}, precedence=-1, doc="Initial widget values, by widget name.")

    def _apply_preset(self, **widgets):
        for name, widget in widgets.items():
            if name in self.preset:
                widget.value = self.preset[name]


def _draw_frame(pipeline):
    """Draws of scalar quantities; curve rows (non-null x) belong to DrawCurves."""
    df = pipeline.data.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "x" in df.columns:
        df = df[df["x"].isna()]
    if df.empty:
        raise ValueError("These rows are curve draws (they have an x value): use DrawCurves instead.")
    return df


def _interval_table(df, by):
    rows = []
    for g, v in df.groupby(by, sort=False)["value"]:
        v = v.dropna().to_numpy()
        if not len(v):
            continue
        q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({by: g, "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else 0.0,
                     "5%": q[0], "25%": q[1], "median": q[2], "75%": q[3], "95%": q[4], "draws": len(v)})
    return pd.DataFrame(rows).round(4)


class PosteriorProbability(PosteriorAnalysis):
    """
    Probability that a posterior quantity is above (or below) a threshold, computed by
    counting posterior draws, with the draws shown as a histogram. Use for questions like
    'what is the chance that X exceeds Y' or 'how sure are we that X is positive'.
    """

    columns = param.List(default=["quantity", "group_name", "draw", "value"])

    def __call__(self, pipeline, context):
        df = _draw_frame(pipeline)
        quantities = list(df["quantity"].unique())
        quantity = pn.widgets.Select(name="Quantity", options=quantities, value=quantities[0])
        groups = pn.widgets.MultiChoice(name="Groups (up to 8)", max_items=8)
        direction = pn.widgets.RadioButtonGroup(options=["above", "below"], value="above")
        threshold = pn.widgets.FloatInput(name="Threshold", step=0.01)

        def reset(q):
            sub = df[df["quantity"] == q]
            opts = list(sub["group_name"].unique())
            groups.options = opts
            groups.value = opts[: min(len(opts), 4)]
            v = sub["value"].dropna()
            # default threshold: zero if the draws straddle it, else the pooled median
            threshold.value = float(0 if v.min() < 0 < v.max() else round(v.median(), 3))
            threshold.step = float(max((v.max() - v.min()) / 100, 1e-6))

        quantity.param.watch(lambda e: reset(e.new), "value")
        reset(quantity.value)
        self._apply_preset(quantity=quantity)
        self._apply_preset(groups=groups, threshold=threshold, direction=direction)

        def view(q, gs, t, d):
            sub = df[(df["quantity"] == q) & df["group_name"].isin(gs or [])]
            if sub.empty or t is None:
                return pn.pane.Markdown("Pick at least one group.")
            rows = []
            for g, part in sub.groupby("group_name", sort=False):
                v = part["value"]
                n_undefined = int(v.isna().sum())
                v = v.dropna()
                p = float((v > t).mean() if d == "above" else (v < t).mean()) if len(v) else np.nan
                rows.append({"group": g, f"P({d} {t:g})": round(p, 3), "in words": _in_words(p),
                             "draws": len(v), "undefined draws": n_undefined})
            table = pd.DataFrame(rows)
            hist = sub.dropna(subset=["value"]).hvplot.hist(
                "value", by="group_name", bins=40, alpha=0.5, color=PALETTE[: len(gs)],
                height=320, responsive=True, xlabel=q, ylabel="draws", legend="top_right", shared_axes=False)
            line = hv.VLine(t).opts(color="black", line_dash="dashed", line_width=1.5)
            note = ("Each draw is one equally likely version of the world the model believes in; "
                    "the probability is the share of draws on the chosen side of the line.")
            if table["undefined draws"].sum():
                note += (" Undefined draws (e.g. an infinite ratio) are left out of the count: "
                         "read them as 'the event was impossible in the comparison scenario'.")
            return pn.Column(_pane((hist * line).opts(shared_axes=False), 330), pn.widgets.Tabulator(table, show_index=False, disabled=True,
                                                                  sizing_mode="stretch_width"),
                             pn.pane.Markdown(note, styles={"color": "#666"}), sizing_mode="stretch_width")

        controls = pn.Row(quantity, groups, pn.Column("Direction", direction), threshold)
        return pn.Column(controls, pn.bind(view, quantity, groups, threshold, direction),
                         sizing_mode="stretch_width")


class JointProbability(PosteriorAnalysis):
    """
    Probability that several groups ALL exceed a threshold at once (e.g. Clinton wins
    Pennsylvania AND Wisconsin AND Michigan), computed draw by draw so correlations between
    groups are respected, and compared with the naive product of the separate probabilities.
    """

    columns = param.List(default=["quantity", "group_name", "draw", "value"])

    def __call__(self, pipeline, context):
        df = _draw_frame(pipeline)
        # only quantities with several groups have a joint distribution worth showing
        multi = df.groupby("quantity")["group_name"].nunique()
        quantities = list(multi[multi > 1].index) or list(df["quantity"].unique())
        quantity = pn.widgets.Select(name="Quantity", options=quantities, value=quantities[0])
        groups = pn.widgets.MultiChoice(name="All of these groups", max_items=10)
        threshold = pn.widgets.FloatInput(name="…are above", value=0.5, step=0.01)

        def reset(q):
            sub = df[df["quantity"] == q]
            groups.options = list(sub["group_name"].unique())
            groups.value = groups.options[:3]
            threshold.value = float(round(sub["value"].median(), 3))

        quantity.param.watch(lambda e: reset(e.new), "value")
        reset(quantity.value)
        self._apply_preset(quantity=quantity)
        self._apply_preset(groups=groups, threshold=threshold)

        def view(q, gs, t):
            if not gs or t is None:
                return pn.pane.Markdown("Pick groups.")
            wide = (df[(df["quantity"] == q) & df["group_name"].isin(gs)]
                    .pivot_table(index="draw", columns="group_name", values="value"))
            wide = wide.dropna()
            hit = wide > t
            p_joint = float(hit.all(axis=1).mean())
            p_each = hit.mean()
            p_indep = float(p_each.prod())
            n = len(wide)
            md = (f"### P(all {len(gs)} above {t:g}) = **{p_joint:.2f}** ({_in_words(p_joint)})\n\n"
                  f"Multiplying the separate chances as if the groups were independent gives "
                  f"**{p_indep:.2f}**. ")
            if p_joint > p_indep + 0.02:
                md += ("The joint chance is higher: the groups move together in the model "
                       "(a shared error hits them all at once), so treating them as "
                       "independent would overstate the risk of a split result.")
            elif p_joint < p_indep - 0.02:
                md += "The joint chance is lower: the groups tend to move in opposite directions."
            else:
                md += "The two agree, so these groups barely move together."
            md += f"\n\nComputed from {n} joint posterior draws."
            each = pd.DataFrame({"group": p_each.index, f"P(above {t:g})": p_each.values.round(3)})
            # how many of the chosen groups clear the threshold in each draw
            counts = hit.sum(axis=1).value_counts().reindex(range(len(gs) + 1), fill_value=0) / n
            bars = pd.DataFrame({"groups above threshold": counts.index, "share of draws": counts.values}).hvplot.bar(
                x="groups above threshold", y="share of draws", color=BAND, height=260, responsive=True,
                shared_axes=False)
            bars = _pane(bars, 270)
            return pn.Column(pn.pane.Markdown(md), pn.Row(
                pn.widgets.Tabulator(each, show_index=False, disabled=True, width=320), bars),
                sizing_mode="stretch_width")

        return pn.Column(pn.Row(quantity, groups, threshold), pn.bind(view, quantity, groups, threshold),
                         sizing_mode="stretch_width")


class CredibleIntervals(PosteriorAnalysis):
    """
    Ranked 90% credible intervals (q05 to q95, with the median) for every row, e.g. every
    state or market. Optionally overlays a reference column such as the actual result or the
    raw survey share, and reports how often the reference falls inside the interval.
    """

    columns = param.List(default=["q05", "median", "q95"])

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        label = _label_column(df)
        if label is None:
            df["row"] = df.index.astype(str)
            label = "row"
        if df[label].duplicated().any():  # e.g. state x age: label by both columns
            others = [c for c in LABEL_COLUMNS + ["centile"] if c in df.columns and c != label]
            if others:
                df[label] = df[label].astype(str) + " · " + df[others[0]].astype(str)
        numeric = [c for c in df.select_dtypes("number").columns
                   if c not in {"q05", "q25", "median", "q75", "q95"}]
        refs = [c for c in ["actual_share", "benchmark_share", "raw_survey_share"] if c in numeric]
        reference = pn.widgets.Select(name="Compare with", options=["(none)"] + refs,
                                      value=refs[0] if refs else "(none)")
        top = pn.widgets.IntSlider(name="Rows shown", start=5, end=max(len(df), 5),
                                   value=min(len(df), 30))
        order = pn.widgets.RadioButtonGroup(options=["highest first", "lowest first", "widest first"],
                                            value="highest first")
        self._apply_preset(reference=reference, top=top, order=order)

        def view(ref, n, how):
            d = df.assign(width=df["q95"] - df["q05"])
            key, asc = {"highest first": ("median", False), "lowest first": ("median", True),
                        "widest first": ("width", False)}[how]
            d = d.sort_values(key, ascending=asc).head(n).iloc[::-1]
            d[label] = d[label].astype(str)
            height = max(260, 18 * len(d) + 60)
            d = d.assign(y0=d[label], y1=d[label])
            seg = hv.Segments(d, kdims=["q05", "y0", "q95", "y1"]).opts(color=BAND, line_width=3, alpha=0.5)
            if {"q25", "q75"} <= set(d.columns):
                seg = seg * hv.Segments(d, kdims=["q25", "y0", "q75", "y1"]).opts(color=BAND, line_width=6)
            plot = seg * hv.Scatter(d, "median", "y0").opts(color="black", size=5)
            md = "Bars: 90% credible interval (thick part: middle 50%). Dot: median."
            if ref != "(none)":
                plot = plot * hv.Scatter(d, ref, "y0").opts(color=TRUTH, marker="diamond", size=8)
                inside = ((df[ref] >= df["q05"]) & (df[ref] <= df["q95"])).mean()
                md += (f" Red diamond: `{ref}`. It falls inside the 90% interval for "
                       f"**{inside:.0%}** of all {len(df)} rows (well-calibrated intervals: about 90%).")
            lo = d[["q05"] + ([ref] if ref != "(none)" else [])].min().min()
            hi = d[["q95"] + ([ref] if ref != "(none)" else [])].max().max()
            pad = (hi - lo) * 0.05
            plot = plot.opts(height=height, responsive=True, xlabel="value", ylabel="", xlim=(lo - pad, hi + pad),
                             show_grid=True, shared_axes=False)
            return pn.Column(_pane(plot, height), pn.pane.Markdown(md, styles={"color": "#666"}),
                             sizing_mode="stretch_width")

        return pn.Column(pn.Row(reference, top, order), pn.bind(view, reference, top, order),
                         sizing_mode="stretch_width")


class UncertaintyFan(PosteriorAnalysis):
    """
    Fan chart of an uncertain curve: posterior median with 50% and 90% bands against a
    continuous x (spend, age, return period, date), one panel per curve. Optionally marks a
    point of interest and says which part of the band it falls in.
    """

    # One of X_COLUMNS is required too; Lumen allows a tuple for "one of" in `columns`, but its
    # planner prompt joins `columns` as strings and crashes on it, so check that in applies().
    columns = param.List(default=["q05", "median", "q95"])

    @classmethod
    async def applies(cls, pipeline) -> bool:
        if not await super().applies(pipeline):
            return False
        data = await get_data(pipeline)
        return any(c in data.columns for c in X_COLUMNS)

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        x = next(c for c in X_COLUMNS if c in df.columns)
        splits = [c for c in ["market", "channel", "climate", "centile", "state"] if c in df.columns]
        # numbers alone ("97") make poor legend entries: prefix them with the column name
        named = df[splits].apply(lambda col: (col.name + " " + col.astype(str)) if col.dtype.kind in "iuf"
                                 else col.astype(str))
        df["curve"] = named.agg(" | ".join, axis=1) if splits else "curve"
        curves = list(df["curve"].unique())
        chosen = pn.widgets.MultiChoice(name="Curves", options=curves, value=curves[: min(6, len(curves))])
        logx = pn.widgets.Checkbox(name="log x axis", value=x == "return_period_years")
        mark_x = pn.widgets.FloatInput(name=f"Mark {x} (optional)", value=None)
        mark_y = pn.widgets.FloatInput(name="…and value (optional)", value=None)
        self._apply_preset(chosen=chosen, logx=logx, mark_x=mark_x, mark_y=mark_y)

        def view(cs, log, mx, my):
            overlays, notes = [], []
            for i, c in enumerate(cs or []):
                d = df[df["curve"] == c].sort_values(x)
                col = PALETTE[i % len(PALETTE)]
                layer = hv.Area(d, x, ["q05", "q95"]).opts(color=col, alpha=0.15, line_alpha=0)
                if {"q25", "q75"} <= set(d.columns):
                    layer = layer * hv.Area(d, x, ["q25", "q75"]).opts(color=col, alpha=0.3, line_alpha=0)
                layer = layer * hv.Curve(d, x, "median", label=c).opts(color=col, line_width=2)
                overlays.append(layer)
                if mx is not None and x != "as_of":
                    xs = d[x].to_numpy(float)
                    q = {k: float(np.interp(mx, xs, d[k])) for k in ["q05", "median", "q95"]}
                    line = f"- **{c}** at {x} = {mx:g}: median {q['median']:.3g}, 90% band {q['q05']:.3g} to {q['q95']:.3g}"
                    if my is not None:
                        where = ("below the band" if my < q["q05"] else "above the band" if my > q["q95"]
                                 else "inside the 90% band")
                        line += f"; the value {my:g} is **{where}**"
                    notes.append(line)
            if not overlays:
                return pn.pane.Markdown("Pick at least one curve.")
            plot = hv.Overlay(overlays)
            if mx is not None and my is not None:
                plot = plot * hv.Points([(mx, my)]).opts(color="black", size=10, marker="x")
            shown = df[df["curve"].isin(cs)]
            lo, hi = shown["q05"].min(), shown["q95"].max()
            if my is not None:
                lo, hi = min(lo, my), max(hi, my)
            pad = (hi - lo) * 0.05
            plot = plot.opts(height=380, responsive=True, logx=log, xlabel=x, ylabel="value",
                             ylim=(lo - pad, hi + pad), legend_position="top_left", show_grid=True,
                             shared_axes=False)
            md = ("Line: posterior median. Dark band: middle 50%. Light band: 90%. The band is the "
                  "uncertainty of the curve itself, not the spread of individual outcomes.")
            return pn.Column(_pane(plot, 390), pn.pane.Markdown("\n".join(notes)) if notes else None,
                             pn.pane.Markdown(md, styles={"color": "#666"}), sizing_mode="stretch_width")

        return pn.Column(pn.Row(chosen, logx, mark_x, mark_y), pn.bind(view, chosen, logx, mark_x, mark_y),
                         sizing_mode="stretch_width")


class BudgetWhatIf(PosteriorAnalysis):
    """
    What-if budget planner for the E33 marketing-mix model: move a slider per market and
    channel to change weekly spend and read off the expected change in new customers per
    week from the posterior response curves, with 90% bands.
    """

    columns = param.List(default=["market", "channel", "spend_eur_per_week", "spend_today_eur_per_week", "median"])

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        cells = df.drop_duplicates(["market", "channel"]).set_index(["market", "channel"])
        sliders = {cell: pn.widgets.IntSlider(name=f"{cell[0]} {cell[1]}: % of today's spend",
                                              start=0, end=200, step=5, value=100)
                   for cell in cells.index}
        plans = {"Today's plan": None}
        for col, name in [("recommended_pct_of_today", "Model's recommended plan"),
                          ("cautious_pct_of_today", "Cautious plan")]:
            if col in cells.columns:
                plans[name] = col
        plan = pn.widgets.RadioButtonGroup(name="Start from", options=list(plans), value="Today's plan")

        def load_plan(event):
            col = plans[event.new]
            for cell, slider in sliders.items():
                slider.value = 100 if col is None else int(5 * round(cells.loc[cell, col] / 5))

        plan.param.watch(load_plan, "value")
        if self.preset.get("plan") in plans:
            plan.value = self.preset["plan"]

        def interp(d, spend, col):
            return float(np.interp(spend, d["spend_eur_per_week"], d[col]))

        def view(*pcts):
            rows = []
            for ((market, channel), pct) in zip(sliders, pcts):
                d = df[(df["market"] == market) & (df["channel"] == channel)].sort_values("spend_eur_per_week")
                today = float(d["spend_today_eur_per_week"].iloc[0])
                new = today * pct / 100
                base = {k: interp(d, today, k) for k in ["q05", "median", "q95"]}
                after = {k: interp(d, new, k) for k in ["q05", "median", "q95"]}
                rows.append({"market": market, "channel": channel,
                             "spend today €/wk": round(today), "new spend €/wk": round(new),
                             "Δ spend €/wk": round(new - today),
                             "customers/wk today": round(base["median"], 1),
                             "customers/wk new (median)": round(after["median"], 1),
                             "new: 90% band": f"{after['q05']:.0f} to {after['q95']:.0f}",
                             "Δ customers/wk (median)": round(after["median"] - base["median"], 1)})
            t = pd.DataFrame(rows)
            d_spend, d_cust = t["Δ spend €/wk"].sum(), t["Δ customers/wk (median)"].sum()
            total_today = t["spend today €/wk"].sum()
            if abs(d_spend) < 0.02 * total_today:
                framing = "The total budget is about the same: this is a re-split, so any gain is free. "
            elif d_spend > 0 and d_cust > 0:
                framing = f"That is about **€{d_spend / d_cust:,.0f} per extra customer** on the added spend. "
            else:
                framing = ""
            md = (f"### Total: {d_spend:+,.0f} € per week → about {d_cust:+,.1f} new customers per week (sum of medians)\n\n"
                  + framing
                  + "Per-cell bands come from the posterior response curves. They cannot simply be added: "
                    "a band for the total needs joint posterior draws (see the E33 notebook). "
                    "The model is fitted without an experiment, so test before moving large sums.")
            bars = t.hvplot.barh(x="channel", y="Δ customers/wk (median)", by="market", height=260,
                                 responsive=True, color=PALETTE[:3], ylabel="change in customers/week (median)",
                                 shared_axes=False)
            return pn.Column(pn.pane.Markdown(md), _pane(bars, 270),
                             pn.widgets.Tabulator(t, show_index=False, disabled=True, sizing_mode="stretch_width"),
                             sizing_mode="stretch_width")

        grid = pn.GridBox(*sliders.values(), ncols=2, sizing_mode="stretch_width")
        return pn.Column(pn.Row("Start from:", plan), grid, pn.bind(view, *sliders.values()),
                         sizing_mode="stretch_width")


class PosteriorDistribution(PosteriorAnalysis):
    """
    Full shape of the posterior of a quantity for several groups: ridgeline densities + 50%/90% intervals,
    the cumulative distribution (read off P(value <= x) directly), a quantile dotplot for lay
    readers (each dot is an equally likely outcome), and a summary table.
    """

    columns = param.List(default=["quantity", "group_name", "draw", "value"])

    def __call__(self, pipeline, context):
        df = _draw_frame(pipeline)
        quantities = list(df["quantity"].unique())
        quantity = pn.widgets.Select(name="Quantity", options=quantities, value=quantities[0])
        groups = pn.widgets.MultiChoice(name="Groups (up to 12)", max_items=12)
        n_dots = pn.widgets.IntSlider(name="Dots in the dotplot", start=10, end=100, step=10, value=20)

        def reset(q):
            opts = list(df.loc[df["quantity"] == q, "group_name"].unique())
            groups.options = opts
            groups.value = opts[: min(len(opts), 6)]

        quantity.param.watch(lambda e: reset(e.new), "value")
        reset(quantity.value)
        self._apply_preset(quantity=quantity)
        self._apply_preset(groups=groups, n_dots=n_dots)

        def view(q, gs, nd):
            sub = df[(df["quantity"] == q) & df["group_name"].isin(gs or [])].dropna(subset=["value"])
            if sub.empty:
                return pn.pane.Markdown("Pick at least one group.")
            order = sub.groupby("group_name")["value"].median().sort_values().index.tolist()
            table = _interval_table(sub, "group_name").set_index("group_name").loc[order].reset_index()
            # Ridgeline on a numeric y axis (one row per group) with the 50% and 90% intervals
            # drawn on each baseline; mixing a categorical violin with segments garbled the axes.
            layers, ticks = [], []
            lo_all, hi_all = sub["value"].quantile([0.005, 0.995])
            grid = np.linspace(lo_all, hi_all, 256)
            for i, g in enumerate(order):
                v = sub.loc[sub["group_name"] == g, "value"].to_numpy()
                col = PALETTE[i % len(PALETTE)]
                if np.ptp(v) > 0:
                    dens = gaussian_kde(v)(grid)
                    dens = 0.85 * dens / dens.max()
                    layers.append(hv.Area((grid, i + dens, np.full_like(grid, i)), q, ["top", "base"]).opts(
                        color=col, alpha=0.35, line_color=col, line_alpha=0.8))
                q05, q25, q50, q75, q95 = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
                layers += [hv.Segments([(q05, i, q95, i)]).opts(color="black", line_width=1.5),
                           hv.Segments([(q25, i, q75, i)]).opts(color="black", line_width=5),
                           hv.Scatter([(q50, i)]).opts(color="white", line_color="black", size=8)]
                ticks.append((i, str(g)))
            height = max(260, 70 * len(order) + 90)
            dens = hv.Overlay(layers).opts(
                yticks=ticks, xlabel=q, ylabel="", show_grid=True, show_legend=False, shared_axes=False,
                ylim=(-0.4, len(order) - 0.05), title="Density per group; bar: middle 50% (thick) and 90% (thin)")
            curves = []
            for i, g in enumerate(order):
                v = np.sort(sub.loc[sub["group_name"] == g, "value"].to_numpy())
                curves.append(hv.Curve((v, np.arange(1, len(v) + 1) / len(v)), q, "P(value ≤ x)", label=str(g))
                              .opts(color=PALETTE[i % len(PALETTE)], interpolation="steps-post"))
            cdf = hv.Overlay(curves).opts(height=360, responsive=True, show_grid=True, legend_position="bottom_right",
                                          shared_axes=False, title="Read off any probability: hover the curve")
            dots = []
            for i, g in enumerate(order[:6]):
                v = sub.loc[sub["group_name"] == g, "value"].to_numpy()
                qd = np.quantile(v, (np.arange(nd) + 0.5) / nd)
                width = (qd.max() - qd.min()) / 25 or 1.0
                bins = np.floor((qd - qd.min()) / width)
                stack = pd.Series(bins).groupby(bins).cumcount().to_numpy()
                dots.append(hv.Scatter((qd.min() + (bins + 0.5) * width, stack + 1), q, "count").opts(
                    color=PALETTE[i % len(PALETTE)], size=9, title=str(g), height=180, responsive=True,
                    yaxis=None, shared_axes=False))
            dotplot = pn.Column(
                pn.pane.Markdown(f"Each dot is one of {nd} equally likely outcomes: 'about {round(100 / nd)} in 100' "
                                 f"per dot, so count dots instead of reading an interval."),
                *[_pane(d, 190) for d in dots], sizing_mode="stretch_width")
            return pn.Tabs(("Density & intervals", _pane(dens, height)), ("Cumulative", _pane(cdf, 370)),
                           ("Quantile dotplot", dotplot),
                           ("Table", pn.widgets.Tabulator(table, show_index=False, disabled=True,
                                                          sizing_mode="stretch_width")),
                           dynamic=True, sizing_mode="stretch_width")

        return pn.Column(pn.Row(quantity, groups, n_dots), pn.bind(view, quantity, groups, n_dots),
                         sizing_mode="stretch_width")


class DrawCurves(PosteriorAnalysis):
    """
    Spaghetti plot of posterior curves (response curves, return levels, growth centiles):
    individual equally likely draws as thin lines over the 50% and 90% bands computed from
    all draws. Shows that the uncertainty is about whole curves, not separate points.
    """

    columns = param.List(default=["quantity", "group_name", "x", "draw", "value"])

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        df = df[df["x"].notna()]
        if df.empty:
            raise ValueError("No curve rows (x is NULL everywhere): use PosteriorDistribution instead.")
        quantities = list(df["quantity"].unique())
        quantity = pn.widgets.Select(name="Quantity", options=quantities, value=quantities[0])
        groups = pn.widgets.MultiChoice(name="Curves (up to 4)", max_items=4)
        n_lines = pn.widgets.IntSlider(name="Draws drawn", start=0, end=200, step=10, value=40)
        logx = pn.widgets.Checkbox(name="log x axis", value=False)

        def reset(q):
            sub = df[df["quantity"] == q]
            groups.options = list(sub["group_name"].unique())
            groups.value = groups.options[:2]
            logx.value = bool(sub["x"].min() > 0 and sub["x"].max() / sub["x"].min() > 100)

        quantity.param.watch(lambda e: reset(e.new), "value")
        reset(quantity.value)
        self._apply_preset(quantity=quantity)
        self._apply_preset(groups=groups, n_lines=n_lines, logx=logx)

        def view(q, gs, n, log):
            layers = []
            rng = np.random.default_rng(0)
            for i, g in enumerate(gs or []):
                col = PALETTE[i % len(PALETTE)]
                sub = df[(df["quantity"] == q) & (df["group_name"] == g)]
                wide = sub.pivot_table(index="draw", columns="x", values="value")
                xs = wide.columns.to_numpy(float)
                band = np.nanquantile(wide.to_numpy(), [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
                layers.append(hv.Area((xs, band[0], band[4]), "x", ["lo", "hi"]).opts(color=col, alpha=0.12, line_alpha=0))
                layers.append(hv.Area((xs, band[1], band[3]), "x", ["lo", "hi"]).opts(color=col, alpha=0.22, line_alpha=0))
                pick = rng.choice(len(wide), size=min(n, len(wide)), replace=False) if n else []
                if len(pick):
                    paths = [np.column_stack([xs, wide.to_numpy()[k]]) for k in pick]
                    layers.append(hv.Path(paths).opts(color=col, alpha=0.25, line_width=0.7))
                layers.append(hv.Curve((xs, band[2]), "x", "value", label=g).opts(color=col, line_width=2.5))
            if not layers:
                return pn.pane.Markdown("Pick at least one curve.")
            n_draws = df[df["quantity"] == q]["draw"].nunique()
            names = df.loc[df["quantity"] == q, "x_name"].dropna() if "x_name" in df.columns else []
            xname = names.iloc[0] if len(names) else "x"
            plot = hv.Overlay(layers).opts(height=420, responsive=True, logx=log, show_grid=True,
                                           legend_position="top_left", shared_axes=False,
                                           xlabel=xname, ylabel=q)
            return pn.Column(_pane(plot, 430), pn.pane.Markdown(
                f"Thick line: median. Bands: middle 50% and 90%, from all {n_draws} draws. Thin lines: "
                f"{n} individual draws, each one equally likely curve.", styles={"color": "#666"}),
                sizing_mode="stretch_width")

        return pn.Column(pn.Row(quantity, groups, n_lines, logx), pn.bind(view, quantity, groups, n_lines, logx),
                         sizing_mode="stretch_width")


class ParameterDiagnostics(PosteriorAnalysis):
    """
    Sampler-level view of the fitted parameters: trace plots per chain, rank plots
    (flat = chains agree), densities per chain, R-hat and effective sample size, and a pair
    plot of two parameters. Use it to check convergence or explore correlations.
    """

    columns = param.List(default=["variable", "coord", "chain", "draw", "value"])

    def __call__(self, pipeline, context):
        df = pipeline.data.copy()
        df["name"] = np.where(df["coord"] == "all", df["variable"], df["variable"] + "[" + df["coord"] + "]")
        names = list(dict.fromkeys(df["name"]))
        first = pn.widgets.Select(name="Parameter", options=names, value=names[0])
        second = pn.widgets.Select(name="Pair with", options=["(none)"] + names,
                                   value=names[1] if len(names) > 1 else "(none)")
        self._apply_preset(first=first, second=second)

        def matrix(name):
            return df[df["name"] == name].pivot_table(index="draw", columns="chain", values="value").to_numpy().T

        def view(a, b):
            m = matrix(a)
            chains = m.shape[0]
            rhat = float(az.rhat(m)) if chains > 1 else np.nan
            ess_b, ess_t = float(az.ess(m, method="bulk")), float(az.ess(m, method="tail", prob=(0.05, 0.95)))
            trace = hv.Overlay([hv.Curve((np.arange(m.shape[1]), m[c]), "draw", a, label=f"chain {c}").opts(
                color=PALETTE[c % len(PALETTE)], alpha=0.7, line_width=0.8) for c in range(chains)]).opts(
                height=260, responsive=True, shared_axes=False, legend_position="right", title="Trace")
            ranks = m.ravel().argsort().argsort().reshape(m.shape)
            bins = np.linspace(0, ranks.size, 21)
            rank = hv.Overlay([hv.Histogram(np.histogram(ranks[c], bins)).opts(
                fill_color=PALETTE[c % len(PALETTE)], fill_alpha=0.35, line_alpha=0.6) for c in range(chains)]).opts(
                height=260, responsive=True, shared_axes=False, xlabel="rank of draw among all chains",
                title="Rank plot: flat and overlapping = chains agree")
            dens = hv.Overlay([hv.Distribution(m[c], a).opts(color=PALETTE[c % len(PALETTE)], fill_alpha=0.15)
                               for c in range(chains)]).opts(height=260, responsive=True, shared_axes=False,
                                                              title="Density per chain")
            ok = rhat < 1.01 and ess_b > 400 if chains > 1 else ess_b > 400
            md = (f"**{a}**: mean {m.mean():.4g}, 90% interval {np.quantile(m, 0.05):.4g} to {np.quantile(m, 0.95):.4g}. "
                  f"R-hat **{rhat:.3f}**, bulk ESS **{ess_b:,.0f}**, tail ESS **{ess_t:,.0f}** from {m.size:,} draws "
                  f"({chains} chains). " + ("Converged by the usual rules (R-hat < 1.01, ESS > 400)." if ok else
                                           "Check this one: R-hat >= 1.01 or ESS <= 400."))
            parts = [pn.pane.Markdown(md), pn.Row(_pane(trace, 270), _pane(rank, 270), sizing_mode="stretch_width"),
                     _pane(dens, 270)]
            if b != "(none)" and b != a:
                mb = matrix(b)
                pts = hv.Overlay([hv.Points((m[c], mb[c]), [a, b]).opts(color=PALETTE[c % len(PALETTE)], alpha=0.25, size=3)
                                  for c in range(min(chains, mb.shape[0]))]).opts(
                    height=360, responsive=True, shared_axes=False,
                    title=f"Pair plot (correlation {np.corrcoef(m.ravel(), mb.ravel())[0, 1]:.2f})")
                parts.append(_pane(pts, 380))
            return pn.Column(*parts, sizing_mode="stretch_width")

        return pn.Column(pn.Row(first, second), pn.bind(view, first, second), sizing_mode="stretch_width")


ANALYSES = [PosteriorProbability, JointProbability, PosteriorDistribution, DrawCurves, CredibleIntervals,
            UncertaintyFan, BudgetWhatIf, ParameterDiagnostics]
