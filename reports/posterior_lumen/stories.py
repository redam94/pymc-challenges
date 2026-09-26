"""Story pages: one per example, told for a lay reader with displays made for that example.

Each ``*_story()`` returns a Panel layout of steps: a headline, a short paragraph that says
what to look at, the display, and a one-line "how to read it". Every number is computed from
the posterior draws in the warehouse (or the notebook's page export) when the page is built.

Colours follow one scheme throughout (the validated reference palette of the dataviz
guidance): blue ``#2a78d6`` / orange ``#eb6834`` / aqua ``#1baf7a`` for identities, one blue
ramp for magnitudes, blue <-> red through grey for the election. Maps use Plotly's built-in
US state shapes (the browser fetches them from cdn.plot.ly).
"""

import json

import duckdb
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from .catalog import RESULTS_DIR, WAREHOUSE

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GREY, INK, MUTED, SURFACE = "#c9c8c4", "#0b0b0b", "#52514e", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0, "#b8302f"], [0.25, "#ee8f8d"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1, "#1c5cab"]]
FONT = dict(family="Inter, Roboto, Helvetica, Arial, sans-serif", size=13, color=INK)


# ---------------------------------------------------------------- helpers

def _con():
    return duckdb.connect(str(WAREHOUSE), read_only=True)


def _sql(query, params=None):
    with _con() as con:
        return con.execute(query, params or []).df()


def _results(model_id):
    return json.loads((RESULTS_DIR / f"{model_id}.json").read_text())


def _draws(model_id, quantity, group="all"):
    return _sql("SELECT draw, value FROM posterior_draws WHERE model_id = ? AND quantity = ? AND group_name = ? "
                "ORDER BY draw", [model_id, quantity, group])["value"].to_numpy(float)


def _layout(fig, height, **kw):
    kw.setdefault("margin", dict(l=10, r=10, t=30, b=10))
    fig.update_layout(template="plotly_white", font=FONT, height=height, paper_bgcolor=SURFACE,
                      plot_bgcolor=SURFACE, hoverlabel=dict(font_size=13), **kw)
    return fig


def _plot(fig, height):
    return pn.pane.Plotly(fig, height=height, sizing_mode="stretch_width",
                          config={"responsive": True, "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]})


def _step(number, heading, text, display, how_to_read=None):
    parts = [pn.pane.Markdown(f"#### {number}. {heading}", margin=(20, 5, 0, 5)),
             pn.pane.Markdown(text, styles={"font-size": "1.05em"}, margin=(0, 5)), display]
    if how_to_read:
        parts.append(pn.pane.Markdown(f"*How to read it:* {how_to_read}", styles={"color": MUTED}, margin=(0, 5, 10, 5)))
    return pn.Column(*parts, sizing_mode="stretch_width")


def _headline(big, small):
    return pn.Column(
        pn.pane.Markdown(f"## {big}", margin=(10, 5, 0, 5)),
        pn.pane.Markdown(small, styles={"font-size": "1.1em", "color": MUTED}, margin=(0, 5, 10, 5)),
        sizing_mode="stretch_width")


def _in100(p):
    p = float(p)
    if p >= 0.995:
        return "over 99 in 100"
    if p <= 0.005:
        return "under 1 in 100"
    return f"{round(100 * p)} in 100"


def quantile_dots(draws, n=100, bins=40):
    """Wilkinson-style quantile dotplot: n equally likely outcomes, stacked in bins."""
    q = np.quantile(draws, (np.arange(n) + 0.5) / n)
    lo, hi = q.min(), q.max()
    width = (hi - lo) / bins if hi > lo else 1.0
    b = np.floor((q - lo) / width).clip(0, bins - 1)
    y = pd.Series(b).groupby(b).cumcount().to_numpy() + 1
    return lo + (b + 0.5) * width, y, q


def icon_grid(n_solid, n_maybe, n=100, cols=10):
    """Positions and states for a 10x10 icon array: solid, maybe (uncertain), empty."""
    idx = np.arange(n)
    x, y = idx % cols, cols - 1 - idx // cols
    state = np.where(idx < n_solid, "solid", np.where(idx < n_solid + n_maybe, "maybe", "empty"))
    return x, y, state


def _icons(fig, x, y, state, color, row=None, col=None, symbol="circle", size=17, names=("", "", "")):
    styles = {"solid": dict(color=color), "maybe": dict(color=color, opacity=0.35), "empty": dict(color="#e4e3df")}
    for key, name in zip(["solid", "maybe", "empty"], names):
        m = state == key
        if m.any():
            tr = go.Scatter(x=x[m], y=y[m], mode="markers", name=name, showlegend=bool(name),
                            marker=dict(symbol=symbol, size=size, line=dict(width=0), **styles[key]),
                            hoverinfo="skip")
            fig.add_trace(tr, row=row, col=col) if row else fig.add_trace(tr)


def _no_axes(fig, **kw):
    """Hide axes; the fixed ranges give every icon a square cell with room around it."""
    off = dict(visible=False, showgrid=False, zeroline=False, showspikes=False, range=[-0.7, 9.7])
    fig.update_xaxes(**off, **kw)
    fig.update_yaxes(**off, **kw)


# ---------------------------------------------------------------- E40 election

def election_story():
    d = _results("E40")
    states = _sql("SELECT * FROM election_states ORDER BY state")
    ev = _draws("E40", "clinton_electoral_votes")
    p_win = (ev >= 270).mean()

    # 1. choropleth: chance Clinton wins each state, toggle to the result
    won = (states.actual_share > 0.5).astype(float)
    hover = [f"<b>{r.state}</b> ({r.electoral_votes} electoral votes)<br>Chance Clinton wins: {_in100(r.p_clinton_win)}"
             f"<br>Forecast Clinton share: {100 * r.median:.0f}% (likely {100 * r.q05:.0f}-{100 * r.q95:.0f}%)"
             f"<br>Actual: {100 * r.actual_share:.1f}% ({'Clinton' if r.actual_share > 0.5 else 'Trump'} won)"
             for r in states.itertuples()]
    fig = go.Figure(go.Choropleth(
        locations=states.abbr, z=states.p_clinton_win, locationmode="USA-states", colorscale=DIVERGING,
        zmin=0, zmax=1, text=hover, hoverinfo="text", marker_line_color="white", marker_line_width=0.8,
        colorbar=dict(title="Chance<br>Clinton<br>wins", tickvals=[0, .25, .5, .75, 1],
                      ticktext=["0 in 100", "25", "50", "75", "100 in 100"], len=0.8)))
    fig.update_layout(geo=dict(scope="usa", projection_type="albers usa", bgcolor=SURFACE, lakecolor=SURFACE),
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.08, xanchor="left", buttons=[
                          dict(label="The forecast (7 Nov 2016)", method="restyle",
                               args=[{"z": [states.p_clinton_win.tolist()]}]),
                          dict(label="What happened", method="restyle", args=[{"z": [won.tolist()]}])])])
    map1 = _plot(_layout(fig, 470, margin=dict(l=0, r=0, t=40, b=0)), 470)

    # 2. 100 simulated elections as dots
    x, y, q = quantile_dots(ev, 100, bins=30)
    fig = go.Figure()
    for mask, color, name in [(q >= 270, BLUE, "Clinton wins"), (q < 270, RED, "Trump wins")]:
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name=name,
                                 marker=dict(size=13, color=color, line=dict(color="white", width=1.5)),
                                 text=[f"{v:.0f} electoral votes for Clinton" for v in q[mask]], hoverinfo="text"))
    fig.add_vline(x=270, line=dict(color=INK, width=1.5, dash="dash"),
                  annotation_text="270 to win", annotation_position="top")
    fig.add_vline(x=d["actual"]["ev_clinton"], line=dict(color=RED, width=2),
                  annotation_text=f"what happened: {d['actual']['ev_clinton']}", annotation_position="bottom left")
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="Clinton's electoral votes")
    dots = _plot(_layout(fig, 330, legend=dict(orientation="h", y=1.15)), 330)
    n_trump = int((q < 270).sum())

    # 3. animated map: plausible election nights (joint draws, spread from low to high EV)
    shares = _sql("SELECT group_name AS state, draw, value FROM posterior_draws "
                  "WHERE model_id = 'E40' AND quantity = 'clinton_state_share'")
    wide = shares.pivot(index="draw", columns="state", values="value")
    abbr = dict(zip(states.state, states.abbr))
    evs = dict(zip(states.state, states.electoral_votes))
    order = np.argsort(ev)
    picks = order[np.linspace(0, len(order) - 1, 12).astype(int)]
    frames = []
    for k, draw in enumerate(picks):
        row = wide.loc[draw]
        c_ev = int(sum(evs[s] for s in row.index if row[s] > 0.5))
        frames.append(go.Frame(name=str(k), data=[go.Choropleth(
            locations=[abbr[s] for s in row.index], z=(row.values > 0.5).astype(float), locationmode="USA-states",
            colorscale=[[0, RED], [1, BLUE]], zmin=0, zmax=1, showscale=False, marker_line_color="white",
            text=[f"{s}: Clinton {100 * v:.1f}%" for s, v in row.items()], hoverinfo="text")],
            layout=dict(title=dict(text=f"Simulated election night {k + 1} of 12: Clinton {c_ev}, Trump {538 - c_ev}"))))
    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        geo=dict(scope="usa", projection_type="albers usa", bgcolor=SURFACE, lakecolor=SURFACE),
        updatemenus=[dict(type="buttons", x=0, y=0, xanchor="left", yanchor="top", buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=1300), fromcurrent=True)]),
            dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.12, len=0.88, y=0, currentvalue=dict(visible=False),
                      steps=[dict(label=str(k + 1), method="animate",
                                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0))]) for k in range(12)])])
    anim = _plot(_layout(fig, 480, margin=dict(l=0, r=0, t=50, b=40)), 480)

    # 4. fan chart of the campaign
    path = d["national_path"]
    wk = pd.to_datetime(path["week_end"])
    fig = go.Figure([
        go.Scatter(x=wk, y=np.array(path["q95"]) * 100, line=dict(width=0), showlegend=False, hoverinfo="skip"),
        go.Scatter(x=wk, y=np.array(path["q05"]) * 100, fill="tonexty", fillcolor="rgba(42,120,214,0.2)",
                   line=dict(width=0), name="9 in 10 range", hoverinfo="skip"),
        go.Scatter(x=wk, y=np.array(path["median"]) * 100, line=dict(color=BLUE, width=2.5), name="best guess",
                   hovertemplate="%{x|%d %b}: Clinton %{y:.1f}%<extra></extra>")])
    fig.add_hline(y=50, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_hline(y=100 * d["actual"]["national_share"], line=dict(color=RED, width=2),
                  annotation_text=f"actual national result: {100 * d['actual']['national_share']:.1f}%",
                  annotation_position="bottom right")
    fig.update_yaxes(title="Clinton share of the two-party vote (%)")
    fan = _plot(_layout(fig, 360, legend=dict(orientation="h", y=1.12)), 360)

    # 5. where the polls missed
    s = states.assign(miss=100 * (states.actual_share - states["median"])).sort_values("miss")
    s = s[s.electoral_votes >= 10]
    fig = go.Figure()
    for r in s.itertuples():
        fig.add_trace(go.Scatter(x=[100 * r.median, 100 * r.actual_share], y=[r.state, r.state], mode="lines",
                                 line=dict(color=GREY, width=3), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=100 * s["median"], y=s.state, mode="markers", name="forecast",
                             marker=dict(size=11, color="white", line=dict(color=INK, width=2)),
                             hovertemplate="%{y}: forecast %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=100 * s.actual_share, y=s.state, mode="markers", name="result",
                             marker=dict(size=11, color=np.where(s.actual_share > 0.5, BLUE, RED)),
                             hovertemplate="%{y}: result %{x:.1f}%<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(title="Clinton share of the two-party vote (%)")
    miss = _plot(_layout(fig, 560, legend=dict(orientation="h", y=1.06)), 560)
    big_miss = s.head(4).state.tolist()

    return pn.Column(
        _headline(f"The model gave Clinton {_in100(p_win)}. Trump won.",
                  "A poll-of-polls forecast the day before the 2016 election, and what it teaches about "
                  "uncertainty: an 88% favourite loses about one time in eight."),
        _step(1, "Where each candidate was expected to win", (
            "Blue states were likely Clinton, red likely Trump; the paler the state, the closer the call. "
            "Press **What happened** to see the result: the pale-blue Rust Belt states flipped."), map1,
              "hover a state for its chance and forecast share."),
        _step(2, "100 elections the model thought possible", (
            f"Each dot is one equally likely election from the model. Clinton passes 270 in most of them, "
            f"but Trump wins **{n_trump}** of these 100. The real result, {d['actual']['ev_clinton']} votes "
            f"for Clinton, was one of the model's unlucky dots, not something it ruled out."), dots,
              "count the red dots: that is Trump's chance, about 1 in 8."),
        _step(3, "Twelve plausible election nights", (
            "Press play to flip through simulated election nights, from Clinton's worst to her best. "
            "States move together: when she loses Pennsylvania she usually loses Michigan and Wisconsin too, "
            "because a polling error in one tends to be an error in all of them."), anim,
              "each frame is one draw from the model; all twelve were plausible on 7 November."),
        _step(4, "How the national picture moved", (
            "Clinton led all year in the model's best guess. The shaded band is the range it thought likely for "
            "her national vote share; the red line is what she got: inside the band, but below the best guess. "
            "A national miss of about one point, concentrated in a few Midwestern states, decided the Electoral "
            "College."), fan, "the line is the best guess week by week; the band is 9-in-10 likely."),
        _step(5, "Where the polls missed", (
            f"Open circles are the forecast, filled dots the result (states with 10+ electoral votes). "
            f"The biggest misses, all towards Trump, were in {', '.join(big_miss)}. The model's ranges "
            f"contained the actual result in only {100 * states.inside_90.mean():.0f} of 100 states: its "
            f"uncertainty was too small."), miss, "the longer the grey line, the bigger the polling miss."),
        sizing_mode="stretch_width")


# ---------------------------------------------------------------- E36 MRP

def mrp_story():
    d = _results("E36")
    st = _sql("SELECT * FROM mrp_state_opinion ORDER BY state")
    nat = _draws("E36", "national_share_agree")
    lo, mid, hi = np.quantile(nat, [0.05, 0.5, 0.95]) * 100

    hover = [f"<b>{r.state}</b><br>About {100 * r.median:.0f}% agree (likely {100 * r.q05:.0f}-{100 * r.q95:.0f}%)"
             f"<br>People surveyed here: {r.n_survey}<br>Chance a majority agrees: {_in100(r.p_majority_agree)}"
             for r in st.itertuples()]
    fig = go.Figure(go.Choropleth(
        locations=st.abbr, z=100 * st["median"], locationmode="USA-states",
        colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)], text=hover, hoverinfo="text",
        marker_line_color="white", marker_line_width=0.8, colorbar=dict(title="% who<br>agree", len=0.8)))
    fig.update_layout(geo=dict(scope="usa", projection_type="albers usa", bgcolor=SURFACE, lakecolor=SURFACE))
    map1 = _plot(_layout(fig, 450, margin=dict(l=0, r=0, t=10, b=0)), 450)

    # value-suppressing: call a majority only where the model is sure
    cat = np.select([st.p_majority_agree >= 0.8, st.p_majority_agree <= 0.2], ["Majority agrees", "Majority disagrees"],
                    "Too close to call")
    colors = {"Majority agrees": BLUE, "Majority disagrees": ORANGE, "Too close to call": GREY}
    fig = go.Figure()
    for c, col in colors.items():
        m = cat == c
        fig.add_trace(go.Choropleth(locations=st.abbr[m], z=np.ones(m.sum()), locationmode="USA-states",
                                    colorscale=[[0, col], [1, col]], showscale=False, name=f"{c} ({m.sum()})",
                                    showlegend=True, marker_line_color="white", text=np.array(hover)[m],
                                    hoverinfo="text"))
    fig.update_layout(geo=dict(scope="usa", projection_type="albers usa", bgcolor=SURFACE, lakecolor=SURFACE),
                      legend=dict(orientation="h", y=1.05))
    map2 = _plot(_layout(fig, 450, margin=dict(l=0, r=0, t=30, b=0)), 450)

    # raw poll vs model, smallest samples first
    s = st.sort_values("n_survey").head(15).iloc[::-1]
    fig = go.Figure()
    for r in s.itertuples():
        if pd.notna(r.raw_survey_share):
            fig.add_annotation(x=100 * r.median, y=r.state, ax=100 * r.raw_survey_share, ay=r.state, xref="x",
                               yref="y", axref="x", ayref="y", arrowhead=2, arrowwidth=1.5, arrowcolor=GREY,
                               showarrow=True, text="")
    fig.add_trace(go.Scatter(x=100 * s.raw_survey_share, y=s.state, mode="markers", name="raw survey",
                             marker=dict(size=6 + 1.2 * np.sqrt(s.n_survey), color="white", line=dict(color=MUTED, width=2)),
                             text=[f"{n} people surveyed" for n in s.n_survey],
                             hovertemplate="%{y}: raw survey %{x:.0f}% (%{text})<extra></extra>"))
    fig.add_trace(go.Scatter(x=100 * s["median"], y=s.state, mode="markers", name="model (with 9 in 10 range)",
                             marker=dict(size=11, color=BLUE),
                             error_x=dict(type="data", symmetric=False, array=100 * (s.q95 - s["median"]),
                                          arrayminus=100 * (s["median"] - s.q05), color=BLUE, thickness=2, width=0),
                             hovertemplate="%{y}: model %{x:.0f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=100 * s.benchmark_share, y=s.state, mode="markers", name="check: 55,000-person survey",
                             marker=dict(size=11, symbol="diamond", color=ORANGE),
                             hovertemplate="%{y}: large-survey check %{x:.0f}%<extra></extra>"))
    fig.update_xaxes(title="% who agree", range=[0, 100])
    shrink = _plot(_layout(fig, 560, legend=dict(orientation="h", y=1.08)), 560)
    wy = st.set_index("abbr").loc["WY"]

    # icon array for any state
    state = pn.widgets.Select(name="Pick a state", options=st.state.tolist(), value="Wyoming", width=260)

    def icons(name):
        r = st.set_index("state").loc[name]
        solid, top = int(round(100 * r.q05)), int(round(100 * r.q95))
        fig = go.Figure()
        x, y, stt = icon_grid(solid, top - solid)
        _icons(fig, x, y, stt, BLUE, symbol="circle", size=19,
               names=(f"agree in almost every plausible world ({solid})", f"could go either way ({top - solid})",
                      f"disagree ({100 - top})"))
        _no_axes(fig)
        _layout(fig, 360, legend=dict(orientation="v", x=1.02, y=0.5), margin=dict(l=10, r=10, t=10, b=10))
        fig.update_yaxes(scaleanchor="x")
        text = (f"**{name}: out of every 100 adults, about {100 * r['median']:.0f} agree, and the model is sure "
                f"it is between {solid} and {top}.** {r.n_survey} people from {name} were in the survey.")
        return pn.Column(pn.pane.Markdown(text), _plot(fig, 360), sizing_mode="stretch_width")

    icon_view = pn.Column(state, pn.bind(icons, state), sizing_mode="stretch_width")

    # plausible worlds animation
    draws = _sql("SELECT group_name AS state, draw, value FROM posterior_draws "
                 "WHERE model_id = 'E36' AND quantity = 'state_share_agree'")
    wide = draws.pivot(index="draw", columns="state", values="value")
    abbr = dict(zip(st.state, st.abbr))
    frames = []
    for k, dr in enumerate(np.linspace(0, len(wide) - 1, 10).astype(int)):
        row = wide.iloc[dr]
        n_maj = int((row > 0.5).sum())
        frames.append(go.Frame(name=str(k), data=[go.Choropleth(
            locations=[abbr[s] for s in row.index], z=100 * row.values, locationmode="USA-states",
            colorscale=[[i / 6, c] for i, c in enumerate(BLUE_RAMP)], zmin=25, zmax=65,
            marker_line_color="white", colorbar=dict(title="% agree", len=0.8),
            text=[f"{s}: {100 * v:.0f}%" for s, v in row.items()], hoverinfo="text")],
            layout=dict(title=dict(text=f"Plausible world {k + 1} of 10: a majority agrees in {n_maj} states"))))
    fig = go.Figure(data=frames[0].data, frames=frames, layout=frames[0].layout)
    fig.update_layout(
        geo=dict(scope="usa", projection_type="albers usa", bgcolor=SURFACE, lakecolor=SURFACE),
        updatemenus=[dict(type="buttons", x=0, y=0, xanchor="left", yanchor="top", buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=900), fromcurrent=True)]),
            dict(label="❚❚", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.12, len=0.88, y=0, currentvalue=dict(visible=False),
                      steps=[dict(label=str(k + 1), method="animate",
                                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0))]) for k in range(10)])])
    anim = _plot(_layout(fig, 470, margin=dict(l=0, r=0, t=50, b=40)), 470)

    return pn.Column(
        _headline(f"About {mid:.0f}% of Americans agree - but what does each state think?",
                  f"{d['question']}. One national survey of 5,000 people, turned into 50 state estimates "
                  f"(national: likely {lo:.0f}-{hi:.0f}%)."),
        _step(1, "The map", (
            "Darker states agree more. The survey reached only a handful of people in many states, so the model "
            "learns from everyone: how opinion varies with age, education and ethnicity, and how Republican each "
            "state is, then reweights to each state's census population."), map1,
              "hover a state for its likely range and how many people were surveyed there."),
        _step(2, "Only call it where we are sure", (
            f"A state is coloured only when the model gives a majority at least 80 in 100 (blue) or at most 20 in "
            f"100 (orange). **{int((cat == 'Too close to call').sum())} states stay grey**: an honest map shows "
            f"where the data cannot tell."), map2, "grey does not mean 50-50 opinion; it means we cannot be sure."),
        _step(3, "What the model does with tiny samples", (
            f"The 15 states with the fewest respondents. Open circles are the raw survey (bigger = more people); "
            f"arrows show where the model moved them. Wyoming had only {wy.n_survey} respondents; the model puts "
            f"it at about {100 * wy['median']:.0f}% instead of trusting a handful of people. Orange diamonds are a "
            f"check against a survey eleven times larger that the model never saw."), shrink,
              "the closer the blue dot sits to the orange diamond, the better the estimate."),
        _step(4, "100 people in one state", (
            "Pick a state. Solid figures agree in nearly every world the model considers plausible; pale ones "
            "are the uncertainty."), icon_view, "count solid plus pale: that is the upper end of the likely range."),
        _step(5, "Ten plausible Americas", (
            "Each frame is one equally likely set of 50 state figures from the model. Watch which states flicker: "
            "those are the ones the survey cannot pin down."), anim,
              "states that keep their colour in every frame are the ones we know."),
        sizing_mode="stretch_width")


# ---------------------------------------------------------------- E37 heatwave

def _f(c):
    return np.asarray(c) * 9 / 5 + 32


def heat_story():
    d = _results("E37")
    obs = _sql("SELECT * FROM heatwave_observed ORDER BY year")
    ev = d["event"]

    # 1. every summer's hottest day
    colors = np.where(obs.year == 2021, RED, np.where(obs.year >= 2000, BLUE, "#86b6ef"))
    fig = go.Figure(go.Scatter(x=obs.year, y=_f(obs.annual_max_c), mode="markers",
                               marker=dict(size=np.where(obs.year == 2021, 16, 9), color=colors,
                                           line=dict(color="white", width=1)),
                               customdata=obs.annual_max_c,
                               hovertemplate="%{x}: %{y:.0f} °F (%{customdata:.1f} °C)<extra></extra>"))
    fig.add_hline(y=_f(ev["previous_record_c"]), line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"record before 2021: {_f(ev['previous_record_c']):.0f} °F",
                  annotation_position="bottom left")
    fig.add_annotation(x=2021, y=_f(ev["value_c"]), text=f"28 June 2021: {ev['value_f']:.0f} °F", showarrow=True,
                       arrowhead=0, ax=-90, ay=0)
    fig.update_yaxes(title="hottest day of the summer (°F)")
    timeline = _plot(_layout(fig, 360, showlegend=False), 360)

    # 2. fan chart: chance of a 100 F day, year by year, from the full posterior
    q = _sql("""SELECT x AS year, QUANTILE_CONT(value, 0.025) q025, QUANTILE_CONT(value, 0.1) q10,
                       QUANTILE_CONT(value, 0.25) q25, MEDIAN(value) q50, QUANTILE_CONT(value, 0.75) q75,
                       QUANTILE_CONT(value, 0.9) q90, QUANTILE_CONT(value, 0.975) q975
                FROM posterior_draws WHERE model_id = 'E37' AND quantity = 'p_100F_by_year'
                GROUP BY x ORDER BY x""")
    fig = go.Figure()
    for lo_, hi_, alpha, name in [("q025", "q975", 0.15, "95 in 100"), ("q10", "q90", 0.25, "80 in 100"),
                                  ("q25", "q75", 0.4, "50 in 100")]:
        fig.add_trace(go.Scatter(x=q.year, y=100 * q[hi_], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=q.year, y=100 * q[lo_], fill="tonexty", line=dict(width=0),
                                 fillcolor=f"rgba(235,104,52,{alpha})", name=f"{name} range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.year, y=100 * q.q50, line=dict(color=ORANGE, width=2.5), name="best guess",
                             hovertemplate="%{x}: about %{y:.0f} summers in 100<extra></extra>"))
    fig.update_yaxes(title="summers in 100 with a 100 °F day", rangemode="tozero")
    fan = _plot(_layout(fig, 380, legend=dict(orientation="h", y=1.12)), 380)
    q1950, q2025 = q.iloc[0], q.iloc[-1]

    # 3. 100 summers in three climates
    labels = {"past climate (1.2 C cooler)": "A world 1.2 °C cooler", "2021 climate": "The 2021 climate",
              "2025 climate": "Today's climate (2025)"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(labels.values()), horizontal_spacing=0.06)
    notes = []
    for i, (key, label) in enumerate(labels.items(), start=1):
        p = _draws("E37", "p_100F_per_summer", key)
        lo_, mid_, hi_ = np.quantile(p, [0.05, 0.5, 0.95]) * 100
        x, y, stt = icon_grid(int(round(lo_)), int(round(hi_)) - int(round(lo_)))
        _icons(fig, x, y, stt, ORANGE, row=1, col=i, symbol="square", size=10)
        notes.append(f"{label}: about {mid_:.0f} (likely {lo_:.0f}-{hi_:.0f})")
    for name, style in [("a 100 °F day", dict(color=ORANGE)), ("maybe", dict(color=ORANGE, opacity=0.35)),
                        ("no 100 °F day", dict(color="#e4e3df"))]:  # legend entries, drawn once
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", size=10, **style)), row=1, col=1)
    _no_axes(fig)
    for i in range(1, 4):
        fig.update_yaxes(scaleanchor=f"x{'' if i == 1 else i}", row=1, col=i)
    grids = _plot(_layout(fig, 330, legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=40, b=30)), 330)

    # 4. 100 explanations of 28 June 2021
    p_past = _draws("E37", "p_108F_per_summer", "past climate (1.2 C cooler)")
    p_now = _draws("E37", "p_108F_per_summer", "2021 climate")
    impossible = (p_past == 0) & (p_now > 0)
    share = impossible.mean()
    n_imp = int(round(100 * share))
    x, y, stt = icon_grid(n_imp, 0)
    fig = go.Figure()
    _icons(fig, x, y, stt, RED, size=19, names=(f"impossible without warming ({n_imp})", "",
                                                f"possible but rare ({100 - n_imp})"))
    _no_axes(fig)
    fig.update_yaxes(scaleanchor="x")
    worlds = _plot(_layout(fig, 330, legend=dict(orientation="v", x=1.02, y=0.5)), 330)
    ratio = _draws("E37", "probability_ratio_108F")
    finite = ratio[np.isfinite(ratio)]
    delta = _draws("E37", "intensity_change_c")

    return pn.Column(
        _headline(f"Seattle hit {ev['value_f']:.0f} °F. How much of that was climate change?",
                  "The June 2021 Pacific Northwest heatwave beat Sea-Tac's record by 5 °F. We fit an extreme-value "
                  "model whose hottest days shift with global temperature, then compare worlds."),
        _step(1, "78 summers, one outlier", (
            "Each dot is the hottest day of one summer at Seattle-Tacoma airport since 1948. Hot days have "
            "crept up since 2000 (dark blue), and then 2021 landed far above anything before it."), timeline,
              "hover a dot for its year and temperature."),
        _step(2, "The odds of a 100 °F summer, as the world warmed", (
            f"The chance that a summer brings at least one 100 °F day rose from about "
            f"{100 * q1950.q50:.0f} in 100 in 1950 to about {100 * q2025.q50:.0f} in 100 today. The shaded "
            f"bands show how sure the model is: the darker, the likelier."), fan,
              "follow the orange line; the fan is the uncertainty, widest where data are thin."),
        _step(3, "100 summers in three worlds", (
            "Each square is a summer; filled ones bring a 100 °F day, pale ones might. "
            + ". ".join(notes) + "."), grids, "compare how many squares are filled in each grid."),
        _step(4, "Could 28 June 2021 have happened without warming?", (
            f"The model gives many equally likely explanations of Seattle's weather. In **{n_imp} of 100** of "
            f"them, a 108 °F day was literally impossible in a world 1.2 °C cooler: the climate had a ceiling "
            f"below it. In the rest it was possible, but warming made it about "
            f"{np.median(finite):,.0f} times more likely (median). Either way, a heatwave this rare is now about "
            f"{np.median(delta) * 9 / 5:.0f} °F ({np.median(delta):.1f} °C) hotter than it used to be."), worlds,
              "each circle is one explanation the data allow; red ones say 'impossible without warming'."),
        sizing_mode="stretch_width")


# ---------------------------------------------------------------- E33 marketing

def mmm_story():
    cells = _sql("SELECT * FROM mmm_budget_cells")
    cells["cell"] = cells.market + " " + cells.channel
    chan_color = {"Google": BLUE, "Meta": ORANGE}

    # 1. cost of the next customer: 20 equally likely costs per cell
    costs = _sql("SELECT group_name, value FROM posterior_draws WHERE model_id = 'E33' AND quantity = 'marginal_cost_eur'")
    order = costs.groupby("group_name")["value"].median().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, g in enumerate(order):
        v = costs.loc[costs.group_name == g, "value"].to_numpy()
        qd = np.quantile(v, (np.arange(20) + 0.5) / 20)
        market, channel = g.split("|")
        fig.add_trace(go.Scatter(x=qd, y=[f"{market} {channel}"] * 20, mode="markers", name=channel,
                                 legendgroup=channel, showlegend=channel not in [t.name for t in fig.data],
                                 marker=dict(size=12, color=chan_color[channel], opacity=0.8,
                                             line=dict(color="white", width=1)),
                                 hovertemplate=f"{market} {channel}: €%{{x:.0f}} per customer<extra></extra>"))
    fig.update_xaxes(title="cost of the next new customer (€, log scale: each step doubles)", type="log",
                     tickvals=[20, 40, 80, 160], ticktext=["€20", "€40", "€80", "€160"])
    dots = _plot(_layout(fig, 360, legend=dict(orientation="h", y=1.1)), 360)

    # 2. budget shift: current -> recommended
    c = cells.sort_values("current_plan_eur_per_year")
    fig = go.Figure()
    for r in c.itertuples():
        fig.add_annotation(x=r.recommended_plan_eur_per_year / 1000, y=r.cell, ax=r.current_plan_eur_per_year / 1000,
                           ay=r.cell, xref="x", yref="y", axref="x", ayref="y", arrowhead=2, arrowwidth=2.5,
                           arrowcolor=chan_color[r.channel], showarrow=True, text="")
    fig.add_trace(go.Scatter(x=c.current_plan_eur_per_year / 1000, y=c.cell, mode="markers", name="today",
                             marker=dict(size=11, color="white", line=dict(color=INK, width=2)),
                             hovertemplate="%{y}: today €%{x:.0f}k a year<extra></extra>"))
    fig.add_trace(go.Scatter(x=c.recommended_plan_eur_per_year / 1000, y=c.cell, mode="markers", name="recommended",
                             marker=dict(size=11, color=[chan_color[ch] for ch in c.channel]),
                             hovertemplate="%{y}: recommended €%{x:.0f}k a year<extra></extra>"))
    fig.update_xaxes(title="yearly budget (€ thousand)", tickprefix="€", ticksuffix="k")
    shift = _plot(_layout(fig, 330, legend=dict(orientation="h", y=1.12)), 330)

    # 3. response curves with "you are here"
    rc = _sql("SELECT * FROM mmm_response_curves")
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{m} {ch}" for ch in ["Google", "Meta"]
                                                        for m in ["Denmark", "Sweden", "Norway"]],
                        vertical_spacing=0.16, horizontal_spacing=0.06)
    for i, ch in enumerate(["Google", "Meta"]):
        for j, m in enumerate(["Denmark", "Sweden", "Norway"]):
            g = rc[(rc.market == m) & (rc.channel == ch)].sort_values("spend_eur_per_week")
            col = chan_color[ch]
            fill = "rgba(42,120,214,0.2)" if ch == "Google" else "rgba(235,104,52,0.2)"
            fig.add_trace(go.Scatter(x=g.spend_eur_per_week, y=g.q95, line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"), row=i + 1, col=j + 1)
            fig.add_trace(go.Scatter(x=g.spend_eur_per_week, y=g.q05, fill="tonexty", fillcolor=fill,
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"), row=i + 1, col=j + 1)
            fig.add_trace(go.Scatter(x=g.spend_eur_per_week, y=g["median"], line=dict(color=col, width=2),
                                     showlegend=False,
                                     hovertemplate="€%{x:,.0f}/week → %{y:.0f} customers/week<extra></extra>"),
                          row=i + 1, col=j + 1)
            today = g.spend_today_eur_per_week.iloc[0]
            rec = today * g.recommended_pct_of_today.iloc[0] / 100
            for x_, name, sym in [(today, "today", "circle-open"), (rec, "recommended", "circle")]:
                y_ = np.interp(x_, g.spend_eur_per_week, g["median"])
                fig.add_trace(go.Scatter(x=[x_], y=[y_], mode="markers", showlegend=False,
                                         marker=dict(size=12, color=INK, symbol=sym, line=dict(width=2)),
                                         hovertemplate=f"{name}: €%{{x:,.0f}}/week<extra></extra>"), row=i + 1, col=j + 1)
    fig.update_xaxes(tickprefix="€", tickformat="~s")
    curves = _plot(_layout(fig, 520), 520)

    # 4. 100 possible futures
    gain = _draws("E33", "plan_gain")
    x, y, q = quantile_dots(gain, 100, bins=30)
    fig = go.Figure()
    for mask, color, name in [(q > 0, AQUA, "more customers"), (q <= 0, RED, "fewer customers")]:
        fig.add_trace(go.Scatter(x=x[mask], y=y[mask], mode="markers", name=name,
                                 marker=dict(size=13, color=color, line=dict(color="white", width=1.5)),
                                 text=[f"{v:+,.0f} customers a year" for v in q[mask]], hoverinfo="text"))
    fig.add_vline(x=0, line=dict(color=INK, width=1.5, dash="dash"))
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title="extra new customers next year from the recommended split")
    futures = _plot(_layout(fig, 320, legend=dict(orientation="h", y=1.15)), 320)
    cheap = cells.sort_values("marginal_cost_median").iloc[0]
    dear = cells.sort_values("marginal_cost_median").iloc[-1]

    return pn.Column(
        _headline("Same budget, more customers: move money from Meta to Google",
                  "A marketing-mix model of a Scandinavian clothing brand's paid ads in Denmark, Sweden and Norway, "
                  "learning across markets so the small ones borrow strength from the big one."),
        _step(1, "What the next new customer costs", (
            f"Each dot is one of 20 equally likely costs for the next customer bought through that channel. "
            f"Google is cheaper everywhere: about €{cheap.marginal_cost_median:.0f} in {cheap.market}, against "
            f"about €{dear.marginal_cost_median:.0f} for Meta in {dear.market}. Meta's dots are more spread out: "
            f"we know its cost less well."), dots, "the wider a row of dots, the less sure the model is."),
        _step(2, "The recommended re-split", (
            "Arrows run from today's yearly budget to the model's recommendation. The total stays the same; the "
            "money moves towards the cheaper channel in every market."), shift,
              "arrow colour shows the channel; the arrowhead is the new budget."),
        _step(3, "Why: every channel runs into diminishing returns", (
            "Weekly spend against new customers per week. The curves flatten as you spend more, so the last euro "
            "buys less than the first. Open circles are where each market is today, filled circles the "
            "recommendation; the band is the model's uncertainty."), curves,
              "a steeper slope at the open circle means more customers per extra euro."),
        _step(4, "100 possible futures", (
            f"Each dot is one equally likely outcome of switching to the recommended split. In "
            f"**{int((q > 0).sum())} of 100** it brings more customers, about {np.median(gain):,.0f} a year at "
            f"the middle. The red dots are why the notebook recommends testing before moving large sums: "
            f"this is a model without an experiment."), futures, "green dots are gains, red dots losses."),
        sizing_mode="stretch_width")


# ---------------------------------------------------------------- E41 growth

def growth_story():
    d = _results("E41")
    gc = _sql("SELECT * FROM growth_centiles ORDER BY centile, age_years")
    cent = sorted(gc.centile.unique())
    zs = norm.ppf(np.array(cent) / 100)

    def med(c):
        g = gc[gc.centile == c]
        return g.age_years.to_numpy(), g["median"].to_numpy(), g.q05.to_numpy(), g.q95.to_numpy()

    def chart(age=None, bmi=None):
        fig = go.Figure()
        # shaded centile zones: 3-97 light, 10-90 darker
        for (a, b), color in [((3, 97), "rgba(42,120,214,0.10)"), ((10, 90), "rgba(42,120,214,0.16)")]:
            xa, ya, _, _ = med(a)
            xb, yb, _, _ = med(b)
            fig.add_trace(go.Scatter(x=xb, y=yb, line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xa, y=ya, fill="tonexty", fillcolor=color, line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
        for c in cent:
            x, m, lo, hi = med(c)
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                     fillcolor="rgba(13,54,107,0.25)", line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=m, line=dict(color=BLUE_RAMP[5] if c == 50 else BLUE_RAMP[3],
                                                        width=2.5 if c == 50 else 1.2), showlegend=False,
                                     hovertemplate=f"{c}th centile at %{{x:.1f}} years: BMI %{{y:.1f}}<extra></extra>"))
            ordinal = {1: "st", 2: "nd", 3: "rd"}.get(c % 10 if c % 100 not in (11, 12, 13) else 0, "th")
            fig.add_annotation(x=x[-1], y=m[-1], text=f"{c}{ordinal}", showarrow=False, xanchor="left",
                               font=dict(color=MUTED))
        if age is not None:
            fig.add_trace(go.Scatter(x=[age], y=[bmi], mode="markers", marker=dict(size=16, color=ORANGE,
                                     symbol="star", line=dict(color=INK, width=1)), showlegend=False,
                                     hovertemplate="your child<extra></extra>"))
        fig.update_xaxes(title="age (years)", range=[0, 23.5])
        fig.update_yaxes(title="BMI (kg/m²)")
        return _layout(fig, 460)

    age = pn.widgets.FloatSlider(name="Age (years)", start=0.5, end=21, step=0.5, value=7, width=280)
    bmi = pn.widgets.FloatSlider(name="BMI (kg/m²)", start=12, end=32, step=0.1, value=18.9, width=280)

    def place(a, b):
        ages = _sql("SELECT DISTINCT x FROM posterior_draws WHERE model_id='E41' AND quantity='bmi_centile' ORDER BY x")["x"]
        a_near = float(ages.iloc[(ages - a).abs().argmin()])
        dr = _sql("SELECT group_name, draw, value FROM posterior_draws WHERE model_id='E41' AND quantity='bmi_centile' "
                  "AND x = ?", [a_near]).pivot(index="draw", columns="group_name", values="value")
        dr = dr[[f"centile {c}" for c in cent]].to_numpy()
        # per draw: interpolate the child's z-score between the centile lines (normal-score scale)
        z = np.array([np.interp(b, row, zs, left=-np.inf, right=np.inf) for row in dr])
        heavier = 100 * (1 - norm.cdf(z))
        lo, mid, hi = np.quantile(heavier, [0.05, 0.5, 0.95])
        if mid < 3:
            text = f"**Above the 97th centile at {a:g} years**: fewer than 3 in 100 boys that age have a higher BMI."
        elif mid > 97:
            text = f"**Below the 3rd centile at {a:g} years**: more than 97 in 100 boys that age have a higher BMI."
        else:
            text = (f"**About {mid:.0f} in 100 boys aged {a:g} have a higher BMI** (likely {lo:.0f} to {hi:.0f}). "
                    f"The range is the uncertainty of the chart itself, from {len(z):,} plausible versions of it.")
        x_, y_, qd = quantile_dots(np.clip(heavier, 0, 100), 20, bins=20)
        fig = go.Figure(go.Scatter(x=x_, y=y_, mode="markers", marker=dict(size=14, color=ORANGE,
                                   line=dict(color="white", width=1)), hovertemplate="%{x:.1f} in 100<extra></extra>"))
        fig.update_yaxes(visible=False)
        fig.update_xaxes(title="boys in 100 with a higher BMI (20 equally likely answers)")
        return pn.Column(pn.pane.Markdown(text, styles={"font-size": "1.1em"}), _plot(chart(a, b), 460),
                         _plot(_layout(fig, 200), 200), sizing_mode="stretch_width")

    tool = pn.Column(pn.Row(age, bmi), pn.bind(place, age, bmi), sizing_mode="stretch_width")

    comp = pd.DataFrame(d["model_comparison"])
    fig = go.Figure()
    fig.add_trace(go.Bar(y=comp.model, x=comp.pct_above_97th, orientation="h", name="flagged above the 97th line",
                         marker_color=ORANGE, hovertemplate="%{y}: %{x:.1f}% of boys<extra></extra>"))
    fig.add_trace(go.Bar(y=comp.model, x=comp.pct_below_3rd, orientation="h", name="flagged below the 3rd line",
                         marker_color=BLUE, hovertemplate="%{y}: %{x:.1f}% of boys<extra></extra>"))
    fig.add_vline(x=3, line=dict(color=INK, dash="dash"), annotation_text="should be 3%")
    fig.update_xaxes(title="% of the 7,294 boys outside the line", ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    bars = _plot(_layout(fig, 300, barmode="group", legend=dict(orientation="h", y=1.15)), 300)

    return pn.Column(
        _headline("Is my child's BMI normal? A growth chart that admits what it doesn't know",
                  "BMI of 7,294 Dutch boys aged 0-21, modelled so the whole distribution (middle, spread and the "
                  "stretched heavy side) changes with age."),
        _step(1, "Place a child on the chart", (
            "Move the sliders. The lines are centiles: 97th means 97 in 100 boys that age are below it. Each "
            "line has its own dark band, because the chart is an estimate too; it is widest at the ends, where "
            "fewer boys were measured."), tool, "the star is your child; the dots below are 20 equally likely answers."),
        _step(2, "Why a bell-curve chart gets the tails wrong", (
            "BMI is not bell-shaped: the heavy side stretches out with age. A chart that assumes a bell curve "
            "with constant spread flags the wrong children. Bars show how many boys each model puts outside its "
            "3rd and 97th lines; a well-calibrated chart puts 3% outside each."), bars,
              "the closer both bars are to the dashed line, the more honest the chart."),
        sizing_mode="stretch_width")


STORIES = {"E33": mmm_story, "E36": mrp_story, "E37": heat_story, "E40": election_story, "E41": growth_story}
