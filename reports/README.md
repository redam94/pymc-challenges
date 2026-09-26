# Posterior reports with Lumen and a local LLM

Interactive, reader-driven reports built from the posteriors of five example notebooks,
with [Lumen](https://lumen.holoviz.org/) and a small Gemma 4 model served locally by
[Ollama](https://ollama.com/). No data leaves your machine.

| model | notebook | what the reader can explore |
|---|---|---|
| E33 | hierarchical geo MMM | budget what-if sliders per market and channel; the chance a re-split beats today's plan |
| E36 | MRP | every state's estimate against a 55,000-person benchmark; joint majorities across states |
| E37 | heatwave attribution | return-level fan charts for two climates; the chance of a 100 °F summer |
| E40 | 2016 election forecast | state intervals against the result; the "blue wall" as a joint event |
| E41 | growth charts | centile lines with their own uncertainty; place a child on the chart |

## Run it

```bash
uv sync --extra reports                      # Lumen, Panel, hvPlot, DuckDB
ollama pull gemma4:e2b-it-qat                # 4.3 GB on disk, about 2 GB loaded; the smallest Gemma 4
uv run python tools/build.py E33 E36 E37 E40 E41   # once: writes reports/posteriors/<id>.nc (~8 min)
uv run --extra reports python reports/build_warehouse.py --refresh

# 1. the report: a tab per model, live widgets, LLM-phrased summaries
uv run --extra reports panel serve reports/report_app.py --show     # http://localhost:5006/report_app
# 2. the explorer: ask questions in plain English (experimental with a 2B model)
uv run --extra reports panel serve reports/explorer_app.py --show
```

`POSTERIOR_LLM=none` runs the report with no LLM (the summaries become the computed fact
sheets). `POSTERIOR_LLM=gemma4:e4b-it-qat` swaps in the 4B model. Gemma's "thinking" is off
by default (it made every call about 10x slower); `POSTERIOR_THINK=1` turns it back on. On an
8 GB budget, run one app at a time and never while a notebook is sampling.

## The stories

Each tab opens with a story for a lay reader (`posterior_lumen/stories.py`): a headline, then
numbered steps, each a short paragraph, a display built for that example, and a one-line
"how to read it". Every number in the text is computed from the draws when the page is built.

| tab | displays |
|---|---|
| E40 election | choropleth of each state's win chance with a **forecast / what happened** toggle; 100 simulated elections as dots (red = Trump wins); an animated map of 12 plausible election nights from joint draws; a fan chart of the campaign against the result; "where the polls missed" dumbbells |
| E36 opinion | choropleth of the share who agree; a value-suppressing map that colours a state only when a majority is at least 80 in 100 certain; raw poll vs model vs 55k-person check for the 15 smallest samples; a 100-person icon array for any state; an animated map of plausible worlds |
| E37 heatwave | 78 summers as dots with 2021 standing out; a fan chart of the chance of a 100 °F summer year by year (50/80/95% bands from 4,000 draws); 100-summer icon grids in three climates; 100 "explanations" of 28 June 2021 |
| E33 marketing | cost of the next customer as 20 equally likely dots per market and channel; today → recommended budget arrows; response-curve fans with "you are here"; 100 possible futures |
| E41 growth | a growth chart whose centile lines carry their own uncertainty, with sliders to place a child and 20 equally likely answers; why a bell-curve chart flags the wrong children |

Maps use Plotly's built-in US state shapes, which the browser fetches from cdn.plot.ly.
Colours follow the validated reference palette: blue, orange and aqua for identities, one
blue ramp for magnitudes, and blue ↔ red through grey for the election.

## The full posterior

Each of the five notebooks ends with a `save_posterior(...)` cell
(`src/pymc_challenges/export.py`), which writes `reports/posteriors/<id>.nc` with two parts:

- `posterior`: every chain and draw of the fitted parameters (4 x 1000). Observation-sized
  deterministics are left out.
- `derived/<name>`: the decision quantities over all the draws the notebook computed them on:
  4,000 for most, 1,000 for E36's poststratified states and E41's centile curves, and 400 for
  E33's response curves.

`build_warehouse.py` turns these into `posterior_draws` (2.6M rows; curves carry an `x`),
`posterior_params` (2.3M rows of raw chain/draw values for every parameter with at most 200
elements) and `param_summary` (percentiles, R-hat, bulk and tail ESS for all 4,444 parameter
elements). The build takes about 16 s and peaks at 2.1 GB. The largest R-hat across all five
models is 1.009, and the smallest bulk ESS is 583. Without the `.nc` files it falls back to
the few hundred draws in the JSON page exports.

## Uncertainty displays

| analysis | shows | applies to |
|---|---|---|
| `PosteriorProbability` | histogram of draws, P(above/below a threshold) you type | `posterior_draws` |
| `JointProbability` | P(several groups all above t), draw by draw, vs the naive product | `posterior_draws` |
| `PosteriorDistribution` | ridgeline densities with 50/90% bars, cumulative curves, quantile dotplot, table | `posterior_draws` |
| `DrawCurves` | spaghetti of individual draws over 50/90% bands (response curves, return levels, centiles) | `posterior_draws` with `x` |
| `CredibleIntervals` | ranked intervals, optionally against a reference (actual result, benchmark) with coverage | any `q05/median/q95` table |
| `UncertaintyFan` | fan chart of a summarised curve; mark a point and see where it falls | curve summary tables |
| `BudgetWhatIf` | E33 spend sliders with today/recommended/cautious presets | `mmm_response_curves` |
| `ParameterDiagnostics` | trace, rank plot, per-chain density, R-hat/ESS, pair plot | `posterior_params` |

## How it fits together

```
notebooks (E33 E36 E37 E40 E41)
   │  save_posterior(): full posterior (.nc)   page export: summaries (.json)
   ▼
reports/posteriors/*.nc + results/*.json ──build_warehouse.py──► reports/posteriors.duckdb
                                                 posterior_draws    model, quantity, group, x, draw, value
                                                 posterior_params   variable, coord, chain, draw, value
                                                 posterior_summary / param_summary  q05 … q95, R-hat, ESS
                                                 quantities, models, one wide table per model
                                                 (every table and column carries a description)
   ▼
Lumen ── SQL / chat agents (Gemma via Ollama) ──► pick tables, write SQL, write captions
     └─ posterior analyses (plain NumPy) ────────► probabilities, intervals, what-ifs, with widgets
```

The main design choice is a split of duties. A 2-billion-parameter model can turn "how
likely is X" into a `WHERE quantity = ...` and choose a display, but it should not do
probability arithmetic or summarise intervals in its head. So:

- **Posteriors are stored as draws, in one long table.** Any probability is a SQL average,
  `AVG(CASE WHEN value > t THEN 1 ELSE 0 END)`, and joint events join on `draw`. The E40
  "blue wall" query shows why draws beat summaries: over 4,000 joint draws, the model's
  chance that Clinton wins PA, MI and WI together (0.88) is well above the product of the
  separate chances (0.81), because the model knows polling errors are shared.
- **The numbers people act on come from deterministic analyses**
  (`posterior_lumen/analyses.py`): `PosteriorProbability`, `JointProbability`,
  `CredibleIntervals`, `UncertaintyFan`, `BudgetWhatIf`. Lumen offers each one whenever the
  current table has the columns it needs. Each returns its own widgets, so readers keep
  exploring without another LLM call.
- **The LLM gets the rules it would otherwise break** (`posterior_lumen/app.py`): find
  quantity names in `quantities` first; never add or average `q05`/`q95` across rows;
  compute sums per draw; say probabilities as "about N in 100"; quote only numbers in the
  data. The table descriptions in `posterior_lumen/catalog.py` are written for a small model.

## The two apps

**Report** (`report_app.py`, built in `posterior_lumen/report.py`): a Lumen `Report` with
one `Section` per model: a `Narrate` summary, `SQLQuery` tables, and `PosteriorView` actions
(analyses with preset widgets), followed by the "Full posterior" views. The summaries are
*facts first*: Python computes every number from the draws, and Gemma only phrases that fact
sheet, which is shown underneath. A 2B model that reads a table itself mixes up percentiles.
Lumen executes the report, but `report_app.py` displays it in its own page (a tab per model,
"Run again" and "Download as notebook"). Served on its own, Lumen 1.3's Report widget never
signals "document ready" to Panel, even when empty, so its play button and auto-run do
nothing.

**Explorer** (`explorer_app.py`): Lumen's `ExplorerUI` over the same warehouse, with Gemma
2B planning each answer. Out of the box, a 2B model breaks Lumen's agents in predictable
ways, so `posterior_lumen/app.py` adds guard rails. Each one fixes a failure seen in testing:

| failure with Gemma 2B | guard rail |
|---|---|
| writes its own SQL for "what is the chance / how much" and gets it wrong (NaN, invented filters) | a `look_up_answer` tool: finds the model from the question's words and returns its precomputed probabilities and fact sheet; the ChatAgent must quote them |
| calls a tool with invented argument names (`table_slug_list`) and the run crashes | `forgiving()`: near-miss names map to the real parameters; other errors go back to the model as text |
| misspells a table in the SQL (`posterior_drawds`) on every retry | `ForgivingDuckDBSource` corrects near-miss table names after FROM/JOIN (similarity at least 0.85) |
| puts its output's name in the "tables used" field and fails validation | that field snaps to the closest real table |
| asks a clarifying question even when the question is specific, then waits | `PosteriorPlanner` skips Lumen's ambiguity check (`POSTERIOR_CLARIFY=1` restores it) |
| thinks for about 8 s before every call | `reasoning_effort="none"` (`POSTERIOR_THINK=1` restores it) |

Tested with Gemma 2B (about 40-100 s per question on an M-series laptop):

- *What is the chance the recommended E33 budget beats the current plan?* → "about 9 in 10 (0.92)", with the caveat
- *How much hotter is a once-in-100-years heatwave in Seattle now than in the past climate?* → "about 2.7 °C (likely 1.2 to 4.1)"
- *Which ad channel gives the cheapest new customer?* → Sweden Google, about €28 (likely €22-40)
- *What is the chance Clinton wins the national vote?* → 98 in 100
- *List the states with the widest 90% intervals in the 2016 election forecast* → correct SQL and table
- *Show the posterior draws of clinton_state_share for Pennsylvania, Michigan and Wisconsin and apply JointProbability* → the joint-probability view
- *Show the heatwave return levels and apply UncertaintyFan* → the fan chart

Phrase requests for data as "show / list …" and questions about findings as "what is the
chance / how much …". Naming an analysis ("apply JointProbability") runs it. The 4B model
(`gemma4:e4b-it-qat`, 3.8 GB loaded) made the same SQL mistakes in testing, so the guard
rails matter more than model size here. The SQL shown in the editor is what the model wrote;
any typo correction happens when it runs.

## Adding a model

1. In its notebook, export draws (a few hundred equally likely samples) and summaries as
   JSON, as E33-E41 do (`.scratch/artifact/E3x.json`).
2. Copy the export to `reports/results/`, or use `build_warehouse.py --refresh` to copy it.
3. In `build_warehouse.py`, add a builder that returns `(model row, draws, wide tables)`.
4. Describe every new table and quantity in `catalog.py` (the build refuses undescribed tables).
5. Optionally add a `Section` to `report.py`.
