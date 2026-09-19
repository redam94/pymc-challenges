# PyMC Challenges

Worked examples and **hard, real-data challenges** for Bayesian modelling with the current
PyMC stack (**PyMC 6 · ArviZ 1 · PyTensor 3**). Every dataset is real. Every challenge ends
in a decision somebody actually has to make, and in every one the obvious first model
breaks in a way you have to diagnose.

Hints are there if you want them - tiered, opt-in, and tracked, so you know how much help
you really needed.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # creates .venv with PyMC 6, ArviZ 1, nutpie, JAX, JupyterLab, ...
uv run python -m pymc_challenges # optional: (re)download all datasets into data/
uv run jupyter lab notebooks
```

Using VS Code or another IDE instead? Point it at the interpreter in `.venv`.

## How to use this

```
notebooks/
  examples/      E01-E15  modelling techniques; D01-D03 data preparation and analysis. Worked, narrated, executed.
  challenges/    C01-C10  your workspace. Tasks, empty cells, hints on request.
  solutions/     C01-C10  full reference solutions, executed, with commentary.
```

1. Work through the **examples** first if you are new to PyMC 6 - the API has moved
   (`DataTree` instead of `InferenceData`, `pm.Data`, new ArviZ plot names). **E01** has a
   then-vs-now table.
2. Open a **challenge**. Each task tells you *what to deliver*, never how.
3. Resist the solutions folder. Use the hint ladder instead.

### Hints, checks and progress

Each challenge notebook creates a helper `h`:

```python
h.tasks()                          # list tasks, and how many hints / checks each has
h.hint("task2")                    # reveal the NEXT hint for task2
h.hint("task2", level=3)           # or jump to a level
h.check("task2", sigma_angle_deg=1.5)   # compare your number with the reference solution
h.progress()                       # hints used, checks passed
h.reset()                          # start over
```

Hints always come in three levels, revealed one at a time:

| level | name | what you get |
|---|---|---|
| 1 | **nudge** | a question or a concept - where to look |
| 2 | **approach** | the modelling idea spelled out, and the PyMC/ArviZ API to use |
| 3 | **skeleton** | code structure with the key parts left blank |

A good target: solve a challenge on nudges alone. Checks use generous ranges (different
priors and seeds will pass) but will catch a wrong model or wrong units. Your hint usage is
stored locally in `.progress/` (git-ignored).

## Curriculum

### Examples

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E01** | The Bayesian workflow in PyMC 6 | Howell !Kung census | `coords`/`dims`, `pm.Data`, prior & posterior predictive checks, `DataTree`, diagnostics, out-of-sample prediction |
| **E02** | Hierarchical models | Minnesota radon survey | pooling, shrinkage, divergences, centred vs non-centred, group-level predictors |
| **E03** | GLMs and model comparison | Bangladesh arsenic wells | logistic regression, interpretable priors, calibration checks, PSIS-LOO |
| **E04** | Changepoints and discrete latents | UK coal-mining disasters | why NUTS cannot do discrete, compound steps, marginalisation |
| **E05** | Gaussian processes with HSGP | Motorcycle crash test | GP priors, exact vs Hilbert-space GPs, heteroskedastic noise |
| **E06** | Survival analysis and censoring | Mastectomy survival | `pm.Censored`, Weibull AFT, survival curves, prior sensitivity |

E01-E06 are the foundations the challenges assume. E07 onwards are advanced, self-contained case
studies (E11 onwards are research-frontier material: expect honest negative results alongside the wins):

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E07** | Geo-temporal: induced earthquakes in Oklahoma | USGS catalogue, every M3+ quake 2005-19 | log-Gaussian Cox process, 2-D HSGP, space-time interaction so a hot-spot can move, taming priors behind an exp link, maps, hold-out forecast |
| **E08** | Physics-based models: weighing the universe | Union2.1, 580 type Ia supernovae | the theory *is* the regression function, a differentiable integral inside the model, a perfect H0-M degeneracy, P(expansion is accelerating) |
| **E09** | Your own JAX code, with gradients, inside a model | Hudson's Bay lynx-hare pelts | hand-written Op + `jax.vjp`, `pytensor.wrap_jax`, whole-model JAX (nutpie/NumPyro), a JAX ODE solver, multimodal likelihoods and start points |
| **E10** | Signal in the noise: GW150914 | Real LIGO Hanford + Livingston strain (GWOSC) | PSD estimation and whitening, exact time-domain likelihood, black-hole ringdown -> remnant mass and spin, Bayes factor signal vs noise, an honest chirp-mass failure |
| **E11** | Scientific ML: a neural network inside a differential equation | Hudson's Bay lynx-hare pelts | universal / neural ODEs in JAX, priors over functions, why weight-space r_hat is meaningless, did the network rediscover mass-action?, physics vs flexibility on a held-out forecast |
| **E12** | The inference frontier: when plain NUTS is not enough | S&P 500 latent volatility (from C07) | a reference posterior, low-rank and normalizing-flow adapted NUTS, Laplace / ADVI / Pathfinder, Pareto k-hat as a trust diagnostic when you have no reference |
| **E13** | State-space models: let the Kalman filter integrate out the states | UK road casualties 1969-84 and the 1983 seat-belt law | the Kalman recursion in 10 lines, `pymc_extras.statespace`, 3 sampled parameters instead of 400, decomposition, one-step-ahead checks, a counterfactual with a negative control, missing data and forecasting for free - and an honest speed comparison |
| **E14** | Simulation-based inference: Bayes without a likelihood | S&P 500 returns; 114 years of Canadian lynx | rejection ABC calibrated against a known posterior, tolerance and summary statistics as the two approximations, `pm.Simulator` + SMC, the g-and-k distribution, Wood's synthetic likelihood, a population model that *cannot* produce the cycle |

### Data work (D-series)

Most "my model will not sample" problems are data problems. These are the craft around the model:

| | Topic | Real data | What you will take away |
|---|---|---|---|
| **D01** | From a raw table to model-ready arrays | Palmer penguins field records, IBM Telco | auditing and the unit of analysis, index vs dummy vs sum-to-zero coding *as prior choices*, centring vs standardising (measured), back-transforming with uncertainty, scaler leakage, the retransformation trap, collinearity before the sampler finds it, Bernoulli -> Binomial aggregation, a validated `prepare()`, saving fits |
| **D02** | Imperfect data as part of the model | Palmer penguins, US divorce rates with standard errors | missing outcomes and covariates (complete-case vs mean-fill vs joint model, under a controlled deletion), marginalising a missing discrete covariate, a missing-not-at-random experiment and what identifies a selection model, measurement error and shrinkage, outliers vs data errors, rounding and heaping |
| **D03** | Bayesian data analysis, end to end | six datasets for EDA, then penguins | exploratory plots that choose the likelihood (mean-variance slopes), priors from domain knowledge with PreliZ, why "weak" priors are not weak, estimands on the outcome scale, ROPE, prior sensitivity, a multiverse table, and a stakeholder report generated from the fit |

### Challenges

| | Challenge | Real data | ★ | You will have to |
|---|---|---|---|---|
| **C01** | Golf putting: physics beats curve fitting | Berry (1996) + Broadie PGA putts | 2 | build a likelihood from geometry; work out why a perfectly sensible model will not mix on the bigger dataset |
| **C02** | Premier League forecasting | football-data.co.uk 2024-26 | 3 | rescue a team-strength model whose first version samples badly; simulate the rest of a season; score yourself against the bookmakers |
| **C03** | Bike-share demand | Capital Bikeshare daily counts | 3 | catch a model with perfect diagnostics and useless intervals; model non-linear weather effects; provision a fleet from a predictive quantile |
| **C04** | Customer churn as survival | IBM Telco churn | 3 | handle censoring; design a PPC for censored data; target a retention offer by expected value |
| **C05** | Likert scales are not numbers | Trolley-problem moral judgements | 3 | replace linear regression with an ordered-logistic model; interpret on the outcome scale |
| **C06** | How many galaxy clusters? | Corona Borealis velocities | 3 | explain why four healthy-looking chains disagree; choose the number of clusters; report co-clustering probabilities |
| **C07** | Stochastic volatility and VaR | S&P 500 daily returns | 4 | sample a latent volatility path without being fooled by a silent sampler failure; backtest a 99% Value-at-Risk |
| **C08** | Forecasting atmospheric CO2 | NOAA Mauna Loa record | 4 | build an additive GP; work out why a stationary GP cannot forecast a trend; tame non-identified components with priors; date a threshold crossing |
| **C09** | Does job training work? | LaLonde NSW experiment + CPS | 4 | g-computation with `pm.do`; find out why a confident, clean-sampling estimate is wrong; compare with the experimental benchmark |
| **C10** | Who draws fouls in the NBA? | Last Two Minute reports 2015-21 | 5 | fit an item-response model with ~1500 player effects; repair a colleague's broken model; measure (not assume) what reparameterising buys; rank with uncertainty |

Roughly in order of difficulty, but they are independent - pick what is closest to your work.
The descriptions are deliberately vague about *what* goes wrong: finding out is the challenge.

## Data

All data is real and public; sources and citations are in
[`src/pymc_challenges/data.py`](src/pymc_challenges/data.py) and printed by
`data.describe("name")`. Files are cached in `data/` and re-downloaded on demand if missing.

## Running on a laptop with limited memory

Multiprocess samplers copy the model once per chain, and a notebook kernel keeps everything
alive until it exits. If memory is tight, run builds (or any script) under the guard, which
kills the whole process tree above a cap and otherwise reports the peak:

```bash
uv run python tools/memguard.py --max-gb 3 -- python tools/build.py C10
```

Every notebook here has been measured this way and peaks below 3 GB (most between 1 and 2 GB).
If your own solution to a challenge balloons, `tools/memprofile_nb.py` reports memory cell by
cell. The usual culprit is model comparison: after `loo = az.loo(idata, pointwise=True)`, free
the two big arrays with `loo.log_weights = None` and `del idata["log_likelihood"]`.

## Rebuilding the notebooks

Notebooks are generated from jupytext sources in `notebook_src/`; a challenge and its
solution come from **one** source file so they cannot drift apart.

```bash
uv run python tools/build.py            # rebuild and re-execute everything (slow)
uv run python tools/build.py C03 E02    # just these
uv run python tools/build.py --no-execute
```

See [AUTHORING.md](AUTHORING.md) for the conventions, and for a verified cheat-sheet of what
changed in PyMC 6 / ArviZ 1.
