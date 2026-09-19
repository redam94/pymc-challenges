# Authoring guide

How notebooks in this repository are written, built and verified. Read this before adding
or changing one.

## Layout

```
notebook_src/examples/E0x_*.py        jupytext (py:percent) sources - the ONLY files you edit
notebook_src/challenges/C0x_*.py
src/pymc_challenges/hints/C0x.yaml    tiered hints + numeric checks for a challenge
src/pymc_challenges/data.py           dataset registry (real data only, cached in data/)
notebooks/                            BUILD OUTPUT - never edit by hand
tools/build.py                        source -> notebooks (executes examples and solutions)
tools/inspect_nb.py                   print outputs / dump figures of an executed notebook
```

Exemplars to imitate: `notebook_src/challenges/C01_golf_putting.py` with
`src/pymc_challenges/hints/C01.yaml`, and `notebook_src/examples/E01_bayesian_workflow.py`.

## Build and verify

```bash
uv run python tools/build.py C03            # builds challenge + executed solution for C03*
uv run python tools/inspect_nb.py notebooks/solutions/C03_x_solution.ipynb /some/dir/for/figs
```

A notebook is **not done** until: the build succeeds; every `assert h.check(...)` passes;
you have read the text outputs (r_hat, ESS, divergences) and *looked at the figures*; and
every sentence of commentary agrees with what the outputs actually show. Never describe a
result you have not seen.

## Memory: this machine has about 8 GB to spare

Running several samplers at once has crashed the whole session before. Rules, no exceptions:

- Run EVERY Python job (prototype scripts and builds) through the guard, which kills the whole
  process tree if it crosses the cap and prints the peak otherwise:
  `.venv/bin/python tools/memguard.py --max-gb 3 -- .venv/bin/python tools/build.py E13`
- One Python job at a time. Never start a second while one is running; no background jobs.
- nutpie runs its chains as threads in one process (memory is shared) - prefer it. The
  multiprocess samplers (`nuts_sampler="pymc"`, PGBART for BART, `pm.sample_smc`, Pathfinder
  with `parallel=True`) copy the model into one worker PER CHAIN: pass `cores=2` to them.
- Keep what you store small: `pm.sample(var_names=[...])` to drop big deterministics,
  `compute_p=False`, thin before `pm.compute_log_likelihood` on large data, `del` big
  idata objects you no longer need (a notebook kernel keeps everything alive to the end).
- **Where notebook memory actually goes** (measured with `tools/memprofile_nb.py`, which
  reports the kernel's memory cell by cell):
  - `az.loo(idata, pointwise=True)` keeps the full importance-weight matrix on its result
    (`log_weights`, draws x observations: 96 MB for 4000 x 3000) and `az.compare` copies it.
    After LOO: `loo.log_weights = None` and `del idata["log_likelihood"]`. `az.compare`,
    `az.plot_khat` and the printed summary still work. This alone took E03 from 3.2 to 1.2 GB.
  - Do not keep a dict of full fits when you only need their LOO results or a few summaries.
  - `some_pytensor_expression.eval()` on big arrays compiles a function whose intermediate
    storage is never released (2 GB for a 4000 x 4000 kernel): use NumPy for one-off numerics.
  - `pcolormesh` holds every quad: slice to the visible window before plotting.
  - Multiprocess samplers (`pm.sample_smc`, PGBART, PyMC NUTS): `cores=2`. nutpie with
    `adaptation="flow"` trains one JAX flow per chain: `cores=1` (2.4 GB vs over 3.4 GB).
    For timing, use nutpie's per-chain `runtime_ms` from the progress callback, which stays
    meaningful when chains run in turn. `jax.clear_caches()` once the JAX part is over.
- If the guard kills a job (exit 137), make the job smaller - do not raise the cap.
- A notebook whose build peaks above ~3 GB is a bug for a reader with a laptop; say the
  measured peak in your report.

## One source, two notebooks

Cell tags in the challenge source decide where a cell goes:

| tag | challenge notebook | solution notebook |
|---|---|---|
| *(none)* | yes | yes |
| `task` | yes | no |
| `solution` | no | yes |

```python
# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task2")
# h.check("task2", slope=...)

# %% [markdown] tags=["solution"]
# ### Solution
# Why this model...

# %% tags=["solution"]
with pm.Model() as m: ...

# %% tags=["solution"]
assert h.check("task2", slope=idata.posterior["slope"].mean())
```

Consequences: untagged code cells run in the solution too, so they may only depend on
untagged cells before them. Solution cells may define helpers that later solution cells
use. Nothing in an untagged cell may depend on a solution cell.

## What a challenge must be

- **Real data and a real decision.** Open with a brief from a stakeholder; end with a task
  that turns the posterior into an answer to their question (a decision, forecast, ranking,
  what-if), not just a parameter table.
- **Tasks say what to deliver, not how.** 5-7 tasks that escalate. No model code in task
  text. Formulas only when deriving them is not the point.
- **At least one thing goes wrong** that the user must diagnose: divergences, a misleading
  PPC, non-identifiability, label switching, a prior that dominates, a confounder,
  overdispersion... The naive first model should be visibly inadequate.
- **Challenging.** Assume the reader has done E01-E06. Do not spoon-feed; that is what
  hints are for.
- Header table with Difficulty (★ out of 5), Time, Data, Skills - same format as C01.
- End with "Going further" (2-4 open extensions) and `h.progress()`.
- Solution commentary explains *why* (prior choices, why the fix works, what the result
  means for the stakeholder), in short markdown cells.

## What an example must be

Two series live in `notebook_src/examples/`: **E** (modelling techniques, E01-E06 foundations,
E07+ advanced) and **D** (data work: preparation, data problems, exploratory analysis and
reporting). Both follow the same rules.

Fully worked and narrated, same header table as E01, no tags. Teaches one family of
techniques on a real dataset with the complete workflow (prior predictive -> fit ->
diagnostics -> PPC -> interpretation). Shows a failure and its fix where that is the
lesson (e.g. centred vs non-centred). Ends with "Try it yourself" (3 modifications).

## Hints YAML

Three levels per task, always in this order: **nudge** (a question or concept), **approach**
(modelling idea + the relevant PyMC/ArviZ API), **skeleton** (code with `___` blanks - never
the complete answer). Task 0 style EDA tasks may have fewer.

```yaml
title: "..."
tasks:
  task1:
    title: Short title
    hints:
      - |
        nudge...
      - |
        approach...
      - |
        ```python
        skeleton with ___ blanks
        ```
    checks:                      # optional; only for well-defined numbers
      slope:
        what: posterior mean of the slope, about -0.26 per foot
        range: [-0.28, -0.23]
```

Ranges must be generous enough to pass with any reasonable priors and a different seed,
yet tight enough to catch a wrong model or wrong units. Set them from the executed
solution. Every check must be exercised by an `assert h.check(...)` in a solution cell.

## Conventions

- Standard first code cell: imports, `RANDOM_SEED`, `rng`, `az.style.use("arviz-variat")`,
  and for challenges `h = Hints("C0x"); h.tasks()`.
- Load data only through `data.load(name)`, preceded by `data.describe(name)`.
- Pass `random_seed=RANDOM_SEED` to every sampling call.
- Use `coords`/`dims` for anything with a meaningful axis. Use `pm.Data` when you will predict.
- Default `pm.sample()` settings (4 chains, 1000 draws) unless there is a stated reason. The
  default warm-up is **400 steps under nutpie**, 1000 under `nuts_sampler="pymc"` (verified);
  report `idata.posterior.attrs["tuning_steps"]` rather than assuming.
  Note that `pm.sample()` **uses nutpie by default** here because nutpie is installed; pass
  `nuts_sampler="pymc"` for PyMC's own NUTS. Divergence counts differ between the two, so
  say which sampler a claim refers to.
- **Runtime budget: a solution or example should execute in under ~6 minutes** on an
  8-core laptop. Subsample large data (and say so) rather than blow the budget.
- matplotlib for custom figures; ArviZ for diagnostics. End plotting cells with `;`.
- Plain ASCII hyphens in prose are fine; keep lines under ~100 characters.

## PyMC 6 / ArviZ 1 cheat-sheet (verified in this environment)

Installed: pymc 6.3.2, arviz 1.3.0, pytensor 3.3, nutpie, pymc-extras 0.15, preliz, patsy,
scipy, pandas 3, seaborn, h5py, and the JAX stack (jax 0.11, numpyro 0.22, diffrax 0.7,
equinox 0.13, flowjax 19, optax), and pymc-bart 0.13.
`pytensor.wrap_jax` exists for bringing JAX functions into a graph. **Do not add
dependencies** without asking. HDF5 datasets (LIGO strain) are opened with
`h5py.File(data.path(name))`; `data.load` is for tables only.

Things that changed relative to older tutorials - trust this list over memory, and when in
doubt run `uv run python -c "..."` to check:

- `pm.sample`, `pm.sample_prior_predictive`, `pm.sample_posterior_predictive` return an
  xarray **`DataTree`**, not `InferenceData`. `idata.posterior["x"]`, `idata.sample_stats`,
  `list(idata.children)` work. There is **no `idata.extend`**: use `idata.update(other)` or
  `pm.sample_posterior_predictive(idata, extend_inferencedata=True)`.
  `pm.compute_log_likelihood(idata)` adds the `log_likelihood` group in place.
- `pm.MutableData` / `pm.ConstantData` are gone: `pm.Data`. For out-of-sample prediction
  give the likelihood `shape=mu.shape`, then `pm.set_data({...}, coords={...})` and
  `pm.sample_posterior_predictive(idata, predictions=True)` -> `pred.predictions["y"]`.
- `az.summary(idata, var_names=[...], ci_kind="hdi", ci_prob=0.94, round_to=2)`; the default
  is an 89% equal-tailed interval (columns `eti89_lb`, `eti89_ub`) with 2 significant figures.
- `az.extract(idata, var_names=[...], group="posterior", num_samples=...)` -> stacked
  `sample` dimension. `az.hdi(dataarray, prob=0.94)` -> last dim `ci_bound` = `lower`/`upper`.
- `az.loo(idata)`, `az.compare({"a": idata_a, "b": idata_b})` (columns include `elpd`,
  `elpd_diff`, `dse`, `weight`, `p`). Also `az.loo_pit`, `az.reloo`, `az.loo_kfold`, `az.psense`.
- Plot names: `plot_dist` (was plot_posterior), `plot_trace`, `plot_trace_dist`,
  `plot_rank`, `plot_forest`, `plot_pair`, `plot_energy`, `plot_ess`, `plot_prior_posterior`,
  `plot_ppc_dist` (was plot_ppc; `group="prior_predictive"` for prior), `plot_ppc_pit`,
  `plot_ppc_tstat(idata, t_stat="max")`, `plot_ppc_rootogram` (counts), `plot_ppc_pava`
  (binary calibration), `plot_ppc_censored`, `plot_ppc_interval`, `plot_loo_pit`, `plot_khat`,
  `plot_compare`, `plot_lm`. There is **no** `plot_posterior`, `plot_ppc`, `plot_hdi`,
  `plot_kde`. Plots return a `PlotCollection` (not axes); for custom figures use matplotlib
  directly with `np.quantile` / `az.hdi` bands.
- **Gotchas found while building these notebooks** (each one cost somebody an hour):
  - HSGP (or any basis expansion) on a `pm.Data` input is recomputed at every gradient
    evaluation: ~30x slower. Sample inside `with freeze_dims_and_data(model):`
    (`from pymc.model.transform.optimization import freeze_dims_and_data`) and keep the
    original model for `pm.set_data` prediction. See E05 and C08.
  - Ordered cutpoints/means need starting values, but `initval=` on the variable makes
    `pm.compute_log_likelihood` raise `NotImplementedError`. Use `pm.sample(initvals={...})`.
  - `pm.sample_prior_predictive` ignores the `ordered` transform - sort prior draws yourself.
  - `pm.OrderedLogistic(..., compute_p=False)` or it stores an (obs x categories) array per draw.
  - After `pm.do(...)`, `pm.set_data` fails with a shape error: rebuild the model on the new
    rows instead (see C09). `pm.sample(var_names=[...])` avoids storing huge deterministics.
  - `pm.Censored` around `pm.HurdleGamma` evaluates but nutpie cannot initialise it.
  - `pymc_extras.marginalize` + `recover` work on PyMC 6.3.2 (`recover_marginals` is a
    deprecated alias). See E04.
  - `az.plot_ppc_rootogram` is slow and unreadable when counts are in the thousands; use
    interval coverage / `az.plot_ppc_pit`. `az.plot_ppc_pit(method="envelope")` raises TypeError.
  - `az.kaplan_meier` only accepts a DataTree; for a quick KM curve compute it in NumPy.
  - `az.compare` rounds to 2 significant figures by default: pass `round_to=1` or close
    models all print the same elpd.
  - **`pm.Uniform("x", 40, 100)` with INTEGER bounds gives a NaN initial point** and sampling
    dies with "All initialization points failed" ((40, 100) and (50, 90) fail; (0, 100)
    works). Always write bounds as floats: `pm.Uniform("x", 40.0, 100.0)`.
  - `nuts_sampler_kwargs=` is deprecated: pass `pm.sample(nuts={...})`. nutpie low-rank mass
    matrix is `nuts={"adaptation": "low_rank"}`. Whole-model JAX: `pm.sample(backend="jax")`.
  - nutpie jitters start points by U(-1, 1) on the unconstrained scale - enough to throw a
    multimodal model (ODEs, oscillators, chirps) into a wrong mode. `initvals=` is honoured.
    nutpie is not always bit-reproducible at a fixed seed: quote round numbers in prose.
  - `nuts_sampler="pymc"` with a JAX-backed Op can deadlock under fork on macOS ARM (0% CPU
    forever): use `mp_ctx="spawn"` or `cores=1`.
  - `pt.switch` leaks NaN gradients from the branch it does not take (sqrt, division at the
    switch point). Use a series expansion or another smooth form. See E08's `sinn`.
  - A wrapped phase prior such as `Uniform(-pi, pi)` gives r_hat ~ 2 with ZERO divergences:
    parameterise by quadratures (A cos phi, A sin phi). See E10.
  - `az.plot_rank`'s envelope assumes independent draws, so autocorrelated but healthy chains
    (r_hat 1.00) get flagged: thin to roughly the ESS first. `method="envelope"` works there.
  - `idata.posterior[["a", "b"]]` raises `NotImplementedError` on a DataTree: select one
    variable at a time or use `az.extract`. `idata.isel(draw=slice(None, None, 10))` thins
    every group at once.
  - `pm.sample_smc` works; `sample_stats["log_marginal_likelihood"]` is an object array
    (chain, stage) padded with NaN - the last finite entry per chain is the estimate.
  - PyTensor 3.3 Ops implement `pullback` (`L_op`/`grad` still work but warn). `wrap_jax` on
    an input declared with `dims=` loses static shape: `pt.specify_shape` at the boundary.
  - `pm.gp.hsgp_approx.approx_hsgp_hyperparams` is unusable for 2-D fields with a wide
    lengthscale range (asks for m > 50 per axis): tabulate approximation error instead (E07).
  - **Fork after JAX = deadlocks.** Once JAX is imported in a process, anything that forks
    can hang at 0% CPU or die oddly on macOS: `pmx.fit_pathfinder` (use `parallel=False`;
    `cores=1` does not help), `nuts_sampler="pymc"` (use `mp_ctx="spawn"`), and one report of
    `pm.sample_smc` raising `EOFError` (it works fine in a JAX-free process - verified).
  - nutpie normalizing-flow adaptation needs BOTH backends on JAX:
    `pm.sample(nuts={"adaptation": "flow"}, compile_kwargs={"backend": "jax",
    "gradient_backend": "jax"})`, or `nutpie.compile_pymc_model(m, backend="jax",
    gradient_backend="jax").with_transform_adapt(num_layers=..., max_epochs=...)`. Anything
    else fails with "All initialization points failed". Defaults take minutes on 100-200
    dimensions and warm-up dominates. See E12.
  - A per-chain LIST in `pm.sample(initvals=[...])` silently switches from nutpie to PyMC's
    NUTS. Pass one dict. nutpie jitter is switched off with `compile_kwargs={"jitter_rvs": set()}`.
  - PSIS in ArviZ 1: `xr.DataArray(...).azstats.psislw(dim=...)` expects NEGATIVE log-ratios
    (pass `-(logp - logq)`). `az.wasserstein(joint=True)` (the default) is an n x n linear
    program - minutes at 2000 draws; use `joint=False`.
  - `pmx.fit_laplace`: `chains=` raises; in a non-centred hierarchy the mode sits at an absurd
    scale with `success=False` - the mode depends on the parameterisation.
    `pmx.fit_pathfinder` reports `pareto_k` only with `importance_sampling="psis"`.
  - ADVI on an ODE model goes NaN within a few iterations from the default start sd of 1: use
    `pm.ADVI(start=..., start_sigma={...: 0.05})` plus `total_grad_norm_constraint`.
  - `verify_grad` can fail from finite-difference error alone on badly scaled inputs: check
    on the unconstrained scale and compare `jax.jacfwd` with `jax.jacrev` instead.
  - A Bayesian neural network's weight r_hat is meaningless (permutation/sign symmetries):
    diagnose in function space - r_hat/ESS of predictions on a grid and of the log-likelihood.
  - **A discrete free variable makes `pm.sample` silently drop nutpie** for a multiprocess
    `CompoundStep` (NUTS + Metropolis/Gibbs) - one model copy per chain, which matters on an
    8 GB machine. Marginalise it instead (E04, D02), or pass `cores=2`.
  - Automatic imputation (NaN / masked array / pandas Series in `observed`): creates
    `y_unobserved` (free), `y_observed`, and a Deterministic `y` in the original row order;
    only `y` keeps your `dims`; the log-likelihood (so LOO) covers `y_observed` only; nutpie
    handles it. `pm.Data` containing NaN raises `NotImplementedError`. `az.rhat`/`az.ess` on
    the stitched `y` give NaN warnings (mostly constants) - exclude it. See D02.
  - `pymc_extras.marginalize` raises `NotImplementedError` if a dependent variable uses
    automatic NaN imputation: split the rows and write the missing values as an explicit RV.
  - `pm.CustomDist(dist=lambda mu, sigma, size: pt.round(pm.Normal.dist(mu, sigma, size=size)))`
    gets a correct automatically derived interval log-probability (rounding / heaping).
  - `az.psense_summary` needs `log_likelihood` AND `log_prior` groups: `pm.compute_log_likelihood`
    plus `pm.stats.compute_log_prior` (the top-level `pm.compute_log_prior` is deprecated).
    `az.ci_in_rope(ci_prob=1.0)` raises "Too few elements". `az.plot_loo_pit(method="envelope")`
    raises TypeError like `plot_ppc_pit`. `az.compare` accepts a dict of stored `az.loo`
    results, so fitted trees can be deleted first.
  - PreliZ 0.28: `pz.maxent(..., ax=ax)` silently ignores `ax` (use `plot_kwargs={"ax": ax}`);
    `pz.maxent`/`pz.quartile` return `(dist, ax)`, or just `dist` with `plot=False`.
  - pandas 3: string columns are Arrow-backed - `.values` gives an `ArrowStringArray` that
    xarray coords reject (use `.to_numpy(dtype=object)`); `.str.split().str[0]` returns
    `object`; `pd.Categorical(values, categories=...)` with unseen values warns and will
    raise - use `pd.Index(levels).get_indexer(values)` (-1 for unseen; and remember -1
    silently indexes the LAST level if you forget to check).
  - A dict in `idata.attrs` makes `to_netcdf` fail (`Object dtype has no native HDF5
    equivalent`): JSON-encode it. `sample_stats` tree depth is `depth` under nutpie,
    `tree_depth` under PyMC's sampler.
  - `pm.Simulator`: parameters arrive as shape-(1,) arrays and `size` is the observed shape;
    `ndim_supp=1` with scalar parameters raises `NotImplementedError`; the documented
    `distance="kullback_leibler"` raises `NotImplementedError`. `pm.sample_smc` `sample_stats`
    (`beta`, `accept_rate`, `log_marginal_likelihood`) are RAGGED when chains take different
    numbers of stages (1-D object array of lists) - do not `np.array(..., float)` them.
    Final-stage SMC-ABC acceptance below ~2% means a degenerate population: check
    `accept_rate` per stage, not only r_hat. See E14.
  - `pymc_extras.statespace` (E13): give parameters as `pm.Deterministic`, NOT `pm.Data`, or
    post-estimation fails with "Cannot sample from flat variable". Put the prior variances of
    the initial level / seasonal / regression states in `P0` so the filter integrates them out
    (NUTS then samples only the innovation sds: ~3x faster). `mvn_method="cholesky"` fails
    with constant states - keep the default `"svd"`. Thin the posterior and name the outputs
    (`filter_output_names=[...]`) before any post-estimation step: smoothed covariances alone
    were 346 MB per 1000 draws. The per-period log-likelihood terms are one-step-ahead
    densities, so `az.loo` on them is NOT leave-one-out. A Kalman gradient costs ~2 ms and
    recompiles per model: budget 40-50 s a fit. The data must have a `DatetimeIndex` with a freq.
- Available and worth using where they fit: `pm.ZeroSumNormal`, `pm.Censored`,
  `pm.Truncated`, `pm.CustomDist`, `pm.Mixture`, `pm.NormalMixture`, `pm.OrderedLogistic`,
  `pm.LKJCholeskyCov`, `pm.GaussianRandomWalk`, `pm.AR`, `pm.gp.HSGP`, `pm.gp.HSGPPeriodic`,
  `pm.ICAR`, `pm.do`, `pm.observe`, `pm.Potential`, `pm.fit` (ADVI), `pm.math.invprobit`,
  `pymc_extras.marginalize`, `pymc_extras.fit_laplace`.
