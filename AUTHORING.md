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
equinox 0.13, flowjax 19, optax), pymc-bart 0.13, and plotly 7 (added for E16).
`pytensor.wrap_jax` exists for bringing JAX functions into a graph. **Do not add
dependencies** without asking. HDF5 datasets (LIGO strain) are opened with
`h5py.File(data.path(name))` and the county GeoJSON with `json.load(open(data.path(name)))`;
`data.load` is for tables only.

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
  - **Many small fits in one notebook (E17).** `pm.Model(name="x")` prefixes every variable as
    `"x::mu"` in the posterior - handy to tell 50 fits apart, but remember the prefix in
    `az.summary(var_names=...)`. `logging.getLogger("pymc").setLevel(logging.WARNING)` silences
    the per-fit `NUTS[nutpie]: [...]` banner. `pm.Censored(pm.Binomial.dist(...), lower=L,
    observed=L)` gives the discrete left-censored likelihood P(X <= L) for a "fewer than L+1"
    cell; pass `lower=-1` (not `None`) elementwise for the uncensored entries.
  - **Plots beyond matplotlib (E16).** Set `pio.renderers.default = "plotly_mimetype+notebook_connected"`
    in the first cell: the headless build defaults to `"browser"` (nothing stored), the
    `"notebook"` renderer embeds 4.8 MB of plotly.js *per figure*, and `notebook_connected`
    stores ~10 KB plus the data (JupyterLab/VS Code render the mimetype; GitHub shows nothing).
    A plotly choropleth ignores `fitbounds="locations"` when `scope="usa"` is also set - drop
    the scope. Maps need no GIS library: a `matplotlib.collections.PolyCollection` of the
    GeoJSON polygons with `ax.set_aspect(1 / cos(latitude))` is a choropleth. A matplotlib
    `FuncAnimation(...).to_jshtml()` is ~50 KB per frame at dpi 72 - keep frames near 40 and
    `plt.close(fig)` before displaying or the static figure appears too. `az.style` uses a
    constrained layout: `fig.tight_layout()` only warns. Verify interactive output with
    headless Chrome (`--headless --screenshot=... file://page.html`) since `inspect_nb.py`
    only dumps PNGs.
  - **Graphs and areal models (E18).** `pm.ICAR` cannot be forward-sampled
    (`sample_prior_predictive` raises `NotImplementedError`): draw prior samples from the
    eigendecomposition of `pinv(Q)`. It applies one global soft sum-to-zero, so on a
    disconnected graph isolated nodes silently get a flat prior; the BYM2 scaling factor
    (geometric mean of `diag(pinv(Q))`) is NaN with an isolated node, giving logp = -inf.
    Connect the graph or scale per component. `pm.CAR` with `alpha ~ Uniform(0, 1)` piles alpha
    near 1 and the intercept mixes badly. The pymc-examples `scotland_lips_cancer.csv`
    neighbour lists for Tweeddale and Annandale are swapped. Areal random effects give
    Pareto k > 0.7 for a third or more of the areas: use exact K-fold via an `obs_idx` subset.
  - **Precision matrices and covariance structure (E19, E22, E23).** `pm.MvNormal(tau=...)` with a
    non-positive-definite matrix is a silent -inf. nutpie's `sample_stats["divergence_message"]`
    says *why* each divergence happened ("Logp function returned error code" = left the
    support). LKJ(eta=2) in 11 dimensions gives partial correlations a prior sd of 0.40 - a
    dense-graph prior. Inside a model **`pm.LKJCorr`'s value is the Cholesky factor**, not the
    correlation matrix (form `L @ L.T`); using it as a matrix gives "All initialization points
    failed". `pytensor.function` on model RVs draws them at random rather than substituting
    values - use `model.replace_rvs_by_values` when debugging a logp. `pt.ndtri_exp(log_u)` is a
    stable inverse normal CDF with gradients (copulas). `StudentT.logcdf` with free nu costs ~4x
    (incomplete-beta gradients). In factor models the continuous rotation symmetry barely moves
    r_hat (1.03) while loadings are meaningless: check r_hat of the implied covariance or align
    draws. **`pm.compute_log_likelihood` on an `MvNormal` gives ONE value per draw**, so `az.loo`
    sees one observation: compute conditional (leave-one-out) densities from the precision
    matrix yourself. Dense `MvNormal` fits are not bit-reproducible under nutpie (threaded BLAS).
  - **Hand-built log-likelihoods for LOO (E20, E21, E23).** For a `pm.Potential` likelihood or a
    NumPy-computed one: `idata["log_likelihood"] = xr.Dataset({"y": (("chain", "draw", "obs"), ll)})`
    or `xr.DataTree.from_dict({"posterior": ..., "log_likelihood": ...})`, then `az.loo(...,
    var_name=...)`. Sum per-node terms into one variable for a joint per-cell LOO.
    `az.loo(..., pointwise=True)` is needed for `pareto_k`.
  - **Causal / intervention API (E20).** `pytensor.tensor.slinalg.expm` is deprecated: use
    `pytensor.tensor.linalg.expm` (exact gradient). `pm.do(model, {"x": v})` on an observed
    variable needs a replacement of exactly its shape (`np.full(n, v)`).
  - **Discrete latents, mixtures and label switching (E21, E24, E26, E27, E28).** nutpie's default
    400 tuning steps were too few for Dirichlet-membership mixtures (chains still climbing:
    check `sample_stats["logp"]` per chain; `tune=1000`). For label-free r_hat, evaluate the
    density at grid points as an `xr.Dataset` and pass it to `az.rhat`. Latent-space positions:
    r_hat on distances or Procrustes-aligned draws. `transform=ordered` on a positive variable
    chains with log, so the value variable is `x_chain__`. `pmx.marginalize` refuses variables
    with `initval=` (pass `pm.sample(initvals=...)`) and cannot marginalise through advanced
    indexing (`z[site_idx]`) - keep a site x visit matrix. `recover` lives in
    `pymc_extras.marginal`. A marginalised spike-and-slab gives zero divergences yet r_hat
    1.15-1.5: divergences do not detect poor spike/slab mixing; Rao-Blackwellise inclusion
    probabilities in NumPy. `pm.StickBreakingWeights(alpha, K)` returns K+1 weights. A sparse
    Dirichlet (e0 = 0.01) defeats NUTS - log-gamma reparametrised chains can freeze with step
    size ~1e-5 and *zero* divergences; e0 = 0.05 works. The textbook horseshoe with tau ~
    HalfCauchy(1) implies ~61 of 64 active coefficients: simulate m_eff first.
  - **Custom likelihoods (E24, E28).** `pm.CustomDist(..., signature="(),(v),(v)->(v)")` gives a
    per-row log-likelihood (LOO works); its draws come back as float. `pytensor.scan(...,
    return_updates=False)` returns outputs only. `pt.i0` exists (with gradient), `pt.i0e` does
    not. `pm.logsumexp` squeezes its result - use `pt.logsumexp(..., axis=...)`. In an HMM a
    log-emission of 0 marginalises a missing observation, so ragged tracks can be padded.
    `sample_prior_predictive` ignores `pm.Potential` terms (warns); simulate in NumPy.
  - **Mixed models (E25).** With plenty of data per group, zero-mean centred random effects give
    ESS ~250 for the population means (a ridge); putting the means *inside* the MvNormal
    (hierarchical centring) gave ~20x the ESS, and non-centred was worst.
  - **DataTree and misc (E23-E28).** `idata.sample_stats.depth` is DataTree's own `.depth`
    attribute: use `idata.sample_stats["depth"]`. A DataTree node has no `.stack`
    (`.to_dataset().stack(...)`). `az.extract(var_names=...)` needs a list, not a tuple.
    `pm.Model(name=None)` raises; omit it. `pm.sample_posterior_predictive(idata.isel(...),
    extend_inferencedata=True)` extends the thinned copy - use the return value.
    `LKJCholeskyCov(compute_corr=True)` summaries warn (NaN r_hat on the constant diagonal).
    R data without R: `.rda`/`.RData` may be gzip or xz (`\xfd7zXZ`); E23/E24 carry a small
    XDR reader, and R's integer NA reads as -2147483648.
  - **Neural networks and approximate inference (E29).** `pmx.fit_pathfinder(initvals=...)` wants
    untransformed names (a `find_MAP` dict with `x_log__` raises `KeyError`). On a small BNN,
    mean-field / full-rank ADVI and Pathfinder all collapse to the prior predictive with inflated
    noise. Diagnose in function space (predictions on a grid), never weights. A NumPy forward
    pass over prior draws x grid x width explodes (20000 x 200 x 512 = 16 GB): evaluate only
    where you need it.
  - **Random partitions and features (E30, E31).** `gammaln(a + n) - gammaln(a + 1)` cancels
    catastrophically for a >> n: a fake high-logp region (log a ~ 50) with zero divergences and
    only r_hat to show it - sum `log(a + arange(1, n))` instead. A whole-partition `pm.Potential`
    has no per-observation terms, so no LOO: use the held-out predictive `logEPPF(full) -
    logEPPF(train)`, log-mean-exp over draws. An integer parameter (a species-pool size) can be
    marginalised on a grid with `pt.logsumexp` inside the Potential. `pm.sample(initvals=...)`
    needs the model-name prefix (`"name::var"`). A stick-breaking IBP with features summed out
    can collapse to "all features off" from nutpie's default start (no divergences): start at
    data rows, `compile_kwargs={"jitter_rvs": set()}`. A `CustomDist` with signature
    `"(k),(k),(k,d)->(d)"` on a 2-D observed failed in logp; a Potential worked. Seed each
    section's own `default_rng` when the prose quotes numbers.
  - **Covariate-dependent mixtures (C11).** Logistic stick-breaking gates mix differently from run
    to run (per-unit CATE max r_hat 1.01-1.09 with the same seed; worse with many gating
    covariates). A label-invariant quantity can still be poorly identified: report per-chain
    summaries of the decision quantity. `pm.StudentT` wants `nu`, `mu`, `sigma` as keywords.
    pandas 4 warns on `.sum(1)`: write `.sum(axis=1)`. Running a solution as a plain script
    writes `.progress/` unless `PYMC_CHALLENGES_NO_PROGRESS=1`.
  - **Hand-derived likelihoods for discrete outcomes (E32).** PyMC's NB `logcdf` is `log(betainc)`:
    in the upper tail it rounds to 0, `pt.ndtri_exp(0)` is +inf and a copula logp returns NaN on
    a Black-Friday-sized count. Use `F = I_p(alpha, y+1)` AND `1 - F = I_{1-p}(y+1, alpha)` via
    `pt.betainc` and invert whichever is below 1/2; `pt.log1mexp` for log-differences.
    `betainc` parameter gradients dominate the cost: get `F(y-1)` from `F(y) - pmf(y)` (two
    evaluations per row instead of four, 1/3 faster). Unit-test a new logp before sampling:
    reduces to a known case, sums to one on a grid, matches a simulator of the same story
    (total-variation distance), finite-difference gradients, finite in the tails. A
    `CustomDist(random=...)` using scipy makes numba fall back to object mode for posterior
    predictive (a warning, it works). A generalised control that recovered the truth in
    simulation split the real-data chains into two modes (ESS 7, zero divergences): print
    per-chain means.
  - **MMM and media data (E32-E35).** Conjura: NaN spend is documented as "channel not used",
    but feeds also stop (a brand's Meta columns all go NaN mid-series; its Google PMax column
    alternates NaN / spend day to day; UK click columns end early) - check before treating NaN
    as 0. Some organisations carry near-identical UK and US series (corr > 0.999): de-duplicate
    donors. In an MMM the baseline level and the media maximum trade off along a ridge (ESS ~100
    at r_hat 1.03, no divergences). Centred and non-centred hierarchies diverge at opposite ends
    (small vs large between-market sd); plot divergences against log sd, `target_accept=0.95`.
    `pm.sample` mutates a `nuts={...}` dict you pass - do not reuse it with `target_accept=`.
    `az.summary` over several variables sharing a dim can mislabel rows: one variable at a time.
    `az.extract(var_names=[one])` returns a DataArray (no `["name"]`). `az.rhat(...)` returns a
    DataTree without `.to_array()`. `nutpie.compile_pymc_model(m).with_data(name=arr)` swaps
    `pm.Data` without recompiling (69 placebo refits, 5 compilations; mask the likelihood to
    keep shapes). A random-walk level integrated out as MvNormal (cov `s0^2 + s^2 min(s,t)`)
    beat the latent version (142 -> 2 divergences, ESS 180 -> 1300).
  - **Custom JAX transforms in an MMM (E45).** Under nutpie's default Numba backend a
    `wrap_jax` node runs in Numba *object mode* (a UserWarning per compile; 115 us vs 35 us per
    logp+grad when the whole model is JAX, `pm.sample(backend="jax")`) - filter the warning and
    say so. `jnp.where` differentiates both branches: `where(x > 0, exp(k*log(x)), 0)` has a NaN
    gradient at x = 0 - replace the bad input first (double `where`). `jnp.power` is safe for
    d/dk at 0 but d/dx is inf for k < 1. `(1 - exp(-c z)) / (1 - exp(-c))` is 0/0 at c = 0 and
    loses 2% in float32 at c = 1e-6: `expm1` plus a Taylor switch. `pm.sample(idata_kwargs=
    {"log_likelihood": True})` is deprecated (FutureWarning): call `pm.compute_log_likelihood`.
    The arviz-variat style uses constrained layout: `fig.tight_layout()` warns, and raises
    RuntimeError once a colorbar exists. Conjura brand `6b89cf94` (UK apparel) has the same
    mid-October 2023 Meta cut-off as E34/E35 - one partial week (£549, 2% of normal) moved
    Meta's average CAC from £56 to £85: curves are learned from spend extremes, check them.
  - **Chaotic dynamics (E46).** Trajectory matching on the chaotic Ricker map (log r = 3.8, 100
    steps) under nutpie: every chain frozen in its own likelihood spike (r_hat 15, ESS 4, ~1100
    divergences) - more tuning cannot help. The **non-centred** state-space model (innovations
    pushed through `scan`) is just as bad (r_hat inf, 2000 divergences): for chaotic maps sample
    the states themselves - `pm.Flat` states + the process density as a `pm.Potential`, start at
    the log-counts, `nuts={"adaptation": "low_rank"}` (2 s, 0 divergences). A `Potential`
    likelihood has no prior predictive: simulate in NumPy. Marginalising an integer delay with
    `pt.logsumexp` over per-delay log-likelihoods + a `p_tau` Deterministic works cleanly.
    Tangent-map Lyapunov exponents: floor the renormalisation norm (`np.maximum(norm, 1e-300)`)
    or stable grid cells give NaN. A lumped multiplicative noise on the whole population sets a
    control noise floor at sigma and made the blowfly cycles too long (25 vs 19 steps).
  - **Chaotic ODEs and regime models (E47, E48).** Lorenz-63 multiple shooting (100 latent
    states, 5 RK4 steps per interval unrolled in PyTensor): the model-error scale q sits in a
    funnel (HalfNormal prior: 136 divergences; LogNormal(log 0.05, 0.7): 0, but q ESS ~17,
    r_hat 1.17 even with tune=2000) while sigma/rho/beta sample well. The exact-Jacobian
    instantaneous LLE of Lorenz-63 is > 0 at 87% of the attractor (a threshold-0 trigger fires
    almost always). NHMM on Lorenz-84: emissions on (x, y, z) left chains in different modes
    (r_hat 1.7) - use jet + log eddy amplitude; order regimes with the `ordered` transform, not a
    `-inf` Potential (K=4: 1,736 -> 36 divergences); start chains at the best of 10 `find_MAP`
    runs (`model.compile_logp()` needs only value-var keys). `idata.posterior.stack` fails on a
    DataTree node: `.to_dataset().stack(...)`.
  - **Jump diffusions (E49).** Sum the daily jump count out with `logsumexp` (Poisson 0..4, or a
    Bernoulli for SVJ). SV latent log-vol: centred gave r_hat 1.10 / ESS < 80 for s_h; non-centred
    via `scan` (stable AR(1), unlike chaotic maps) + `target_accept=0.9`: r_hat <= 1.011. Compare
    latent-state time-series models with one-step scores from a particle filter, not PSIS-LOO.
    Vectorised systematic resampling: `searchsorted` on row-offset cumsums - a broadcast
    comparison (D x N x N per step) turned a 2-minute cell into > 10 minutes.
  - **Reaction-diffusion PDEs (E50).** Method of lines in `scan`: 480 explicit Euler steps on 38
    finite volumes sample in ~25 s under nutpie; check D dt/dx^2 <= 1/2 over the prior. Scale the
    initial profile by the ESTIMATED capacity (count0 / K) inside the model: using the nominal 122
    moved D by 18% and beta by 15%. Linear PDEs (SDD morphogen, FRAP): diagonalise the fixed discrete
    Laplacian once with `np.linalg.eigh`, then steady states and c(t) are matrix products in PyTensor
    (exact, no time stepping, 4-5 s fits). A no-flux far wall gives a cosh, not exponential, profile.
    Positional error with only upstream (ligand) noise is independent of the receptor Kd - add
    receptor counting noise for Kd to matter. Data transcribed from Julia arrays: keep the CSV in data/.
  - **PDE-based design (E51).** A 288-cell 2-D heat equation inside PyMC via dense
    `pt.linalg.solve` samples in ~20 s. Calibration with temperature rises identifies k, h, eta
    only as ratios (correlations ~0.9); a weakly identified edge coefficient's 89% interval missed
    the truth (prior median 10 vs true 15). Unconstrained greedy heater placement put point
    heaters inside the working area (10 C hot spots): restrict candidates. Use common random numbers
    (fixed sensor noise and scenarios) when comparing sensor layouts, or greedy picks noise. A
    design that runs a heater at its limit leaves the controller nothing to correct with: cap the
    nominal powers below the rating. `boxplot(vert=False)` is deprecated in Matplotlib 3.11:
    `orientation="horizontal"`.
  - **Bulk-surface PDEs (E52).** A 2-D cytosol coupled to a 1-D membrane on a polar
    finite-volume disk (1,025 unknowns): rotational symmetry makes the operator block-diagonal in
    angular cosine modes (put sector centres at multiples of the sector angle, or the Nyquist
    cosine mode vanishes), 33 blocks of 17. Binding/unbinding makes K non-symmetric; scaling the
    membrane rows by koff/kon symmetrises it, then batched `pt.linalg.eigh` gives exact time
    courses with gradients (~60 s per NUTS fit; mode vs dense expm error 1e-13). A "membrane
    only" limit via tiny kon/koff keeps the weighting finite. Membrane-only data still identified
    D_c here. Arrival-at-the-back metrics depend on the arc chosen: check the prose against the
    printed numbers (membrane-only diffusion reached 10% of the back half in 165 s, not the
    30-minute equilibration time). Polar subplots overlap titles under the arviz constrained
    layout: use `fig.subfigures`.
  - **Luria-Delbrück likelihood (E53).** The compound-Poisson pmf is a triangular solve
    `(I - m A) x = p0 e0`; it overflows when p0 is tiny - solve with `exp(shift - a) e0`,
    `shift = max(a - 600, 0)`. Cost grows as N^2: cap the pmf (200) and treat the rest as a
    censored "> cap" class. `log1mexp(logsumexp(lp[lo:]))` with `lo` beyond the vector sums
    nothing, so censored rows silently contribute 0: build the mask over 0..N. Concatenating
    integer edges with a `logspace` that starts at the last edge makes a zero-width bin.
  - **Conductance-based neurons (E54).** Trace matching of spikes with NUTS: r_hat 1.55, and
    the best mode had no spikes (a misplaced spike costs twice a missing one) - fit features. An
    AR(1) noise likelihood conditioned on the first residual drove rho to 1 (tau 6 vs 17 ms, 28
    divergences); a stationary AR(1) + a fast second exponential fixed it. Sentinel values in a
    feature extractor (1e9 for a trough past the record) make likelihood notches and fake modes:
    simulate past the step. `pm.Simulator(distance="gaussian", epsilon=1)` on standardised
    features is exactly a Gaussian feature likelihood; Numba works in SMC's forked workers if
    nothing imported JAX. Numba compiles a second specialisation when a defaulted argument is
    omitted - it skews timings.
  - **Chemical master equation (E55).** `pt.linalg.solve(assume_a="tridiagonal")` still takes a
    dense matrix and has a dense gradient (2.2 ms for 288 systems vs 0.55 ms for
    `jax.lax.linalg.tridiagonal_solve` via `wrap_jax`); PyTensor's JAX backend cannot convert
    `LUFactorTridiagonal`. `LKJCholeskyCov(compute_corr=True)` under nutpie's JAX backend panics
    ("expected chol_stds but found chol_corr"): `compute_corr=False` + `expand_packed_triangular`.
    Time `pytensor.function(mode="JAX")` with `np.asarray` on outputs (async dispatch looks 50x
    faster). A three-term recurrence for a minimal (decaying) solution explodes forwards: solve it
    as a boundary-value problem. Binomial capture thinning in the telegraph model is exactly a
    rescaled synthesis rate - absolute capture efficiency is never identified from counts.
  - **Lineage models (E56).** A `scan` Kalman filter over 279 lineages x 68 generations: ~4 ms
    per gradient, 160 s per fit, no faster under JAX; fixed 8-generation windows as one `MvNormal`
    with closed-form covariance: ~20 s. `az.loo` with several observed `MvNormal`s raises "several
    log likelihood arrays": concatenate into one variable in a hand-built DataTree. HalfNormal
    priors hugging 0 on noise scales froze a chain (1,002 divergences, then PSIS "All tail values
    are the same"): Gamma(2, 50). Parameters near 1e-3 (growth per minute) gave r_hat 1.6: rescale
    units (per hour). Dropping outlier cycles (> 4 MAD) selects on the outcome and moves slopes.
  - **Particle tracks (E57).** The MA(1) displacement covariance (localisation error) is tridiagonal
    Toeplitz, whose eigenvectors are a fixed discrete sine basis: rotate each track once in NumPy
    and the likelihood is a plain `Normal` (200 tracks, 400 parameters, 2 s). Model variables used
    inside a `scan` step must be passed as `non_sequences`, or gradients fail with
    "'RandomGeneratorVariable' object has no attribute 'shape'". scan/batched-Cholesky models ran
    2x faster with `compile_kwargs={"backend": "jax", "gradient_backend": "jax"}`. `0**a` has a
    NaN gradient in a: `exp(a * log(max(x, 1e-300)))`. Hierarchical fBM with free per-track
    amplitudes biased alpha upward on 30-step Brownian tracks: pool the amplitudes. An HMM with
    independent emissions cannot separate a bound state's D from localisation error.
  - **Localisation microscopy (E58).** Batched Fisher matrices via `np.einsum("bpd,bpe,bp->bde")`
    build a (B, pixels, D, D) temporary (~800 MB for 4,000 patches): use
    `np.matmul(np.swapaxes(J * w[..., None], 1, 2), J)`. They can be exactly singular: `pinv`, and
    treat non-positive variance as infinite sd. A Laplace evidence with an emitter whose position no
    data constrain (masked pixels + flat prior) is +inf: use a proper Gaussian position prior.
    `pm.sample_smc(cores=2)` peaked at 2.3 GB here - `cores=1` gave 1.3 GB;
    `compute_convergence_checks=False` silences per-run ESS warnings. On real tissue frames the
    emitter-count model added faint companions in 51% of detections (13% in simulation): forward
    model misfit, not molecules - report it.
  - **Step-detection HMMs (E59).** Scaling the forward pass by the max emission over ALL states
    overflows when an unreachable state has a large emission (0 * inf = NaN energy): take the max
    over reachable predecessors and clip. `pm.CustomDist(signature="(n),()->()")` passes batched
    params to `logp`: `lk.reshape((-1,))[-n:]`. In `xr.Dataset`, a variable named like its own dim
    becomes a coordinate and `az.loo` fails ("list index out of range"). nutpie's start jitter split
    lattice-HMM chains into a noise-absorbs-steps mode: `compile_kwargs={"jitter_rvs": set()}` +
    common `initvals`. Hypoexponential rates without an `ordered` transform: r_hat 1.53, ESS 7.
  - **Inverse problems / TFM (E60).** `pm.HalfCauchy(..., shape=np.int64(n))` raises "shape must be
    tuple/int": cast with `int()` or use dims. A stationary Gaussian prior with white noise gives a
    flat posterior-sd map - it cannot show where the fit is wrong (the error sat at the
    adhesions). On PIV displacement fields a white-prior marginal likelihood drove the noise to
    2e-7 nm (PIV already smoothed the field): compare the data's power spectrum with each model's.
    Dense sufficient statistics (Cholesky of A^T A) let a 1,969-parameter horseshoe sample in ~60 s
    without touching 8k observations. The L-curve may have no corner; max-curvature then depends
    on the lambda range scanned. Nonlinear summaries (strain energy, total |t|) can be biased far
    beyond their posterior width.
  - **Pattern formation / proofreading (E61).** A batched 4x4 `pt.linalg.cholesky`/`solve_triangular`
    over ~1,600 wavenumbers was too slow to finish; a rank-2 Woodbury rewrite (2x2 algebra) ran at
    0.1 ms per gradient. nutpie `adaptation="low_rank"` gave ~150 divergences on a curved posterior;
    default diagonal + `target_accept=0.95`, `tune=1500` gave 0. Summing out a discrete N while
    sampling a rate whose meaning depends on N split chains (r_hat 1.44): parameterise by a quantity
    identified for every N (the local slope). A shell `alarm` that kills memguard orphans its child.
  - **Failure and pivots (E62).** Prior-to-posterior contraction against a vague prior was > 90% for
    an MMM's Google - Meta ROI difference, which three priors then moved from P = 0.65 to 1.00: test
    identification by swapping priors (or `az.psense`), not by contraction. Conjura brand `f7493de0`
    (US apparel) has weekly Google/Meta spend correlated 0.89 - a ready-made collinear example. A
    4-weekly random-walk baseline instead of a linear trend moved total media ROI from 31 to 25 with
    non-overlapping intervals; LOO preferred it by 44 +- 6 (some k-hat > 0.7) - fit, not
    attribution. Exact `pm.gp.Marginal` gradients grow as n^2.7 (168 ms at n = 2000; timing up to
    n = 3000 peaked at 2 GB); an HSGP NegBin on all 17,379 bike hours costs 1.25 ms per gradient but
    took ~8 minutes to sample. `az.extract(pp, var_names=[...])` on a posterior-predictive DataTree
    raises "Can not extract posterior": pass `group="posterior_predictive"`.
  - **Quasi-experiments (E63).** In a triangular IV likelihood (outcome conditioned on the observed
    endogenous regressor) posterior-predictive outcomes are built around the *observed* regressor, so
    the replicated schooling-wage correlation came out near 0 against 0.31: add
    `beta * (e_rep - e_obs)`. The LKJ IV posterior is a beta-rho ridge: nutpie
    `nuts={"adaptation": "low_rank"}` took bulk ESS from ~250 to ~1800; with a weak instrument the
    tails of beta are set by the LKJ eta, not the prior on beta. N(0, 1) on a standardised jump that
    is ~2 sd shrank it by 20%: compare posterior against the raw estimate.
  - **Population PK (E64).** A lag time as `pt.maximum(t - tlag, 0)` has no gradient for samples before
    the lag; nutpie's start jitter left one chain in a spurious lag mode (r_hat 1.13, no divergences):
    `initvals={"tlag": 0.6}` plus `compile_kwargs={"jitter_rvs": {...all free RVs but tlag}}`. The oral
    one-compartment flip-flop (ka <-> ke) is an exact second mode with unit Jacobian: only multi-start
    chains (PyMC NUTS with a list of per-chain `initvals`, `cores=2`) reveal it. Write the solution with
    `expm1` so it stays finite at ka = ke. Many held-out refits: one compiled nutpie model with a 0/1
    likelihood mask in `pm.Data` and `compiled.with_data(...)`.
  - **Hawkes / ETAS (E65).** An exact O(n^2) Omori pair sum cost 3.6 ms per gradient at n = 790 (fit
    ~2 min); writing (t + c)^-p as a trapezoid mixture of 79 exponentials driven by a `pytensor.scan`
    recursion cost 0.4 ms and matched logp to 1e-5. PyTensor 3.3 has no `pt.logcumsumexp`.
    `model.compile_logp()(point)` wants value variables only: a `find_MAP` result (with deterministics)
    raises "Too many parameter passed". Cap forecast cascades (near-critical draws explode) and report
    how many hit the cap. A constant magnitude of completeness after a mainshock biased c, p, b and
    alpha; a time-varying Mc(t) in the likelihood fixed it.
  - **A/B tests at scale (E66).** `pm.sample(var_names=[...])` under nutpie still stored every free
    variable. Student-t effects written non-centred (`tau * StudentT(nu, 0, 1)`) gave tau/nu ESS ~130
    with no divergences; centred `StudentT(nu, 0, tau)` tripled it (≈45 clicks per arm is not weak
    data). A test intercept plus arm effects is a ridge: subtract each test's mean arm effect. A pandas
    `Styler` shows as `<Styler at 0x...>` in `inspect_nb.py`, so print a rounded DataFrame.
    `plt.hist(density=True)` on log-spaced bins misleads: use `weights=1/n`.
  - **MRP (E36).** Group effects as plain `Normal` z plus a separate intercept (and a main
    effect plus its interaction) gave 28 divergences, 271 with r_hat 1.10 without state
    predictors; `pm.ZeroSumNormal` (`n_zerosum_axes=2` for interactions) gave 0. Non-centred
    effects with 4-6 levels and 55k respondents: r_hat up to 1.03 on sd and z, no divergences -
    judge by r_hat of the poststratified estimates (`az.rhat(xr.Dataset({"x": (("chain",
    "draw", ...), arr)}))` works on NumPy draws). PSIS-LOO on binomial cells scores cells, not
    states, and barely penalised dropping state predictors: validate MRP against a held-out
    benchmark. Diverging map colours: `TwoSlopeNorm(vcenter=0.5)`, not `Normalize(lo, hi)`.
  - **Extreme values (E37).** `pymc_extras` `GenExtreme` uses Coles' sign (scipy `c = -xi`);
    its `logcdf` returns -inf ABOVE the upper bound when xi < 0 (should be 0) - `logp` is fine.
    The GEV support wall causes hundreds of "Logp function returned error code" divergences:
    sample `sigma = L(mu, xi) + exp(u)` with the prior + Jacobian as a `pm.Potential` (339 -> 0-1).
    nutpie's `divergence_draw` is an index, not the failing position: read `divergence_message`.
    `az.loo(pointwise=True)` has `.elpd_i` / `.pareto_k`, so exact refits for flagged points can
    be spliced in. `fig.add_axes` under the arviz style warns: `plt.figure(layout="none")`.
  - **Renewal equation / nowcasting (E38).** Given R_t the renewal equation is linear in the
    infections: `pt.linalg.solve_triangular` replaces `scan` (checked against a loop). A
    non-centred weekly random walk on log R with strong count data hit tree depth 10; centred
    `pm.GaussianRandomWalk` + `pm.sample(nuts={"adaptation": "low_rank"}, target_accept=0.9)`
    gave depth 4 (~40 s). macOS has no `timeout`: `perl -e 'alarm N; exec @ARGV' cmd`.
  - **Radiocarbon calibration (E39).** NUTS on a continuous calendar date through the
    interpolated curve gets trapped in its wiggles (r_hat 2.85, ESS 5, zero divergences): sum
    the date out on a 1-year grid with `pt.logsumexp`. In a sequence model integrate each date
    over its phase via its precomputed cumulative calibrated likelihood interpolated at the
    boundaries (23 smooth parameters). Dirichlet-spaced boundaries: tree depth 10 and r_hat 1.25
    with 0 divergences; `nuts={"adaptation": "low_rank"}` fixed it. A likelihood floored with
    `log(max(mass, 1e-300))` leaves flat plateaus that strand chains; an outlier mixture removes
    them. `az.extract` fails if the model has a coordinate named `sample`.
  - **Poll aggregation (E40).** Two latent terms identified only as a sum (true opinion + shared
    polling bias, fixed prior scales): rotate the non-centred z's by the angle of the two scales
    so the data see one and the other is exactly its prior (r_hat 1.05 -> 1.006). Sparse early
    polls: ~60 divergences at `target_accept` 0.8, 0 at 0.95 at no time cost. A predictive check
    in poll space cannot see an error shared by all polls - validate against results. Plotly
    `USA-states` choropleths render blank in headless Chrome from the CDN; inline
    `plotly.min.js` to check them. The Economist `all_polls.csv` is the 2016 cycle; some of its
    CSVs use CR-only line endings (`lineterminator="\r"`).
  - **Distributional regression (E41).** `az.summary(round_to=None)` still rounds to 2
    significant figures (r_hat 1.010 prints as 1.0): use `round_to=4`. `dist_math.normal_lcdf`
    evaluates both switch branches; the `erfcx` branch overflows above ~37 and leaks NaN
    gradients (divergences) - for positive arguments use `log1p(-0.5*erfc(b/sqrt(2)))`.
    P-splines with thousands of observations: centred RW2 + `nuts={"adaptation": "low_rank"}`
    gave 0 divergences, non-centred ~100. A t CDF with free df (`betainc` gradients) cost
    190-450 s per fit; the untruncated Box-Cox t lost < 0.2% mass and fits in 19 s. Timing
    `nutpie.sample` right after compiling includes numba JIT: time a compiled `dlogp` instead.
  - **xarray on posteriors (E42-E44; xarray 2026.7).** `DataArray.rank` needs `bottleneck` (not
    installed): `xr.apply_ufunc(scipy.stats.rankdata, da, input_core_dims=[[d]],
    output_core_dims=[[d]], kwargs={"axis": -1})`. Two `az.extract` results carry different
    `(chain, draw)` MultiIndexes on `sample`, so arithmetic/concat between them outer-joins into
    NaN-padded samples: `.drop_vars(["sample", "chain", "draw"])` first (also before `xr.concat`
    over draws). An index-less new dim aligns by position with a `sample` MultiIndex of the same
    length (handy for fresh noise). Grouping keys must be COORDS: `groupby(name=Grouper)` on a
    data variable raises `KeyError` - `assign_coords` first. `groupby` sorts string labels
    alphabetically (Fri, Mon...; DJF, JJA, MAM, SON): reorder with `.sel`. `UniqueGrouper(labels=)`
    fails on object coords containing NaN. A `BinGrouper(labels=...)` dim is named `<coord>_bins`.
    xarray 2-D `.plot`/FacetGrid refuse string coords: plot integer positions, set tick labels.
    FacetGrid warns "layout has changed to tight" under the arviz style (harmless); calling
    `invert_yaxis()` per shared-y panel inverts twice. `coarsen` averages extra coords
    (string coords raise TypeError; pass `coord_func` or drop them) and labels blocks by their
    centre, `resample(date="W")` by the closing Sunday. ISO week numbers duplicate across a
    year boundary and `unstack` refuses them: use `(dayofyear - 1 + jan1.dayofweek) // 7` for a
    calendar grid. `xr.Coordinates.from_pandas_multiindex(idx, "obs")` + `unstack` puts
    `sample_posterior_predictive` rows back on a grid. `.polyfit` on a datetime dim gives
    per-nanosecond slopes: convert to years. pandas 3 string `pd.Index` as a concat dim later
    clashes with NumPy string labels (`Cannot interpret '<StringDtype...>'`): `dtype=object`.
    scipy.stats functions on DataArrays return bare ndarrays (wrap in `apply_ufunc`) and are slow
    inside `apply_ufunc(vectorize=True)` root-finding (`skewnorm.logcdf` 150 s vs 4 s via
    `special.ndtr - 2*special.owens_t`). `az.rhat(idata.posterior)` is a DataTree:
    `.to_dataset().to_dataarray().max()`. `model.to_graphviz()` raises ImportError (no graphviz).
    Quantiles over masked or rolling-edge draws warn "All-NaN slice": filter it.
  - **Plotting from labelled arrays (E44).** `with plt.rc_context({...}):` around a plot made every
    LATER figure vanish from the executed notebook: set/restore `plt.rcParams` by hand. `.values`
    is where names end - `.transpose(<names>)` right before handing an array to plotly/matplotlib
    (a `go.Heatmap(z=...)` rendered transposed with no error). Drop non-dimension coords along a
    dim (e.g. `units` on `measurement`) before `rename(measurement="m2")` or
    `to_dataset(dim=...)`. Vectorised `.sel(species=obs.species)` carries the indexer's other
    coords (e.g. `sex`) and can clash with the target's dims: drop them. `stack(group=(...))`
    MultiIndexes are awkward to relabel: drop the index and assign plain labels plus non-index
    coords. `np.linalg.solve` under `apply_ufunc` with a batched vector RHS (NumPy 2):
    `lambda A, b: np.linalg.solve(A, b[..., None])[..., 0]`.
  - **Models from E42/E43.** `pm.SkewNormal` with varying alpha in the direct parameterisation:
    r_hat 2.4, 0 divergences; parameterise by mean/sd (`delta = a/sqrt(1+a^2)`, `omega =
    sd/sqrt(1-2 delta^2/pi)`, `xi = mean - omega delta sqrt(2/pi)`): r_hat 1.006. 731 per-day
    effects next to intercept + trend: ESS ~170 whatever the zero-sum choice (low-rank: 225
    divergences); centring the day level on intercept + trend fixed it. Daily models with
    independent days overstate P(any extreme day) and return levels because hot days cluster.
  - **Checking animations.** Headless Chrome `--screenshot` can hang on a `to_jshtml()` page;
    decode the base64 frames from the output HTML and look at a few instead; strip every
    non-base64 character (escaped backslash-newlines) before decoding.
- Available and worth using where they fit: `pm.ZeroSumNormal`, `pm.Censored`,
  `pm.Truncated`, `pm.CustomDist`, `pm.Mixture`, `pm.NormalMixture`, `pm.OrderedLogistic`,
  `pm.LKJCholeskyCov`, `pm.GaussianRandomWalk`, `pm.AR`, `pm.gp.HSGP`, `pm.gp.HSGPPeriodic`,
  `pm.ICAR`, `pm.CAR`, `pm.StickBreakingWeights`, `pm.MvStudentT`, `pm.do`, `pm.observe`, `pm.Potential`, `pm.fit` (ADVI), `pm.math.invprobit`,
  `pymc_extras.marginalize`, `pymc_extras.fit_laplace`.
