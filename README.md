# PyMC Challenges

Worked examples and **hard, real-data challenges** for Bayesian modelling with the current
PyMC stack (**PyMC 6 · ArviZ 1 · PyTensor 3**). Every challenge dataset is real; a few examples
(E47, E48, E51, E52, and parts of E46, E50 and E58-E61) simulate systems with no public data, and say so. Every challenge ends
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
  examples/      E01-E61  modelling techniques; D01-D03 data preparation and analysis. Worked, narrated, executed.
  challenges/    C01-C11  your workspace. Tasks, empty cells, hints on request.
  solutions/     C01-C11  full reference solutions, executed, with commentary.
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

| | Topic | Data | Key techniques |
|---|---|---|---|
| **E07** | Geo-temporal: induced earthquakes in Oklahoma | USGS catalogue, every M3+ quake 2005-19 | log-Gaussian Cox process, 2-D HSGP, space-time interaction so a hot-spot can move, taming priors behind an exp link, maps, hold-out forecast |
| **E08** | Physics-based models: weighing the universe | Union2.1, 580 type Ia supernovae | the theory *is* the regression function, a differentiable integral inside the model, a perfect H0-M degeneracy, P(expansion is accelerating) |
| **E09** | Your own JAX code, with gradients, inside a model | Hudson's Bay lynx-hare pelts | hand-written Op + `jax.vjp`, `pytensor.wrap_jax`, whole-model JAX (nutpie/NumPyro), a JAX ODE solver, multimodal likelihoods and start points |
| **E10** | Signal in the noise: GW150914 | Real LIGO Hanford + Livingston strain (GWOSC) | PSD estimation and whitening, exact time-domain likelihood, black-hole ringdown -> remnant mass and spin, Bayes factor signal vs noise, an honest chirp-mass failure |
| **E46** | Taming chaos: inference, prediction and control of chaotic dynamics | Simulated chaotic Ricker population (Wood 2010's benchmark); Nicholson's blowflies, 180 bidaily counts (`gamair`) | bifurcation diagram and Lyapunov exponent, why trajectory matching fails (a comb of ~1000 likelihood spikes, gradients growing like e^(lambda t), frozen NUTS chains), multiple shooting, a centred latent-state model that re-synchronises with the data (and why non-centred puts the chaos back), forecast skill vs climate, OGY control with tiny nudges, a delay-difference model with the delay marginalised by `logsumexp`, checking dynamics by features not paths, posterior Lyapunov exponents on a regime map (a noisy limit cycle, not chaos), feedback control chosen by its risk across the posterior vs a plug-in design |
| **E47** | Weather jiu-jitsu on the butterfly: Bayesian assimilation and regime control of Lorenz-63 | Simulated Lorenz-63 with process noise and multiplicative observation error (the setting of Liu, Huang & Lall 2026, *Chaos, Solitons & Fractals*) | two-regime attractor and residence times, instantaneous vs finite-time Lyapunov exponents (and what the LLE does not predict), multiple-shooting centred latent states with RK4 in PyTensor recovering sigma/rho/beta, posterior ensemble forecasts of regime switches, a linear-response minimal-energy nudge from the tangent propagator, a controlled experiment over trigger rules (LLE, geometric, every step) and nudge budgets, plug-in vs ensemble state and posterior vs prior parameters |
| **E48** | Regimes and extremes in a seasonal chaotic atmosphere: a Bayesian NHMM for Lorenz-84 and a test of NHMM-triggered control | Simulated seasonally forced Lorenz-84 (Lorenz 1990), daily noisy observations (the setting of Liu, Huang & Lall 2026, *Phys. Rev. E*) | nonautonomous chaos, choosing regime variables (eddy amplitude, not phase), nonhomogeneous HMM with seasonal softmax transitions and AR(1) emissions via `scan`, HMM multimodality and multi-start MAP initialisation, ordered transform vs a -inf Potential, K by held-out one-step log score, danger score and its link to local instability, control through the statistical model tested against a same-size random placebo, a physical threshold rule and an LLE trigger |
| **E49** | Jumps: Bayesian jump-diffusion SDEs for the S&P 500 | S&P 500 daily returns 2008-19 | compound-Poisson jump-diffusion simulated exactly and by Euler, the Poisson-mixture transition density with the jump count summed out, Merton model (jumps absorbing volatility clustering), non-centred stochastic volatility with and without jumps (SVJ), posterior jump probabilities per day (a jump is relative to current volatility), one-step-ahead scores from a bootstrap particle filter instead of LOO, crash-day probabilities and a Merton-formula implied-volatility smile with posterior bands |
| **E50** | Reaction-diffusion in biology: cell invasion in a scratch assay, and reading a signaling gradient | PC-3 scratch assay, 38 columns x 5 times (Jin et al. 2016, counts via Simpson, Murphy & Maclaren 2024); simulated Nodal/Lefty-scale morphogen images and FRAP | Fisher-KPP by the method of lines (finite volumes, no-flux, explicit Euler in `scan` with a stability check), Poisson counts with an estimated carrying capacity, a start-up delay tanh(beta t) and the fourfold bias in D without it, wave speed 2 sqrt(D lambda) and parameter trade-offs, exact SDD gradients via the eigenbasis of the discrete Laplacian, steady state identifies only sqrt(D/k) and FRAP separates D and k, positional error and distinguishable positions from receptor occupancy with ligand and receptor noise |
| **E51** | Designing a thermal gradient plate: heaters, insulation and sensors for the 2-D heat equation | Simulated 40 x 20 cm plate, calibration experiment (3 test heaters, 12 thermocouples) and operating disturbances | finite-volume 2-D steady heat equation as a linear system, influence maps and superposition, Bayesian calibration with `pt.linalg.solve` inside PyMC (a weakly identified edge loss whose interval misses the truth), greedy heater placement with constrained least squares over posterior draws (heaters kept out of the working area), plug-in vs posterior design, back-face insulation as passive robustness, Bayesian disturbance estimation and power re-optimisation for feedback, greedy sensor placement vs even spacing, sensors on heaters and an oracle, power headroom |
| **E11** | Scientific ML: a neural network inside a differential equation | Hudson's Bay lynx-hare pelts | universal / neural ODEs in JAX, priors over functions, why weight-space r_hat is meaningless, did the network rediscover mass-action?, physics vs flexibility on a held-out forecast |
| **E12** | The inference frontier: when plain NUTS is not enough | S&P 500 latent volatility (from C07) | a reference posterior, low-rank and normalizing-flow adapted NUTS, Laplace / ADVI / Pathfinder, Pareto k-hat as a trust diagnostic when you have no reference |
| **E13** | State-space models: let the Kalman filter integrate out the states | UK road casualties 1969-84 and the 1983 seat-belt law | the Kalman recursion in 10 lines, `pymc_extras.statespace`, 3 sampled parameters instead of 400, decomposition, one-step-ahead checks, a counterfactual with a negative control, missing data and forecasting for free - and an honest speed comparison |
| **E14** | Simulation-based inference: Bayes without a likelihood | S&P 500 returns; 114 years of Canadian lynx | rejection ABC calibrated against a known posterior, tolerance and summary statistics as the two approximations, `pm.Simulator` + SMC, the g-and-k distribution, Wood's synthetic likelihood, a population model that *cannot* produce the cycle |
| **E15** | Bayesian additive regression trees | Capital Bikeshare hourly rentals (subsampled); LaLonde job training | the BART prior over functions, `pmb.BART` inside an ordinary PyMC model, PGBART + NUTS and diagnosing in function space, partial dependence / ICE / variable importance, out-of-sample prediction, an honest comparison with a GLM, BART for causal g-computation |
| **E16** | Showing uncertainty: one posterior, twenty displays | Motorcycle crash test; Minnesota radon with county boundaries; Premier League 2024/25 | epistemic vs predictive bands, fan charts, why a band is not a set of curves, spaghetti, animated hypothetical outcome plots, quantile dotplots, exceedance curves, caterpillars and ridgelines, maps of the mean / the uncertainty / hatching / bivariate and value-suppressing palettes / exceedance / small-multiple and animated draws, icon arrays, rank-probability heatmaps, pairwise matrices, re-simulated seasons, uncertainty in a table, interactive plotly figures with hover |
| **E17** | Meta-analysis: a database of studies that never shares a patient | `metadat` benchmarks: 13 BCG trials, 16 magnesium trials (ISIS-4), 56 studies in 11 districts | sufficient statistics and why aggregate tables are lossless, two-stage vs one-stage models, forest plots with prediction intervals, tau / I² posteriors, meta-regression, leave-one-out and cumulative meta-analysis, contour-enhanced funnels and a Bayesian Egger model, small-cell suppression as censoring, differential-privacy noise as measurement error (privacy-utility curve), three-level models |

#### Correlation structures and graphs (E18-E23)

The examples above correlate things through space, time or group membership. These six put the
correlation on a **graph** or a **tree**, learn the graph itself, or model many outcomes jointly:

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E18** | Areal models on a graph: ICAR, proper CAR, BYM2 | Scottish lip cancer 1975-80, 56 district polygons (GeoDa) + WinBUGS neighbour list | contiguity graph from polygons in NumPy, islands and disconnected components, graph Laplacian and the BYM2 scaling factor, `pm.ICAR` / `pm.CAR` / BYM2, Moran's I residual check, exceedance-probability maps, PSIS-LOO vs refit K-fold, spatial confounding |
| **E19** | Learning a graph: Bayesian Gaussian graphical models | Sachs et al. (2005) T-cell flow cytometry, 11 phosphoproteins + 20-edge consensus network | marginal vs partial correlation, `LKJCholeskyCov` precision and partial-correlation deterministics, why a horseshoe on the precision matrix leaves the positive-definite cone, Bayesian neighbourhood selection (node-wise horseshoe, AND/OR), NumPy graphical lasso + EBIC, precision-recall against the consensus, subsample replication |
| **E20** | Learning a directed graph: Bayesian causal discovery | Sachs et al. (2005), 7466 cells in 9 intervention conditions | linear-Gaussian SEMs, Markov equivalence seen through LOO and `pm.do`, NOTEARS acyclicity as a `pm.Potential` (and why it is multimodal and scale-dependent), fixed-order horseshoe DAGs, exact BGe + dynamic-programming averaging over all orders, interventions as a per-cell mask - and a wrongly specified intervention |
| **E21** | Network data as the outcome: latent-space and stochastic block models | Lazega law-firm coworker network (71 lawyers, 2485 dyads) | dyad logistic vs sociality random effects, Hoff latent space + Procrustes, mixed-membership SBM with marginalised roles (degree-corrected or not), co-clustering matrix, network PPCs (degree, transitivity, shared partners, geodesics), dyad-level LOO and held-out AUC |
| **E22** | Many correlated outcomes: full covariance, factor models, copulas | Ken French 12 and 30 US industry portfolios, daily 2015-2024 | EWMA volatility standardisation, `MvNormal` + `LKJCholeskyCov`, low-rank + diagonal factor models, sign-flip / rotation non-identifiability and what r_hat does (not) see, anchored loadings vs Procrustes alignment, choosing K by held-out density, a Gaussian copula with t margins by hand, `MvStudentT` and tail dependence, VaR and joint-crash backtest |
| **E23** | Correlation from a tree: phylogenetic regression | Primates301 brain / body / group size + 10kTrees phylogeny | reading an R phylo object without R, covariance from shared branch length, Brownian motion / Pagel's lambda / OU-as-GP regression, the OU ridge and a reparameterisation, conditional LOO from the precision matrix (and two wrong shortcuts), whitening, the tree as a sparse GMRF, imputing missing traits on the tree |

#### More model families (E24-E29)

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E24** | Hidden Markov models | USGS M7+ world earthquake counts 1900-2024; elk GPS tracks (Morales et al. 2004) | forward algorithm in `pytensor.scan` vs `pymc_extras` `DiscreteMarkovChain` + `marginalize`, label switching and ordering, forward-backward and Viterbi per draw, dwell times, pseudo-residuals, integrated LOO and leave-future-out, padded multi-track HMM, covariate-dependent transitions |
| **E25** | Joint longitudinal-survival models | Mayo Clinic PBC trial: serial bilirubin + survival | mixed model + Weibull hazard sharing LKJ random effects, cumulative hazard by Gauss-Legendre quadrature, baseline / LOCF / two-stage biases, informative dropout, centred vs non-centred vs hierarchically centred, dynamic survival predictions for held-out patients, landmark calibration |
| **E26** | Bayesian nonparametrics: Dirichlet process mixtures | Enzyme activity of 245 people (Richardson & Green 1997); PBC bilirubin slopes | stick-breaking and the implied prior on the number of clusters, `pm.StickBreakingWeights` + `pm.NormalMixture`, truncation check, label-free diagnostics, co-clustering and Binder / VI point clustering, sensitivity to alpha, sparse finite mixtures (and why a tiny Dirichlet parameter defeats NUTS), a DP random-effects distribution |
| **E27** | Sparse regression: horseshoe, regularised horseshoe, spike-and-slab | Prostate cancer (ESL split); LARS diabetes, 64 predictors on 120 patients | shrinkage factor kappa, ridge vs Bayesian lasso vs horseshoe, tau0 from a prior guess and the simulated m_eff prior, horseshoe divergences and their fixes, spike-and-slab via `pymc_extras.marginalize` vs exact enumeration, correlated predictors, projection predictive selection and CV lasso in NumPy, held-out elpd |
| **E28** | Imperfect detection: occupancy, N-mixture, capture-recapture | Swiss breeding-bird survey (crossbill, mallard); snowshoe hare capture histories | latent presence / abundance summed out in `pm.CustomDist` (checked against `pymc_extras.marginalize`), finite-sample occupancy, naive vs corrected trends, visits x prior confounding, N-mixture truncation check, Poisson vs NegBin, data augmentation M0 / Mt / Mh, a non-identifiable heterogeneity model |
| **E29** | Bayesian neural networks: what the posterior over weights buys you | Motorcycle crash test; concrete compressive strength (Yeh 1998, UCI) | an MLP in PyMC with `dims`, priors over functions and the 1/sqrt(width) GP limit, NUTS vs ADVI / Pathfinder / MAP ensemble / last-layer Laplace on held-out density, function-space r_hat, per-input prior scales (ARD), BNN vs HSGP vs GLM, PIT and coverage, chains stuck in different functional modes |

#### Random partitions and random features (E30-E31)

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E30** | Exchangeable partitions: Pólya urns, the Chinese restaurant process, Pitman-Yor and unseen species | *Moby-Dick* word tokens (Project Gutenberg); Barro Colorado Island 50-ha tree census | urn simulation and de Finetti limits, the EPPF checked by brute force, NUTS on (alpha, d) with no latent assignments, a stable log rising factorial, frequency-of-frequencies and Heaps-curve PPCs, exact held-out predictive from EPPF ratios, new-type predictions vs Good-Toulmin with a coverage study, a negative discount (finite species pool) with the pool size marginalised |
| **E31** | The Indian buffet process: latent binary features | UCI optdigits handwritten digits (300 training / 200 test digits by other writers) | the IBP as a restaurant, stick-breaking and finite beta-Bernoulli, collapsed Gibbs in NumPy checked by brute-force enumeration, noise level vs alpha as the driver of K+, stuck chains and split-merge, label-free summaries, a truncated PyMC IBP with 2^K enumeration (and a collapse-to-empty trap), held-out half-digit completion vs probabilistic PCA |

#### Media measurement (E32-E35, E45)

One real open dataset (Conjura's multi-brand e-commerce MMM data, CC BY 4.0) and four
questions a marketing team actually asks. Each ends with displays for people who decide,
not people who model; the results are also exported for a single plain-language web page.

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E32** | Building a custom likelihood: media spend that chases demand | Conjura MMM data: a UK skincare brand, 884 days of new customers, discounts and Google + Meta spend | the data-generating story before the likelihood, identification strategies compared, a Gaussian copula with a negative-binomial margin derived for a discrete outcome (Park-Gupta endogeneity correction), stable log-space implementation (incomplete-beta tails, `ndtri_exp`, `log1mexp`) as `pm.CustomDist` with `random`, five unit tests for a likelihood, fake-data recovery where the textbook fix fails under persistent shocks and a generalised control fixes it, the generalisation failing on real data (chains in two modes), a time-shift placebo, LOO and PPC, marginal CAC for a +20% budget |
| **E33** | Hierarchical multi-market MMM and a budget decision under uncertainty | Conjura MMM data: Scandinavian apparel brand, 3 markets x Google/Meta, 81 weeks | Geometric adstock + Hill saturation, priors on cost per incremental customer (Meridian ROI-style), partial pooling across markets (centred vs non-centred), holiday/trend/seasonal baseline, prior predictive in euros, LOO vs no/full pooling, SLSQP budget optimisation evaluated over posterior draws (expected and cautious objectives), prior sensitivity, endogeneity caveat, quantile dotplots, spaghetti curves, probability heatmap, shrinkage display |
| **E34** | Bayesian synthetic control of a media natural experiment | Conjura MMM data: a brand's US Meta switch-off (Oct 2023) + 21 donor brands | change-point scan for interventions, simplex (Dirichlet) SC, augmented/structural SC (horseshoe + random-walk level integrated out as a GP, CausalImpact-style), own control series, placebos in time/space/outcome, donor-pool sensitivity, cost per lost customer, placebo line-up, an honestly inconclusive answer |
| **E35** | Halo effects as Bayesian causal mediation: does Meta work through branded search? | Conjura MMM data: one UK clothing brand, weekly new customers, branded-search clicks, Meta/Google spend, UK + US | natural direct/indirect effects, joint mediator + outcome negative binomial model with adstock and saturation, counterfactual propagation from posterior draws (plug-in and Monte Carlo), product-of-coefficients check, "search as control" vs "search left out", replication in a second market, sensitivity analysis for mediator-outcome confounding, Sankey and stacked quantile-bar displays |
| **E45** | One family for every response curve: a Weibull transform in JAX for media models | Conjura MMM data: a UK apparel brand, 180 weeks of new customers and Google + Meta spend | the saturation family as a prior, a normalised Weibull response $(1-e^{-cx^k})/(1-e^{-c})$ spanning power, concave and S-curves with a shape map and closed-form inflection, a float32-safe custom JAX function (`expm1` + series switch, double-`where`) with geometric adstock in `lax.scan` and unit tests, `vmap` + optax batch fits of 812 Hill curves and other standard transforms (and where the tails differ), translating a Hill prior into (k, c), `pytensor.wrap_jax` + `verify_grad`, fake-data check against Hill and logistic models, LOO on a real brand, marginal CAC via `vmap(grad)`, Numba+JAX-node vs whole-model JAX timing |

#### State of the art, for everyone (E36-E38)

Three model families that practitioners use today, each with a plain-language opening,
"In plain words" takeaways, and a closing section of uncertainty displays for readers with no
statistics. Their key results are also exported for a single plain-language web page.

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E36** | Multilevel regression and poststratification (MRP): what every state thinks, from one national survey | 2018 CCES (5,000-person sample + 55,000 held out; abortion-coverage question) + ACS poststratification table + 2016 vote share (Lopez-Martin, Phillips & Gelman case study) | why raw state means and raking fail in small states, binomial-cell multilevel logit with `ZeroSumNormal` effects and interactions, random-walk prior on ordered groups, state-level predictors, per-draw poststratification (states, age within state), validation against a held-out benchmark with coverage, a failure (no state predictors) that LOO barely sees, value-suppressing tile map, animated map of plausible outcomes, "6 people vs the model" dotplots, icon arrays |
| **E37** | Extreme-event attribution: how much did warming load the dice for the 2021 Pacific Northwest heatwave? | NOAA GHCN-Daily annual maximum temperature at Sea-Tac and Portland; NASA GISTEMP global temperature | block maxima and the GEV (`pymc_extras` `GenExtreme` checked against scipy), a geophysical shape prior, a reparameterisation that removes support-wall divergences, non-stationary GEV on smoothed global temperature, return-level plot / PIT / LOO with exact refits, fitting without, with and selection-corrected for the event, probability ratio with an honest "impossible before" share, intensity change, shared shape across two stations, icon arrays of summers, 30-year-mortgage odds, animated shifting distribution |
| **E38** | Real-time epidemic tracking: is it growing right now? Nowcasting + a renewal-equation R | German COVID-19 hospitalisation reporting triangle (RKI via the Hospitalization Nowcast Hub), winter 2021/22 | the reporting triangle and real data-as-of-then snapshots, a delay-hazard nowcast (reporting weekday + weekly drift, negative binomial cells), the renewal equation as a triangular solve, a spurious "wave is ending" from truncated data, generation-interval sensitivity (R vs growth rate), a 5-date backtest with coverage and CRPS - and honestly under-covering intervals, traffic-light timeline with "said then vs hindsight" dials, icon array of reports still to come, fog-lifting animation, quantile dotplot of next week |

#### State of the art, for everyone II (E39-E41)

Three more model families in the same format as E36-E38: a plain-language opening, "In plain
words" takeaways, a closing "Explaining it to everyone" section, and a JSON export of the key
results for a web page.

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E39** | How old is it? Radiocarbon calibration and Bayesian chronologies | IntCal20 calibration curve; all 49 measurements from the 1988 Shroud of Turin dating (3 labs, shroud + 3 known-age controls, hand-transcribed from Damon et al. 1989); 31 dates from the Sluggan Bog peat core (Bchron) | calibration through a wiggly, uncertain curve, why NUTS on a calendar date fails (r_hat 2.85) and summing the date out on a 1-year grid instead, lab offsets and lab-by-sample heterogeneity anchored by controls, the 1989 chi-square test as a posterior predictive check, an OxCal-style sequence model with Dirichlet-spaced boundaries and a 5% outlier model, low-rank mass matrix, "shadow through the curve" plot, 2,000-year timeline, 20 equally likely dates, contamination icon array, animation of plausible histories |
| **E40** | Who will win? Dynamic Bayesian poll aggregation and election forecasting | 2016 US presidential polls (HuffPost Pollster via the Economist's us-potus-model), 1976-2016 state results, fundamentals, ACS state demographics | Linzer reverse random walks from a fundamentals prior, correlated state innovations, house/mode/population effects, poll-level + shared polling error with a rotation for the unidentified truth/bias split, electoral-college simulation and tipping points, backtest at 6 dates, coverage and a joint Mahalanobis check, sampling-error-only vs shared-error models (100% vs 88% for Clinton), 100-elections dotplot, animated electoral map, honest needle, plotly hover map |
| **E41** | Is my child normal? Distributional regression (GAMLSS/LMS) and growth centile charts | Fourth Dutch Growth Study: BMI of 7,294 boys aged 0-21 (`gamlss.data` dbbmi) | constant-spread vs smooth-spread vs skewed vs heavy-tailed models, Bayesian P-splines (centred RW2, low-rank mass matrix) for every distribution parameter, hand-written BCCG (Cole-Green LMS) and Box-Cox t likelihoods checked against scipy/integration/simulation, age-bin ML LMS fit for comparison, centile coverage by age band, worm plots, LOO, uncertainty of the centile lines themselves, blurred growth chart, "of 100 boys" icon array, 20-boys dotplot, animated distribution, plotly hover |

#### Posteriors as labelled arrays: xarray for Bayesian work (E42-E44)

A posterior is a labelled array with named `chain`, `draw` and model dims. These three notebooks
do all their data and posterior work by name (coords and dims) instead of axis numbers, and use
that to build derived quantities, bins, time aggregations and some unusual plots.

| | Topic | Real data | Key techniques |
|---|---|---|---|
| **E42** | xarray foundations for posteriors: dims, coords, broadcasting and time | Capital Bikeshare hourly rentals 2011-12 (17k hours) | table -> `date x hour` Dataset with non-dimension coords, stack/unstack round trip to model rows, DataTree navigation and thinning, `az.extract`, broadcasting counterfactual grids by name, `apply_ufunc` RNG, "sum the draws, then summarise" with `resample`/`rolling`/`coarsen`/`cumulative`, a posterior over the date of the millionth rental, `BinGrouper` and multi-key groupby calibration, vectorised `.sel` for new-day prediction, `xr.concat` over a model dim; polar clock, calendar PIT heatmap, fan charts |
| **E43** | Transforming posteriors with coords: dates, exceedances, ranks and anomalies | NOAA GHCN-Daily TMAX, Sea-Tac + Portland 1948-2025 | long -> wide xarray, `align` join pitfalls, `concat` over a named `station` dim, a harmonic basis as a DataArray shared by model and posterior, date posteriors with `idxmax` + `interp` (summer onset and length), exceedance probabilities and return periods on draws, P(warmest year) from ranks inside each draw, groupby anomalies, `BinGrouper` x season, `apply_ufunc(vectorize=True)` root-finding; warming stripes with uncertainty, ridgelines, climate loops |
| **E44** | Plotting posteriors creatively from labelled arrays | Palmer penguins (raw field records) | multivariate normal with one LKJ covariance per species, `apply_ufunc` for KDEs / Cholesky / `rankdata` / `solve`, `xr.dot` ellipses and correlated draws, pairwise contrast matrices by renaming a dim, `stack`/`sortby`, FacetGrids straight from the posterior, Simpson's paradox in correlations; ridgeline, raincloud, bump chart, parallel coordinates of draws, hypothetical-outcome animation, quantile dotplots, seaborn hand-off via `to_dataframe`, plotly hover heatmap, P(male) for unsexed birds |

#### Cell biology and biophysics (E52-E61)

Mechanistic models of living cells. See also E50 (reaction-diffusion in tissues).

| | Topic | Data | Key techniques |
|---|---|---|---|
| **E52** | Signals in two dimensions and one: bulk-surface reaction-diffusion in a cell, the cytosolic shortcut, and polarity | Simulated 20 um round cell; photoactivation of a membrane-cytosol shuttling protein imaged at 6 times (64 membrane arcs, 961 cytosolic volumes) | 2-D cytosolic PDE coupled to a 1-D membrane PDE by a Robin (binding/unbinding) boundary condition, finite volumes on a disk, exact angular Fourier decomposition (33 problems of size 17), detailed-balance symmetrisation and batched `pt.linalg.eigh` for exact differentiable time courses, bulk-mediated surface diffusion, signal reach under cytosolic phosphatases and cell size, the membrane profile identifies D_c, 1-D diffusion and well-mixed-cytosol models overestimate D_m (3x) and LOO ranks them, posterior signaling predictions, wave-pinning polarity phase diagram (IMEX, sparse LU) |
| **E53** | The Luria-Delbrück fluctuation test: jackpots and Bayesian mutation rates | Luria & Delbrück 1943 fluctuation tables (E. coli / phage T1, 281 cultures; transcribed, cross-checked with the flan package) + simulated cultures | compound-Poisson likelihood as a triangular solve, partial plating, censored/binned counts, estimator simulation study (mean vs p0 vs median vs MLE), PSIS-LOO induced vs spontaneous mutation, hierarchical rates, tail posterior predictive checks, Mandelbrot-Koch fitness |
| **E54** | Why a neuron fires: Bayesian inference of a conductance-based model from patch-clamp recordings | Allen Cell Types Database, one mouse Sst interneuron (21 current steps) | RC fit with two exponentials + stationary AR(1), Pospischil HH model, Rush-Larsen in Numba and scan, trace-matching pathology, pm.Simulator + SMC feature likelihood, bimodal posterior, prior-whitened sloppiness spectrum, held-out f-I predictions (an honest misfit: the f-I curve is too steep) |
| **E55** | Transcriptional bursting: the telegraph model and what mRNA snapshots identify | Larsson et al. 2019 allele-resolved scRNA-seq UMIs (24 genes, 165 G1 fibroblasts) + simulated cells | Gillespie SSA, chemical master equation / finite state projection, tridiagonal recurrence, pt.linalg.expm time course, NB-limit map, capture as binomial thinning, Poisson/NB/telegraph LOO, prior sensitivity, experiment design, hierarchical LKJ, wrap_jax |
| **E56** | Sizer, adder or timer? Bacterial size control and regression dilution | Tanouchi et al. 2017 mother machine, E. coli at 25/27/37 °C, 19k cycles | noisy linear map, regression dilution, errors-in-variables as a closed-form 16-dim MvNormal over 8-generation windows, measurement noise identified from lineage autocorrelation, add-noise stress test, over-correction check, PSIS-LOO sizer/adder/timer, PPC, hierarchical growth rates |
| **E57** | Single-particle tracking: diffusion states, localisation error, anomalous diffusion | RARA-HaloTag and HaloTag-NLS tracks in live U2OS nuclei (saSPT example data, Heckert et al. 2022; MIT) | exact MA(1) track likelihood via a sine eigenbasis, MSD-fit bias, D-sigma trade-off, Spot-On-style state mixture with state-dependent track loss, LOO, hierarchy across cells, GPB1 switching HMM in scan, two-filter smoothing, fractional Brownian motion, fake subdiffusion from localisation error |
| **E58** | Seeing below the diffraction limit: Bayesian localisation, detection and counting in STORM | Real raw dSTORM EMCCD frames of Unc-13 at a Drosophila NMJ (Dannhäuser et al. 2022, Zenodo CC BY) + simulated nuclear-pore rings | photon-transfer camera calibration, erf-pixelated PSF, EMCCD excess noise, NUTS vs MLE vs least squares, Cramér-Rao/Mortensen bounds, batched Fisher scoring + Laplace evidence, SMC evidence check, emitter counting as model selection, recall/precision benchmark, uncertainty rendering, FRC, negative-binomial blink counting with marginalised copy number |
| **E59** | Counting steps you cannot see: kinesin stepping and dwell-time kinetics | MINFLUX kinesin-1 step tables + raw traces (Wolff, Scheiderer et al. 2023, Zenodo CC BY 4.0); simulated traces for step detection | hypoexponential dwells, randomness parameter, penalised step finder vs 2-phase lattice HMM (forward algorithm in scan), missed-step bias, ordered rates and label switching, LOO over hidden steps, dead-time/hidden-step equivalence, ATP Michaelis-Menten with molecule heterogeneity |
| **E60** | How hard does a cell pull? Traction force microscopy as a Bayesian inverse problem | Simulated cell with 16 known adhesions + real PIV displacement field of a 7-cell colony (pyTFM example data, Bauer et al. 2021) | FTTC Boussinesq Green's function, longitudinal/transverse Fourier diagonalisation, Tikhonov as a Gaussian prior, L-curve vs discrepancy vs marginal likelihood, hyperparameters in PyMC with the field integrated out, exact FFT posterior draws, GP spectra (SE/Matérn) + spectral predictive check, Young's-modulus uncertainty, constrained (footprint) TFM, regularised horseshoe via a sufficient statistic |
| **E61** | Decisions from dynamics: Turing patterns and kinetic proofreading | Simulated Schnakenberg patterns; Nodal/Lefty diffusion coefficients (Müller et al. 2012, transcribed); real 1G4 T-cell CD69 dose-response + SPR KDs (Pettmann et al. 2021, eLife CC BY) | linear stability, FFT semi-implicit PDE solver, calibrated feature likelihood by importance sampling, exact Fourier-mode likelihood (Woodbury), P(Turing conditions), discrete N marginalised with logsumexp, reparameterising to remove multimodality, expected information gain design |

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
| **C11** | Who needs the second drug? | ACTG 175 HIV trial (zidovudine vs zidovudine + didanosine) | 5 | build a Dirichlet-process mixture of regressions; take apart a colleague's convincing "responder" analysis; let covariates choose the subgroups and find out what LOO cannot tell you; turn conditional effects into a treatment rule |

Roughly in order of difficulty, but they are independent - pick what is closest to your work.
The descriptions are deliberately vague about *what* goes wrong: finding out is the challenge.

## Interactive posterior reports (Lumen + a local LLM)

[`reports/`](reports/README.md) turns the posteriors of E33, E36, E37, E40 and E41 into
reader-driven reports with [Lumen](https://lumen.holoviz.org/) and a small Gemma 4 model
running locally in Ollama. It has two apps: a report whose sections have live widgets
(budget what-ifs, joint probabilities, credible-interval rankings, fan charts) and short
LLM summaries, and a chat explorer over a DuckDB database of posterior draws.

```bash
uv sync --extra reports && ollama pull gemma4:e2b-it-qat
uv run --extra reports python reports/build_warehouse.py
uv run --extra reports panel serve reports/report_app.py --show
```

## Website (GitHub Pages)

`tools/build_site.py` renders every example, challenge (with its hint ladder) and solution into a
static website - narrative, maths, code, figures, tables, animations and plotly figures - plus a
landing page, a catalogue, a "choosing a model" guide and a challenges overview. Nothing is
re-executed; the pages come from the outputs stored in the notebooks.

```bash
uv run python tools/build_site.py      # -> _site/ (~75 MB, a few seconds)
python -m http.server -d _site         # preview at http://localhost:8000
```

`.github/workflows/pages.yml` builds and deploys it on every push to `main` that touches the
notebooks or the site sources (set *Settings -> Pages -> Source* to **GitHub Actions**). Site prose
lives in `tools/site_content.py`, styles and scripts in `tools/site_assets/`; per-notebook titles,
data and techniques are read from the curriculum tables above, so keep those up to date.

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
