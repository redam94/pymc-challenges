# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # C06 · Galaxies: how many clusters, and who belongs where?
#
# | | |
# |---|---|
# | **Difficulty** | ★★★☆☆ |
# | **Time** | 3-4 hours |
# | **Data** | Postman, Huchra & Geller (1986) via Roeder (1990): recession velocities of 82 galaxies in the Corona Borealis region |
# | **Skills** | Finite mixture models · diagnosing a sampler whose chains disagree · constrained parameters and initial values · marginalised vs explicit discrete assignments · membership and co-clustering probabilities · choosing K with LOO and PPCs (and knowing when you cannot) · predictive probabilities of a void |
#
# ## The brief
#
# An astronomy group has measured the recession velocities of 82 galaxies in six well-separated
# conic sections of sky towards Corona Borealis. By Hubble's law velocity is a proxy for
# distance, so if galaxies were spread evenly through space the velocities would form one
# smooth hump. Theories of large-scale structure say otherwise: galaxies should sit in
# **clusters separated by voids**, which would show up as several humps with gaps in between.
#
# The group asks three questions:
#
# 1. Do the velocities show distinct clusters separated by voids, and **how many**?
# 2. **Which galaxies belong together?** For the ones near a boundary, how sure are you?
# 3. Is the apparent **void** between the main body and the fastest galaxies real, or the kind
#    of gap that 82 draws from a smooth distribution produce by chance?
#
# They want answers with uncertainty, in language they can put in a paper.
#
# ## How this notebook works
#
# - Each task states **what to deliver**, not how. Write your code in the `YOUR CODE HERE` cells.
# - Stuck? `h.hint("task2")` reveals hints one level at a time: *nudge → approach → code skeleton*.
#   Try to get by on nudges.
# - `h.check("task2", mu_main=...)` compares your numbers with the reference solution.
# - A full worked solution lives in `notebooks/solutions/`. Open it only when you are done (or truly stuck).

# %%
import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

from pymc_challenges import Hints, data

RANDOM_SEED = 1986
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-variat")

h = Hints("C06")
h.tasks()

# %% [markdown]
# ## The data
#
# A single column, `dat`: velocity in km/s. We sort it and work in **thousands of km/s**, so
# galaxy 0 is the nearest and galaxy 81 the most distant, and sensible prior scales are of
# order 1-10 rather than 1,000-10,000.
#
# A historical footnote: this is the version distributed with R's `MASS` package, which
# contains a famous typo - the 78th value is 26690 where Roeder's paper has 26960. It makes no
# difference to anything below, but it is the reason published analyses of "the galaxy data"
# do not always agree to the last digit.

# %%
data.describe("galaxies")
galaxies = data.load("galaxies")
v = np.sort(galaxies["dat"].to_numpy() / 1000)
N = len(v)
print(N, "galaxies, from", v.min(), "to", v.max(), "thousand km/s")

# %% [markdown]
# ## Task 0 · Look at the data
#
# **Deliver** a plot that shows every single galaxy (82 points deserve better than a 10-bin
# histogram alone) and a list of the largest gaps between neighbouring velocities. How many
# groups do you see, how many galaxies in each, and how wide is each group?

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task0")

# %% tags=["solution"]
def plot_velocities(ax=None, bins=40):
    ax = ax or plt.gca()
    ax.hist(v, bins=bins, density=True, color="0.8")
    ax.plot(v, np.full(N, -0.008), "|", color="k", ms=12)
    ax.set(xlabel="velocity (1000 km/s)", ylabel="density")
    return ax


plot_velocities()

gaps = pd.DataFrame({"from": v[:-1], "to": v[1:], "gap": np.diff(v)}).sort_values("gap", ascending=False)
gaps.head(5).round(3)

# %% tags=["solution"]
groups = pd.cut(v, [0, 13, 29, 40], labels=["near", "main", "far"])
pd.DataFrame({"v": v, "group": groups}).groupby("group", observed=True).v.agg(["size", "mean", "std", "min", "max"]).round(2)

# %% [markdown] tags=["solution"]
# Three groups, separated by the two largest gaps (5.7 and 5.1 thousand km/s - the next
# largest is 2.2): a tight **near** group of 7 galaxies around 9,700 km/s, a **main** body of
# 72 between 16,000 and 27,000, and a **far** group of just 3 beyond 32,000. Two things to
# remember for later. The groups are very unequal in width: the near group has a standard
# deviation of 0.46, the main body 2.2. And the main body is not obviously one hump: the
# histogram dips around 21, and a pair of galaxies at 16.1 sits 2,200 km/s below the rest.

# %% [markdown]
# ## Task 1 · The textbook mixture
#
# Fit a mixture of **three Gaussians**: three means, one standard deviation shared by all
# components, and three mixture weights. Write it the way a textbook would, with **the same
# prior for every component** - nothing in the model should distinguish "component 0" from
# "component 2". Use the default four chains.
#
# **Deliver**
# 1. The summary table and whatever plots you need to decide whether you can trust it.
# 2. A diagnosis. Be precise: what exactly went wrong, is any *individual* chain wrong, and
#    is there a quantity in the summary table that is perfectly fine - and why that one?

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task1")
# h.check("task1", max_rhat=...)   # the largest r_hat among the component means

# %% [markdown] tags=["solution"]
# ### Solution
#
# Priors: means anywhere in the surveyed range (`Normal(20, 10)`), a spread of a few thousand
# km/s at most (`HalfNormal(3)`), and a flat Dirichlet on the weights. `pm.NormalMixture`
# sums over the unknown component of each galaxy inside the likelihood.

# %% tags=["solution"]
K = 3
with pm.Model(coords={"component": range(K)}) as naive_model:
    mu = pm.Normal("mu", 20, 10, dims="component")
    sigma = pm.HalfNormal("sigma", 3)
    w = pm.Dirichlet("w", np.ones(K), dims="component")
    pm.NormalMixture("velocity", w=w, mu=mu, sigma=sigma, observed=v)
    naive_idata = pm.sample(random_seed=RANDOM_SEED)

print("divergences:", int(naive_idata.sample_stats["diverging"].sum()))
az.summary(naive_idata, round_to=2)

# %% tags=["solution"]
az.plot_trace(naive_idata, var_names=["mu"]);

# %% tags=["solution"]
naive_idata.posterior["mu"].mean("draw").to_pandas().round(1)

# %% tags=["solution"]
max_rhat = float(az.rhat(naive_idata.posterior["mu"]).max())
assert h.check("task1", max_rhat=max_rhat)

# %% [markdown] tags=["solution"]
# The summary is a disaster - `r_hat` between 1.5 and 2.3 for the means and the weights,
# effective sample sizes in single digits - and yet there are **no divergences**, and the trace
# of every chain is a flat, well-mixed band. The per-chain table explains it. Every chain found
# the same three clusters, at 9.9, 21.4 and 32.6, but they disagree about what to *call* them:
# chains 0 and 1 use the order (near, far, main), chain 2 (main, near, far), chain 3 (far, near,
# main).
#
# Nothing in the prior or the likelihood distinguishes the components, so the posterior
# consists of 3! = 6 identical copies of the same mode, one for each way of naming the
# clusters. The copies are separated by regions of very low probability, so a chain stays in
# whichever copy it falls into first. **Each chain is a perfectly good sample of one copy.**
# `r_hat` compares the variation *between* chains with the variation *within* them, sees
# chains sitting at 9.9 and 32.6 for "the same" parameter, and rightly reports that they have
# not converged to a common distribution. The pooled estimate `mu[0] = 18.4 ± 9.5` is an
# average over names and means nothing.
#
# The exception is `sigma`: `r_hat` 1.00 and over 4,000 effective draws. It is shared by all
# components, so it does not care what they are called. The same holds for any quantity
# that is invariant to relabelling - the fitted density, for instance. This is called
# **label switching**, and it is a property of the model, not a failure of the sampler: no
# `target_accept` or longer run will cure it. (A sampler good enough to jump between the
# copies would make every marginal an identical three-humped mixture - also useless.)

# %% [markdown]
# ## Task 2 · Fix the model
#
# **Deliver**
# 1. A three-component mixture with the same likelihood and clean diagnostics. Fix the
#    *model*, not the sampler settings. If your fix refuses to start sampling, find out which
#    term of the log-probability is to blame before changing anything else.
# 2. Evidence for your Task 1 diagnosis: get an acceptable `r_hat` for the means out of the
#    *Task 1* posterior draws, without refitting anything.
# 3. One sentence each: does your fix change the fitted density? Would it still work if two
#    clusters had nearly the same mean?

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task2")
# h.check("task2", mu_main=..., w_main=..., sigma=...)   # posterior means; "main" = the big middle cluster

# %% [markdown] tags=["solution"]
# ### Solution
#
# The standard fix is an **ordering constraint** $\mu_0 < \mu_1 < \mu_2$: of the 3! = 6
# equivalent labellings exactly one satisfies it, so the posterior keeps one of the six
# mirror-image modes. `transform=pm.distributions.transforms.ordered` does this by sampling
# the first mean and the log of the gaps between neighbours.
#
# The catch is the starting point. All three means start at the prior mean, the gaps are zero,
# and the log of zero is `-inf`:

# %% tags=["solution"]
ordered = pm.distributions.transforms.ordered

with pm.Model(coords={"component": range(K)}) as ordered_model:
    mu = pm.Normal("mu", 20, 10, dims="component", transform=ordered)
    sigma = pm.HalfNormal("sigma", 3)
    w = pm.Dirichlet("w", np.ones(K), dims="component")
    pm.NormalMixture("velocity", w=w, mu=mu, sigma=sigma, observed=v)

print("default start, on the transformed scale:", ordered_model.initial_point()["mu_ordered__"])
print("log-probability terms there:", ordered_model.point_logps())

# %% [markdown] tags=["solution"]
# So the transform needs **strictly increasing initial values**. Spreading them evenly over
# the range of the data is a good general-purpose choice: it also starts each component near
# a different part of the data, which matters more and more as the mixtures get flexible
# (Task 5). We pass them to `pm.sample` rather than as `initval=` on the variable, because
# `pm.compute_log_likelihood` - needed for LOO in Task 4 - refuses models with non-default
# initial values.

# %% tags=["solution"]
def spread_init(K):
    return {"mu": np.linspace(v.min(), v.max(), K)}


with ordered_model:
    start = time.time()
    ordered_idata = pm.sample(initvals=spread_init(K), random_seed=RANDOM_SEED)
    ordered_seconds = time.time() - start

print("divergences:", int(ordered_idata.sample_stats["diverging"].sum()))
az.summary(ordered_idata, round_to=3)

# %% tags=["solution"]
# Part 2: relabel the naive draws after the fact, by sorting the means within every draw
naive_mu = naive_idata.posterior["mu"]
relabelled = naive_mu.copy(data=np.sort(naive_mu.values, axis=-1))
print("r_hat of mu, naive draws as sampled: ", az.rhat(naive_mu).values.round(2))
print("r_hat of mu, naive draws relabelled:", az.rhat(relabelled).values.round(3))
print("posterior means, relabelled naive:  ", relabelled.mean(("chain", "draw")).values.round(2))
print("posterior means, ordered model:     ", ordered_idata.posterior["mu"].mean(("chain", "draw")).values.round(2))

# %% tags=["solution"]
post = ordered_idata.posterior
assert h.check(
    "task2",
    mu_main=post["mu"].sel(component=1).mean(),
    w_main=post["w"].sel(component=1).mean(),
    sigma=post["sigma"].mean(),
)

# %% [markdown] tags=["solution"]
# Clean: no divergences, `r_hat` of 1.00 throughout, roughly 1,000 effective draws or more. The
# clusters sit at 9.8, 21.4 and 32.6 with weights 0.10, 0.86 and 0.05 and a common spread of
# 2.1.
#
# **The naive chains were fine.** Sorting the means within each draw maps all six copies onto
# one, and the `r_hat` of the naive fit drops from 2.3 to 1.001, with the same posterior means
# as the ordered model. (Relabelling after the fact like this works here because the clusters
# are far apart; doing it properly means permuting `w` along with `mu`.)
#
# **Does the constraint change the fit?** No. Prior and likelihood are symmetric, so
# restricting to $\mu_0 < \mu_1 < \mu_2$ keeps exactly one of six identical copies; the
# implied density over velocities is untouched.
#
# **Would it always work?** No. Ordering the means only separates the labels if the means are
# far apart relative to their posterior uncertainty. Two clusters with nearly equal means
# (but, say, different spreads) can still swap roles *inside* the ordered region. Then you
# order on a different parameter, or drop exchangeability and give each component its own
# informative prior (`mu ~ Normal([10, 21, 33], 2)`) - a softer fix that requires you to know
# roughly where the clusters are. Keep this in mind for Tasks 4 and 5.

# %% [markdown]
# ## Task 3 · Which galaxies belong together?
#
# The mixture likelihood you have been using sums over the unknown cluster of each galaxy,
# so the cluster assignments appear nowhere in the posterior. The astronomers want them.
#
# **Deliver**
# 1. From the Task 2 posterior, $P(\text{galaxy } i \text{ belongs to cluster } k)$ for every
#    galaxy and cluster. Plot it against velocity. Which galaxies are genuinely ambiguous?
# 2. The same model written the other way: with an explicit discrete assignment for every
#    galaxy. Compare what the sampler had to do, how long it took, the effective sample sizes
#    and the membership probabilities.
# 3. Which formulation would you use by default, and why?

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task3")
# h.check("task3", p_main_16084=..., p_main_26995=...)   # P(main cluster) for the galaxies at 16.084 and 26.995

# %% [markdown] tags=["solution"]
# ### Solution
#
# Given parameters $\theta = (w, \mu, \sigma)$, Bayes' rule for a single galaxy gives the
# **responsibility** of cluster $k$:
#
# $$r_{ik}(\theta) = P(z_i = k \mid v_i, \theta) =
# \frac{w_k \, \text{N}(v_i \mid \mu_k, \sigma_k)}{\sum_j w_j \, \text{N}(v_i \mid \mu_j, \sigma_j)}$$
#
# Averaging $r_{ik}$ over posterior draws of $\theta$ integrates the parameters out and
# gives the membership probability. Nothing needs to be refitted. (The helper takes an
# optional `nu` so that it also works for the Student-t components of Task 5.)

# %% tags=["solution"]
def responsibilities(idata, x=v, nu=None, num_samples=None):
    """P(z = k | x, theta) for every posterior draw: array (draws, len(x), K)."""
    post = az.extract(idata, var_names=["mu", "sigma", "w"], num_samples=num_samples, random_seed=RANDOM_SEED)
    mu, w = post["mu"].values.T[:, None, :], post["w"].values.T[:, None, :]  # (draws, 1, K)
    sigma = post["sigma"].values
    sigma = sigma.T[:, None, :] if sigma.ndim == 2 else sigma[:, None, None]
    z = (np.asarray(x)[None, :, None] - mu) / sigma
    logpdf = stats.norm.logpdf(z) if nu is None else stats.t.logpdf(z, nu)
    log_r = np.log(w) + logpdf - np.log(sigma)
    log_r -= log_r.max(axis=-1, keepdims=True)
    r = np.exp(log_r)
    return r / r.sum(axis=-1, keepdims=True)


CLUSTERS = ["near", "main", "far"]
resp = responsibilities(ordered_idata)
membership = pd.DataFrame(resp.mean(axis=0), columns=CLUSTERS, index=pd.Index(v, name="velocity"))

fig, ax = plt.subplots(figsize=(9, 3.5))
for k, name in enumerate(CLUSTERS):
    ax.plot(v, membership[name], "o", ms=4, color=f"C{k}", label=name)
ax.set(xlabel="velocity (1000 km/s)", ylabel="membership probability")
ax.legend(title="cluster");

# %% tags=["solution"]
ambiguous = membership[membership.max(axis=1) < 0.99]
ambiguous.round(3)

# %% [markdown] tags=["solution"]
# Now the explicit version. Each galaxy gets a `Categorical` assignment `z`, and the
# likelihood is an ordinary Normal around `mu[z]`. NUTS cannot move discrete variables, so
# PyMC builds a compound sampler: NUTS for the continuous parameters, a Gibbs-type
# Metropolis step for the 82 assignments.

# %% tags=["solution"]
with pm.Model(coords={"component": range(K), "galaxy": range(N)}) as explicit_model:
    mu = pm.Normal("mu", 20, 10, dims="component", transform=ordered)
    sigma = pm.HalfNormal("sigma", 3)
    w = pm.Dirichlet("w", np.ones(K), dims="component")
    z = pm.Categorical("z", p=w, dims="galaxy")
    pm.Normal("velocity", mu[z], sigma, observed=v, dims="galaxy")
    start = time.time()
    explicit_idata = pm.sample(initvals=spread_init(K), random_seed=RANDOM_SEED)
    explicit_seconds = time.time() - start

az.summary(explicit_idata, var_names=["mu", "sigma", "w"], round_to=3)

# %% tags=["solution"]
z_draws = explicit_idata.posterior["z"].stack(sample=("chain", "draw")).values  # (galaxy, draws)
membership_explicit = pd.DataFrame(
    {name: (z_draws == k).mean(axis=1) for k, name in enumerate(CLUSTERS)}, index=membership.index
)
print(f"wall-clock time including compilation: marginalised {ordered_seconds:.0f} s, explicit {explicit_seconds:.0f} s")
print(f"largest difference in any membership probability: {np.abs(membership_explicit - membership).values.max():.3f}")

ess = pd.DataFrame({
    name: az.summary(idata, var_names=["mu", "sigma", "w"])["ess_bulk"].astype(float)
    for name, idata in [("marginalised", ordered_idata), ("explicit", explicit_idata)]
})
ess.round(0)

# %% tags=["solution"]
assert h.check(
    "task3",
    p_main_16084=membership.loc[16.084, "main"],
    p_main_26995=membership.loc[26.995, "main"],
)

# %% [markdown] tags=["solution"]
# Only four galaxies are at all ambiguous, and they are the ones next to the two voids: the
# pair at 16.1 (main body with probability 0.95, otherwise near group) and the two just below
# 27 (0.94 and 0.90 main body, otherwise far group). Everything else is assigned with
# probability above 0.99.
#
# The two formulations are the same model and give the same answers: identical parameter
# summaries, and membership probabilities within 0.013 of each other - Monte Carlo noise
# in the explicit version, which estimates each probability by counting how often a discrete
# `z` took a value, whereas the marginalised version averages exact conditional probabilities.
# (The `RuntimeWarning` comes from computing `r_hat` for assignments that never change.)
#
# With clusters this well separated the explicit sampler copes: its effective sample sizes
# are in the same range (lower for five of the seven parameters, higher for the near-cluster
# mean). But it needed a compound sampler with 82 extra discrete variables, cannot use
# nutpie, and took several times longer. It also scales badly: when components overlap, a
# Gibbs step that moves one galaxy at a time is slow to reassign a whole group, and there is
# no gradient to help. **Marginalise by default** - you lose nothing, because the assignments
# can always be recovered afterwards as above. Reach for explicit assignments only when
# something downstream needs them inside the model.

# %% [markdown]
# ## Task 4 · How many clusters?
#
# Three was the eye's answer. Let the data have a say: fit the same shared-spread mixture for
# **K = 2, 3, 4, 5, 6**.
#
# **Deliver**
# 1. A table of sampler diagnostics per K (worst `r_hat`, smallest bulk ESS, divergences).
#    Do not expect all of them to be pretty - say what the ugly ones are telling you.
# 2. The `az.compare` table, and how far you trust it with N = 82.
# 3. A posterior predictive check that separates at least one K from the rest: the fitted
#    density over the data, plus a test statistic aimed at something the astronomers care
#    about.
# 4. For the largest K: what do the posterior weights of the extra components look like?
# 5. Your answer to "how many clusters?", with the honest amount of hedging.

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task4")
# h.check("task4", elpd_gain_3_vs_2=..., sigma_k6=...)   # elpd_loo(K=3) - elpd_loo(K=2); posterior mean of sigma for K=6

# %% tags=["solution"]
def fit_mixture(K, kind="shared", nu=4, target_accept=0.95):
    """Ordered K-component mixture. kind: 'shared' sd, 'unequal' sds, or 'student' (unequal sds, t components)."""
    with pm.Model(coords={"component": range(K), "galaxy": range(N)}) as model:
        mu = pm.Normal("mu", 20, 10, dims="component", transform=ordered)
        w = pm.Dirichlet("w", np.ones(K), dims="component")
        if kind == "shared":
            sigma = pm.HalfNormal("sigma", 3)
            pm.NormalMixture("velocity", w=w, mu=mu, sigma=sigma, observed=v, dims="galaxy")
        else:
            sigma = pm.LogNormal("sigma", 0, 0.5, dims="component")
            if kind == "unequal":
                pm.NormalMixture("velocity", w=w, mu=mu, sigma=sigma, observed=v, dims="galaxy")
            else:
                components = pm.StudentT.dist(nu=nu, mu=mu, sigma=sigma, shape=K)
                pm.Mixture("velocity", w=w, comp_dists=components, observed=v, dims="galaxy")
        idata = pm.sample(initvals=spread_init(K), target_accept=target_accept, random_seed=RANDOM_SEED)
        pm.compute_log_likelihood(idata, progressbar=False)
    return model, idata


def diagnostics(fits):
    rows = {}
    for name, (_, idata) in fits.items():
        summ = az.summary(idata, var_names=["mu", "sigma", "w"])
        rows[name] = {"max r_hat": summ.r_hat.astype(float).max(), "min bulk ESS": summ.ess_bulk.astype(float).min(),
                      "divergences": int(idata.sample_stats["diverging"].sum())}
    return pd.DataFrame(rows).T


shared = {f"K={k}": fit_mixture(k) for k in range(2, 7)}
diagnostics(shared).round(3)

# %% tags=["solution"]
comparison = az.compare({name: idata for name, (_, idata) in shared.items()})
comparison

# %% tags=["solution"]
x_grid = np.linspace(5, 40, 400)


def density_draws(idata, nu=None, num_samples=500):
    """Posterior draws of the mixture density on x_grid: (draws, len(x_grid))."""
    post = az.extract(idata, var_names=["mu", "sigma", "w"], num_samples=num_samples, random_seed=RANDOM_SEED)
    mu, w = post["mu"].values.T[:, None, :], post["w"].values.T[:, None, :]
    sigma = post["sigma"].values
    sigma = sigma.T[:, None, :] if sigma.ndim == 2 else sigma[:, None, None]
    z = (x_grid[None, :, None] - mu) / sigma
    pdf = stats.norm.pdf(z) if nu is None else stats.t.pdf(z, nu)
    return (w * pdf / sigma).sum(axis=-1)


def plot_density(idata, ax, title, nu=None, color="C0"):
    dens = density_draws(idata, nu=nu)
    lo, hi = np.quantile(dens, [0.03, 0.97], axis=0)
    plot_velocities(ax)
    ax.fill_between(x_grid, lo, hi, color=color, alpha=0.3)
    ax.plot(x_grid, dens.mean(axis=0), color=color)
    ax.set_title(title, fontsize=10)


fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, name in zip(axes, ["K=2", "K=3", "K=6"]):
    plot_density(shared[name][1], ax, f"shared spread, {name}")

# %% tags=["solution"]
# Test statistic: how many of 82 galaxies in a replicated survey are faster than 30,000 km/s? (observed: 3)
rows = {}
for name, (model, idata) in shared.items():
    with model:
        ppc = pm.sample_posterior_predictive(idata, random_seed=RANDOM_SEED, progressbar=False)
    n_far = (ppc.posterior_predictive["velocity"] > 30).sum("galaxy").values.ravel()
    rows[name] = {"mean": n_far.mean(), "P(none)": (n_far == 0).mean(), "P(3 or more)": (n_far >= 3).mean()}
print("observed number of galaxies above 30:", int((v > 30).sum()))
pd.DataFrame(rows).T.round(3)

# %% tags=["solution"]
k6 = shared["K=6"][1]
w6 = az.extract(k6, var_names="w").values  # (component, draws)
pd.DataFrame({
    "mean of mu": k6.posterior["mu"].mean(("chain", "draw")).values,
    "mean of w": w6.mean(axis=1),
    "expected galaxies": N * w6.mean(axis=1),
    "P(w < 1/82)": (w6 < 1 / N).mean(axis=1),
}, index=pd.Index(range(6), name="component")).round(3)

# %% tags=["solution"]
az.plot_forest(k6, var_names=["w"], combined=True);

# %% tags=["solution"]
elpd = {name: az.loo(idata).elpd for name, (_, idata) in shared.items()}
assert h.check(
    "task4",
    elpd_gain_3_vs_2=elpd["K=3"] - elpd["K=2"],
    sigma_k6=k6.posterior["sigma"].mean(),
)

# %% [markdown] tags=["solution"]
# **Diagnostics.** K = 2 to 4 sample cleanly. K = 5 and 6 are usable but visibly harder
# (`r_hat` 1.02, a couple of hundred effective draws, a stray divergence), even at
# `target_accept=0.95`. The ordering constraint removed the relabelling copies, but an
# over-specified mixture has other ways to be multimodal: a component the data do not need can
# sit on the pair at 16.1, help tile the main body, or wait in the upper tail, and the sampler
# has to move between these. Sticky sampling at large K is itself a symptom of redundancy.
#
# **LOO** is clear about one thing: K = 2 is not enough (about 18 elpd behind K = 3, 20 behind
# the leader). Among the rest it barely discriminates. K = 3, 4 and 5 are within 1-2 elpd of
# each other; K = 6 leads them by 4-5 with a standard error of 2-3 - under two standard
# errors, with N = 82 flagged as small and two Pareto-k values above 0.7 for K = 6, which
# make precisely that estimate the least trustworthy one in the table.
#
# **Posterior predictive checks** agree. K = 2 has to treat the three far galaxies as the
# tail of the main body: a replicated survey contains 0.4 galaxies above 30,000 km/s on
# average and three or more only 2% of the time. Every K of 3 or more reproduces the far
# group (3.3-3.7 on average) - the check cannot tell them apart. In the density plot K = 6
# differs from K = 3 mainly by splitting the main body into two humps at about 20 and 23.
#
# **Weights.** At K = 6 the two big components carry 0.41 and 0.31; the two extra ones (at
# 16.4 and 25.6) get 0.06 and 0.09, with intervals stretching from about 0.01 to 0.27. They
# are not empty - each found a handful of galaxies to explain - but how many is almost
# undetermined. Note also what happened to `sigma`: it halved, from 2.1 at K = 3 to 1.0.
# Hold that thought for Task 5.
#
# **Answer so far:** at least three clusters, and the two voids appear in every model with
# K >= 3. Whether the main body is one cluster or several, 82 velocities and this model
# family cannot say.

# %% [markdown]
# ## Task 5 · Clusters are not all the same size
#
# One standard deviation for a group of 7 tightly packed galaxies *and* for a main body that
# spans 10,000 km/s is an assumption of convenience. Relax it, in two steps, staying with
# K = 3:
#
# - each cluster gets **its own standard deviation**;
# - additionally, the components get **heavy tails** (Student-t with a small, fixed number of
#   degrees of freedom), the usual recipe for making a mixture robust to stragglers.
#
# **Deliver**
# 1. A prior for the cluster spreads that an astronomer would sign off on. (Rich galaxy
#    clusters have velocity dispersions of the order of 1,000 km/s.)
# 2. Both fits with their diagnostics. More flexible mixtures have more ways to go wrong:
#    watch the initial values, and if a fit misbehaves work out *what* it is undecided about.
# 3. A LOO comparison with all the Task 4 models. What was K = 6 really doing?
# 4. What changed for the boundary galaxies of Task 3?

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task5")
# h.check("task5", sigma_near=..., sigma_main=...)   # posterior means, unequal-spread Gaussian model

# %% [markdown] tags=["solution"]
# ### Solution
#
# `LogNormal(0, 0.5)` puts the median spread at 1,000 km/s with 95% of its mass between about
# 400 and 2,700 km/s: centred on what a rich cluster looks like, with room for a looser
# structure. It also keeps a component from inflating into a flat "background" that covers
# everything, which is the classic way for an unequal-variance mixture to go wrong.
# `fit_mixture` above already knows the two new kinds.

# %% tags=["solution"]
flexible = {"K=3 unequal": fit_mixture(3, kind="unequal"), "K=3 student-t": fit_mixture(3, kind="student")}
diagnostics(flexible).round(3)

# %% tags=["solution"]
az.summary(flexible["K=3 unequal"][1], var_names=["mu", "sigma", "w"], round_to=3)

# %% tags=["solution"]
az.summary(flexible["K=3 student-t"][1], var_names=["mu", "sigma", "w"], round_to=3)

# %% tags=["solution"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
plot_density(flexible["K=3 unequal"][1], axes[0], "K=3, own spread per cluster", color="C1")
plot_density(flexible["K=3 student-t"][1], axes[1], "K=3, Student-t(4) components", nu=4, color="C2")

# %% tags=["solution"]
az.plot_trace(flexible["K=3 student-t"][1], var_names=["mu"]);

# %% tags=["solution"]
all_fits = {**shared, **flexible}
comparison_all = az.compare({name: idata for name, (_, idata) in all_fits.items()})
comparison_all

# %% tags=["solution"]
final_idata = flexible["K=3 unequal"][1]
resp_final = responsibilities(final_idata)
membership_final = pd.DataFrame(resp_final.mean(axis=0), columns=CLUSTERS, index=membership.index)
boundary = [10.406, 16.084, 16.170, 18.419, 25.633, 26.690, 26.995, 32.065]
pd.concat({"shared spread": membership.loc[boundary], "own spread": membership_final.loc[boundary]}, axis=1).round(3)

# %% tags=["solution"]
post = final_idata.posterior
assert h.check(
    "task5",
    sigma_near=post["sigma"].sel(component=0).mean(),
    sigma_main=post["sigma"].sel(component=1).mean(),
)

# %% [markdown] tags=["solution"]
# **Own spread per cluster.** Clean diagnostics, and the spreads are as different as Task 0
# suggested: 0.64 for the near group, 2.2 for the main body and 1.3 for the far group (wide
# interval, 0.7-2.2: three galaxies and a prior).
#
# **What was K = 6 really doing?** The unequal-spread K = 3 model ties with shared-spread K = 6
# at the top of the table (difference 2, standard error 2.8) using half the effective
# parameters (6 against 12.5), and beats shared-spread K = 3, 4 and 5 by 6-7 elpd with a
# standard error of 2.5. A single spread cannot be right for both a group with sd 0.5 and one
# with sd 2.2. Given one spread only, the best the model can do is to shrink it (from 2.1 at
# K = 3 to 1.0 at K = 6) and tile the main body with several narrow components. Much of the
# apparent evidence for "many clusters" was evidence against "equal spreads". One Pareto-k
# warning remains, so treat differences of a few elpd as ties.
#
# **Heavy tails.** The Student-t fit has `r_hat` 1.08 and a few dozen effective draws, and the
# trace shows why. This is *not* label switching - the means are ordered. The chains jump
# between two different **explanations** of the data: one where the third component is the far
# group (means near 21 and 33), and one where components two and three split the main body
# at about 20 and 23 and the three far galaxies are absorbed by the tails as stragglers.
# Heavy tails make three galaxies 10,000 km/s away cheap to explain without a cluster of
# their own, which frees the third component to do something else. With so few jumps the time
# spent in each explanation is not estimated reliably, so every number from this fit (including
# its row in the LOO table, a tie with the leaders) is rough. The lesson is about modelling:
# robustness changes what "cluster" means. For a question that is *about* small distant groups
# and the voids around them, a component that explains them away is the wrong tool, and we
# do not use it below.
#
# **Boundary galaxies.** With a tight near cluster the pair at 16.1 belongs to the main body
# with certainty (before: 0.95), and the two galaxies just below 27 are now main-body members
# with probability 0.98 rather than 0.90-0.94, because the far group is no longer credited
# with the main body's width.

# %% [markdown]
# ## Task 6 · What the astronomers can say
#
# Two deliverables for the paper. Use the model you would defend, and show how much the
# answers move under a credible alternative.
#
# **1. Co-clustering.** Cluster labels are a modelling convenience; "are galaxies *i* and *j*
# in the same cluster?" is not. **Deliver** the posterior co-clustering probability for all
# pairs (a picture) and for the pairs listed below (numbers).
#
# **2. The void.** No galaxy was observed between 27,500 and 31,500 km/s. **Deliver** the
# posterior predictive mass of that interval, the number of galaxies a new survey of the same
# size should expect to find in it, and the probability that such a survey finds it empty
# again. Compare with what a single smooth (one-component) population would predict. Then
# write the two or three sentences that go in the paper.

# %%
VOID = (27.5, 31.5)
PAIRS = [(16.084, 16.170), (16.084, 10.406), (16.084, 18.419), (26.995, 25.633), (26.995, 32.065)]

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task6")
# h.check("task6", coclust_16084_18419=..., void_expected=..., p_void_empty=...)

# %% [markdown] tags=["solution"]
# ### Solution
#
# Given the parameters the assignments of two different galaxies are independent, so
# $P(z_i = z_j \mid \theta) = \sum_k r_{ik}\, r_{jk}$; averaging over posterior draws gives the
# co-clustering probability. It is invariant to relabelling - you could compute it from the
# *naive* Task 1 fit and get the right answer - and it can be compared across models with
# different K.

# %% tags=["solution"]
def coclustering(resp):
    """Posterior P(z_i = z_j) from responsibilities (draws, N, K): array (N, N)."""
    return np.einsum("sik,sjk->ij", resp, resp) / resp.shape[0]


cc_final = coclustering(resp_final)
cc_k6 = coclustering(responsibilities(shared["K=6"][1], num_samples=1000))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, cc, title in zip(axes, [cc_final, cc_k6], ["K=3, own spread per cluster", "K=6, shared spread"]):
    im = ax.imshow(cc, cmap="viridis", vmin=0, vmax=1, origin="lower")
    ax.set(title=title, xlabel="galaxy (sorted by velocity)", ylabel="galaxy")
    ax.grid(False)
fig.colorbar(im, ax=axes, label="P(same cluster)", shrink=0.8);

# %% tags=["solution"]
idx = {val: int(np.argmin(np.abs(v - val))) for pair in PAIRS for val in pair}
pair_table = pd.DataFrame(
    {"K=3 own spread": [cc_final[idx[a], idx[b]] for a, b in PAIRS],
     "K=6 shared spread": [cc_k6[idx[a], idx[b]] for a, b in PAIRS]},
    index=[f"{a} & {b}" for a, b in PAIRS],
)
pair_table.round(3)

# %% tags=["solution"]
def interval_mass(idata, lo, hi, nu=None):
    """Posterior draws of the predictive mass of [lo, hi] under a fitted mixture."""
    post = az.extract(idata, var_names=["mu", "sigma", "w"])
    mu, w, sigma = post["mu"].values.T, post["w"].values.T, post["sigma"].values
    sigma = sigma.T if sigma.ndim == 2 else sigma[:, None]
    cdf = (lambda x: stats.norm.cdf((x - mu) / sigma)) if nu is None else (lambda x: stats.t.cdf((x - mu) / sigma, nu))
    return (w * (cdf(hi) - cdf(lo))).sum(axis=1)


rows = {}
for name, nu in [("K=3 unequal", None), ("K=3", None), ("K=6", None), ("K=3 student-t", 4)]:
    mass = interval_mass(all_fits[name][1], *VOID, nu=nu)
    rows[name] = {"mass": mass.mean(), "expected galaxies": N * mass.mean(),
                  "lo": N * np.quantile(mass, 0.03), "hi": N * np.quantile(mass, 0.97),
                  "P(empty)": ((1 - mass) ** N).mean()}

# a single smooth population, fitted by moments
single = stats.norm(v.mean(), v.std(ddof=1))
m1 = single.cdf(VOID[1]) - single.cdf(VOID[0])
rows["single Normal"] = {"mass": m1, "expected galaxies": N * m1, "P(empty)": (1 - m1) ** N}
void_table = pd.DataFrame(rows).T
void_table.round(3)

# %% tags=["solution"]
assert h.check(
    "task6",
    coclust_16084_18419=pair_table.loc["16.084 & 18.419", "K=3 own spread"],
    void_expected=void_table.loc["K=3 unequal", "expected galaxies"],
    p_void_empty=void_table.loc["K=3 unequal", "P(empty)"],
)

# %% [markdown] tags=["solution"]
# **Who belongs together.** The three-way split by the two voids is solid under both models:
# galaxies on opposite sides of a void share a cluster with probability 0.03 or less (16.084
# with 10.406: 0.00; 26.995 with 32.065: 0.03 and 0.00). Inside the main body the answer
# depends on a modelling choice the data cannot settle (Task 5: the two models tie on LOO).
# With one broad main cluster everything between 16,000 and 27,000 km/s belongs together.
# With six narrow components the main body splits into two sub-groups at roughly 21,500
# km/s, and the pair at 16.1 stands alone: the probability that 16.084 and 18.419 share a
# cluster drops from 1.00 to 0.17. Report the first as established and the second as a
# hypothesis.
#
# **The void.** Under our preferred model a new survey of 82 galaxies should expect **0.8
# galaxies** between 27,500 and 31,500 km/s (94% interval 0.1-2.9) and would find the stretch
# empty again with probability **0.54**. The alternatives move this between 0.7 and 1.5
# expected galaxies and between 0.27 and 0.60 for the probability of an empty void. A single
# smooth population would put **5.1** galaxies there and leave it empty once in 200 surveys.
#
# ### For the paper
#
# > *The velocities are incompatible with a single smooth population. They resolve into a
# > foreground group of 7 galaxies near 9,700 km/s, a main body of 72 between 16,000 and
# > 27,000 km/s and a background group of 3 beyond 32,000 km/s; cross-group co-clustering
# > probabilities are below 0.03 under all models considered. The interval 27,500-31,500
# > km/s, in which no galaxy was observed, is expected to hold 0.7-1.5 of 82 galaxies under
# > the cluster models, against 5.1 for a single Gaussian population - a density three to
# > eight times below the smooth expectation. We cannot claim that the void is empty: with
# > only three galaxies beyond it, the extent of the background group is poorly constrained,
# > and a repeat survey would find at least one galaxy there with probability 0.4-0.7.
# > Whether the main body is a single structure or two, divided near 21,500 km/s, is not
# > resolved by these data (difference in expected log predictive density 2, s.e. 2.8).*
#
# One caveat belongs in the paper as well: the interval was chosen *after* seeing where the
# gap is. That flatters the void in the comparison with the smooth population (some stretch
# of 4,000 km/s would look thin by chance), though not by anything like a factor of 200.

# %% [markdown]
# ## Going further
#
# - The famous typo: set the 78th velocity to Roeder's 26960 and rerun. Which of your numbers
#   move, and which conclusions?
# - Fit an *overfitted* mixture, K = 10 with a sparse `Dirichlet(0.1)` prior on the weights,
#   and report the posterior of the number of components holding at least one galaxy's worth
#   of weight. Expect the sampler to struggle, and decide how much of the output you believe.
# - Verify the invariance claim: compute the co-clustering matrix from the *naive* Task 1
#   fit, broken `r_hat` and all, and compare it with the ordered model's.
# - Repeat the void analysis for the other gap (10,400-16,100 km/s), and do the
#   look-elsewhere correction properly: simulate surveys from the single-Normal model and
#   record the emptiest 4,000 km/s window in each.

# %%
h.progress()
