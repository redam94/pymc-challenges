# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # C10 · Who draws the whistle? An item-response model of NBA foul calls
#
# | | |
# |---|---|
# | **Difficulty** | ★★★★★ |
# | **Time** | 5-6 hours |
# | **Data** | NBA Last Two Minute reports 2015-2021: 46,861 reviewed plays with the player who (may have) committed a foul, the player who was disadvantaged, and the league's verdict |
# | **Skills** | Item-response (Rasch) models · crossed random effects with ~1,600 parameters · identifiability and location invariance · centred vs non-centred geometry · `coords`/`dims` with large index arrays · `nutpie` · group-level predictors · stratified posterior predictive checks · shrinkage · decisions about ranks |
#
# ## The brief
#
# Close NBA games are decided at the free-throw line. A team's analytics group is preparing
# for the trade deadline and asks two questions:
#
# 1. **Which players are best at drawing fouls** late in close games, and which defenders are
#    best at **not being whistled**?
# 2. Both answers must be **separated from the matchup**: a guard who is always defended by
#    foul-prone opponents looks good for free, and a centre who only ever guards stars looks
#    bad for free.
#
# Their intern ranked players by raw foul rate. The top of the list is a set of players
# nobody has heard of, each at 100%. They would like something better - and they want to know
# how far to trust any ranking at all.
#
# Since 2015 the league has reviewed every material play in the last two minutes of close
# games and published its verdict. Each row of the data is one reviewed play.
#
# ## How this notebook works
#
# - Each task states **what to deliver**, not how. Write your code in the `YOUR CODE HERE` cells.
# - Stuck? `h.hint("task2")` reveals hints one level at a time: *nudge → approach → code skeleton*.
#   Try to get by on nudges.
# - `h.check("task2", sigma_theta=...)` compares your numbers with the reference solution.
# - A full worked solution lives in `notebooks/solutions/`. Open it only when you are done (or truly stuck).
#
# The models here have about 1,600 parameters and 47,000 observations.
# `pm.sample(nuts_sampler="nutpie")` fits each in roughly a minute; budget accordingly.

# %%
import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats
from scipy.special import expit

from pymc_challenges import Hints, data

RANDOM_SEED = 2021
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-variat")

h = Hints("C10")
h.tasks()

# %% [markdown]
# ## The data
#
# | column | meaning |
# |---|---|
# | `committing` | the player who committed the foul - or *would have*, had one been called |
# | `disadvantaged` | the player on the receiving end |
# | `decision` | the league's review: `CC` correct call, `IC` incorrect call, `CNC` correct no-call, `INC` incorrect no-call |
# | `committing_position`, `disadvantaged_position` | G(uard), F(orward), C(entre) and hybrids such as `G-F` |

# %%
data.describe("nba_fouls")
fouls = data.load("nba_fouls")
fouls.head()

# %%
fouls.decision.value_counts()

# %% [markdown]
# ## Task 0 · Define the outcome, then look at raw rates
#
# **Deliver**
# 1. A binary outcome built from `decision`, with a justification: the four codes allow more
#    than one defensible definition - which one answers the team's question?
# 2. The intern's table: the ten players with the highest raw rate of drawing fouls, with the
#    number of plays each rate is based on.
# 3. A plot of raw foul-drawing rate against number of plays for every player, and one
#    sentence on why ranking by raw rate is a bad idea.

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task0")
# h.check("task0", call_rate=..., players_with_5_or_fewer=...)   # as disadvantaged player

# %% tags=["solution"]
fouls["foul_called"] = fouls.decision.isin(["CC", "IC"]).astype(int)
y = fouls.foul_called.values
print(f"a foul was called on {y.mean():.3f} of reviewed plays")
print(f"a foul *occurred* (CC or INC) on {fouls.decision.isin(['CC', 'INC']).mean():.3f}")

drawn = fouls.groupby("disadvantaged").foul_called.agg(plays="size", raw_rate="mean")
drawn.sort_values(["raw_rate", "plays"], ascending=False).head(10)

# %% tags=["solution"]
print(drawn.plays.describe().round(1).to_string())
print(f"\nplayers with <= 5 plays as the disadvantaged player: {(drawn.plays <= 5).sum()} of {len(drawn)}")
print(f"players with a raw rate of exactly 0 or 1: {drawn.raw_rate.isin([0, 1]).sum()}")

n_grid = np.unique(np.round(np.geomspace(1, 700, 80))).astype(int)
lo, hi = stats.binom.interval(0.95, n_grid, y.mean())
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(drawn.plays, drawn.raw_rate, s=10, alpha=0.5)
ax.fill_between(n_grid, lo / n_grid, hi / n_grid, color="k", alpha=0.12, label="95% range if every player were average")
ax.axhline(y.mean(), color="k", lw=1)
ax.set(xscale="log", xlabel="plays as the disadvantaged player", ylabel="raw rate of fouls drawn")
ax.legend();

# %% tags=["solution"]
assert h.check("task0", call_rate=y.mean(), players_with_5_or_fewer=(drawn.plays <= 5).sum())

# %% [markdown] tags=["solution"]
# **Outcome.** The team wants free throws, and free throws come from the whistle, not from
# what the review later decided should have happened. So `foul_called = decision in {CC, IC}`
# (23% of plays). The alternative, *a foul occurred* (`CC` or `INC`, 28%), measures contact
# rather than calls; it would be the right outcome for a question about referee accuracy.
# With our definition a player's parameter includes any benefit of the doubt he gets from
# referees - for this stakeholder that is a feature.
#
# **Raw rates.** The top ten are all at 100% on one or two plays. The funnel plot shows why:
# the spread of raw rates is mostly binomial noise, and it is widest exactly where players
# have the fewest plays. A quarter of players have five plays or fewer, and more than a
# hundred have a rate of exactly 0 or 1. Ranking by raw rate ranks by *sample size*. It also
# ignores the opponent entirely.

# %% [markdown]
# ## Task 1 · A colleague's Rasch model
#
# Item-response theory was built for exams: the probability that student $i$ answers
# question $j$ correctly depends on the student's ability minus the question's difficulty.
# Here the "student" is the disadvantaged player and the "question" is the committing player:
#
# $$\text{logit}\,P(\text{foul called}) = \mu + \theta_{\text{disadvantaged}} - b_{\text{committing}}$$
#
# A colleague proposes the natural hierarchical version, "with weakly-informative priors":
#
# $$\theta_i \sim \text{Normal}(\mu_\theta, \sigma_\theta), \quad b_j \sim \text{Normal}(\mu_b, \sigma_b), \quad
# \mu \sim \text{Normal}(0, 1.5), \quad \mu_\theta, \mu_b \sim \text{Normal}(0, 5), \quad \sigma_\theta, \sigma_b \sim \text{HalfNormal}(1)$$
#
# **Deliver**
# 1. This model, fitted to **all** plays, with named dimensions for the two sets of players.
# 2. Diagnostics for the five top-level parameters. Something is badly wrong. Show with a
#    plot *which direction* in parameter space the sampler cannot resolve.
# 3. Find a function of $\mu, \mu_\theta, \mu_b$ that **is** well determined - report its
#    posterior mean and `r_hat` - and explain in two sentences what the data can and cannot
#    tell you in this model.

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task1")
# h.check("task1", identified_combination=...)   # posterior mean of the well-determined function

# %% [markdown] tags=["solution"]
# ### Solution
#
# `pd.factorize` turns the two name columns into integer index arrays plus the matching
# labels for `coords`. A player who appears in both roles gets two unrelated parameters, one
# on each dimension.

# %% tags=["solution"]
d_idx, d_names = pd.factorize(fouls.disadvantaged, sort=True)
c_idx, c_names = pd.factorize(fouls.committing, sort=True)
coords = {"disadvantaged": d_names, "committing": c_names}
print(f"{len(d_names)} disadvantaged players, {len(c_names)} committing players, {len(y):,} plays")

with pm.Model(coords=coords) as colleague_model:
    mu = pm.Normal("mu", 0, 1.5)
    mu_theta = pm.Normal("mu_theta", 0, 5)
    mu_b = pm.Normal("mu_b", 0, 5)
    sigma_theta = pm.HalfNormal("sigma_theta", 1)
    sigma_b = pm.HalfNormal("sigma_b", 1)
    theta = pm.Normal("theta", mu_theta, sigma_theta, dims="disadvantaged")
    b = pm.Normal("b", mu_b, sigma_b, dims="committing")
    pm.Bernoulli("foul_called", logit_p=mu + theta[d_idx] - b[c_idx], observed=y)
    colleague_idata = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)

TOP = ["mu", "mu_theta", "mu_b", "sigma_theta", "sigma_b"]
az.summary(colleague_idata, var_names=TOP, round_to=2)

# %% tags=["solution"]
post = colleague_idata.posterior
combo = (post["mu"] + post["mu_theta"] - post["mu_b"]).rename("mu + mu_theta - mu_b")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for chain in post.chain.values:
    axes[0].plot(post["mu_theta"].sel(chain=chain), post["mu_b"].sel(chain=chain), ".", ms=2, alpha=0.5)
    axes[1].plot(post["mu_theta"].sel(chain=chain), lw=0.6)
    axes[2].plot(combo.sel(chain=chain), lw=0.6)
axes[0].set(xlabel="mu_theta", ylabel="mu_b", title="one colour per chain")
axes[1].set(xlabel="draw", title="mu_theta")
axes[2].set(xlabel="draw", title="mu + mu_theta - mu_b")

az.summary(combo.to_dataset(), round_to=3)

# %% tags=["solution"]
assert h.check("task1", identified_combination=combo.mean())

# %% [markdown] tags=["solution"]
# The three location parameters have `r_hat` between 2.6 and 2.9 and an effective sample size
# of about 5: the four chains sit in four different places and each drifts slowly (middle
# panel). Yet the two scales are fine, and the combination $\mu + \mu_\theta - \mu_b$ is
# estimated to ±0.03 with `r_hat` 1.00 and an ESS in the thousands.
#
# The likelihood sees the locations only through that combination. Add a constant to every
# $\theta$ and subtract it from $\mu$, and no probability changes. In the two remaining
# directions the only information is the prior, so the posterior is a ridge as long as the
# priors are wide (sd 1.5 to 5) and as narrow as the data are informative (sd 0.03). NUTS
# with a diagonal mass matrix crosses such a ridge by slow diffusion; more draws or a higher
# `target_accept` would not help. What the data *can* tell us is the log-odds of a call for an
# average player against an average opponent (−1.14, about 24%) and how players differ from
# one another. What they cannot tell us is how to split that level between "players draw
# fouls" and "defenders commit them" - that is a convention, and we have to choose one.

# %% [markdown]
# ## Task 2 · Identify it, then worry about geometry
#
# **Deliver**
# 1. An identified version of the model, with a justification of the constraint you chose
#    (there is more than one), and a prior predictive check on the probability scale.
# 2. Fits of **both** a centred and a non-centred parameterisation of the player effects.
#    Compare them on divergences and on the effective sample size of $\mu$, $\sigma_\theta$
#    and $\sigma_b$, and choose one.
# 3. An explanation of the difference in terms of the data: which players favour which
#    parameterisation, and how many of each are there? Show the relevant posterior geometry
#    for one player of each kind.

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task2")
# h.check("task2", mu=..., sigma_theta=..., sigma_b=...)   # posterior means, identified model

# %% [markdown] tags=["solution"]
# ### Solution
#
# The likelihood only sees $\mu + \theta_i - b_j$. Add a constant to every $\theta$ and
# subtract it from $\mu$ and nothing changes: there are three location parameters and the
# data determine one. Two ways out:
#
# - **soft**: fix the population means at zero, $\theta_i \sim \text{Normal}(0, \sigma_\theta)$.
#   $\mu$ is then the log-odds for an average player against an average opponent.
# - **hard**: `pm.ZeroSumNormal`, which forces the effects to sum to exactly zero.
#
# Either works; we take the soft one here (it generalises directly to the position model of
# Task 3) and use `ZeroSumNormal` there for the position effects: the average of only seven
# position means is far less certain than the average of 770 players, so a soft constraint
# would leave $\mu$ poorly determined.
#
# Priors: calls happen on roughly a quarter of plays, so $\mu \sim \text{Normal}(-1, 1)$.
# $\sigma \sim \text{HalfNormal}(0.5)$ says that a player one sd above average multiplies the
# odds of a call by $e^{\sigma}$: typically less than 1.6, rarely more than 2.7. The prior
# predictive check shows what that means for a random matchup.

# %% tags=["solution"]
def rasch(centred):
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", -1, 1)
        sigma_theta = pm.HalfNormal("sigma_theta", 0.5)
        sigma_b = pm.HalfNormal("sigma_b", 0.5)
        if centred:
            theta = pm.Normal("theta", 0, sigma_theta, dims="disadvantaged")
            b = pm.Normal("b", 0, sigma_b, dims="committing")
        else:
            z_theta = pm.Normal("z_theta", 0, 1, dims="disadvantaged")
            z_b = pm.Normal("z_b", 0, 1, dims="committing")
            theta = pm.Deterministic("theta", sigma_theta * z_theta, dims="disadvantaged")
            b = pm.Deterministic("b", sigma_b * z_b, dims="committing")
        pm.Bernoulli("foul_called", logit_p=mu + theta[d_idx] - b[c_idx], observed=y)
    return model


with rasch(centred=False):
    rasch_prior = pm.sample_prior_predictive(500, random_seed=RANDOM_SEED)

prior = rasch_prior.prior
p_matchup = expit(prior["mu"] + prior["theta"].isel(disadvantaged=0) - prior["b"].isel(committing=0)).values.ravel()
print("prior quantiles of P(foul called) for a random matchup (3%, 50%, 97%):",
      np.quantile(p_matchup, [0.03, 0.5, 0.97]).round(2))

# %% tags=["solution"]
fits, rows = {}, []
for name, centred in [("centred", True), ("non-centred", False)]:
    start = time.perf_counter()
    with rasch(centred):
        fits[name] = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)
    seconds = time.perf_counter() - start
    top = az.summary(fits[name], var_names=["mu", "sigma_theta", "sigma_b"])
    players = az.summary(fits[name], var_names=["theta", "b"])
    rows.append(
        {
            "parameterisation": name,
            "divergences": int(fits[name].sample_stats["diverging"].sum()),
            "ESS mu": top.loc["mu", "ess_bulk"],
            "ESS sigma_theta": top.loc["sigma_theta", "ess_bulk"],
            "ESS sigma_b": top.loc["sigma_b", "ess_bulk"],
            "min ESS players": players.ess_bulk.min(),
            "max r_hat": max(top.r_hat.max(), players.r_hat.max()),
            "wall time (s)": round(seconds),
        }
    )

pd.DataFrame(rows).set_index("parameterisation")

# %% tags=["solution"]
rasch_idata = fits["non-centred"]
az.summary(rasch_idata, var_names=["mu", "sigma_theta", "sigma_b"], ci_kind="hdi", ci_prob=0.94, round_to=3)

# %% tags=["solution"]
busiest = drawn.plays.idxmax()
one_play = drawn.index[(drawn.plays == 1) & (drawn.raw_rate == 1)][0]
print(f"{busiest}: {drawn.plays[busiest]} plays; {one_play}: {drawn.plays[one_play]} play")

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
for row, (name, var) in enumerate([("centred", "theta"), ("non-centred", "z_theta")]):
    draws = az.extract(fits[name], var_names=["sigma_theta", var])
    for ax, player in zip(axes[row], [one_play, busiest]):
        ax.plot(draws[var].sel(disadvantaged=player), np.log(draws["sigma_theta"]), ".", ms=2, alpha=0.4)
        ax.set(xlabel=f"{var}[{player}]  (n = {drawn.plays[player]})", title=name)
    axes[row, 0].set_ylabel("log sigma_theta")

# %% tags=["solution"]
assert h.check(
    "task2",
    mu=rasch_idata.posterior["mu"].mean(),
    sigma_theta=rasch_idata.posterior["sigma_theta"].mean(),
    sigma_b=rasch_idata.posterior["sigma_b"].mean(),
)

# %% [markdown] tags=["solution"]
# Neither parameterisation diverges, and that is worth a moment's thought, because "use
# non-centred when groups have little data" is usually sold as a cure for divergences. Here
# there are so many players that both scales are pinned down to about ±7%, and the funnel is
# correspondingly shallow: in the top-left panel the cloud for a one-play player barely
# narrows as $\sigma_\theta$ falls.
#
# The centred model still pays, in efficiency: ESS for $\sigma_\theta$ is about 280 against
# 980, for $\sigma_b$ 480 against 910, and its worst `r_hat` is 1.016 against 1.006. The
# reason is the same funnel seen collectively. Given the current values of 770 $\theta$s,
# $\sigma_\theta$ is determined to within roughly $1/\sqrt{2 \cdot 770} \approx 2.5\%$, while
# its marginal posterior is ±7% wide - so the sampler can only move $\sigma_\theta$ by
# dragging hundreds of weakly-informed $\theta$s along with it. Non-centring removes that
# coupling for every player whose posterior is essentially his prior.
#
# Who is that? 214 of the 770 disadvantaged players have five plays or fewer; only about
# 150 have a hundred or more. For the second kind the likelihood dominates and it is the
# *non-centred* coordinates that are correlated with the scale (bottom right: Harden's $z$
# must fall when $\sigma_\theta$ rises, because the data fix their product). The many beat
# the few, and we keep the non-centred model. With a different mix of players the answer
# could flip - measure, do not assume.

# %% [markdown]
# ## Task 3 · Does position explain it?
#
# Guards handle the ball and attack the rim; centres set screens and protect it. Give both
# skills a **group-level predictor**: a player's effect is drawn around the mean of his
# position.
#
# **Deliver**
# 1. The model with position means for both $\theta$ and $b$, identified, with clean
#    diagnostics.
# 2. The posterior of the difference between guards (`G`) and centres (`C`) in foul-drawing
#    skill, and the same for $b$. **Do guards draw more fouls than centres?**
# 3. What happens to $\sigma_\theta$ and $\sigma_b$ compared with Task 2, and what does that
#    tell you? Be careful about what a *play* is in this dataset before you interpret the
#    position effects on $b$.

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task3")
# h.check("task3", guard_minus_centre_theta=..., sigma_b=...)   # posterior means

# %% tags=["solution"]
POSITIONS = ["G", "G-F", "F-G", "F", "F-C", "C-F", "C"]  # ordered from small to big
d_position = fouls.groupby("disadvantaged").disadvantaged_position.first().reindex(d_names)
c_position = fouls.groupby("committing").committing_position.first().reindex(c_names)
assert fouls.groupby("disadvantaged").disadvantaged_position.nunique().max() == 1  # one position per player
d_pos_idx = pd.Categorical(d_position, categories=POSITIONS).codes
c_pos_idx = pd.Categorical(c_position, categories=POSITIONS).codes

with pm.Model(coords={**coords, "position": POSITIONS}) as position_model:
    mu = pm.Normal("mu", -1, 1)
    sigma_theta = pm.HalfNormal("sigma_theta", 0.5)
    sigma_b = pm.HalfNormal("sigma_b", 0.5)
    pos_theta = pm.ZeroSumNormal("pos_theta", 0.5, dims="position")
    pos_b = pm.ZeroSumNormal("pos_b", 0.5, dims="position")
    z_theta = pm.Normal("z_theta", 0, 1, dims="disadvantaged")
    z_b = pm.Normal("z_b", 0, 1, dims="committing")
    theta = pm.Deterministic("theta", pos_theta[d_pos_idx] + sigma_theta * z_theta, dims="disadvantaged")
    b = pm.Deterministic("b", pos_b[c_pos_idx] + sigma_b * z_b, dims="committing")
    pm.Bernoulli("foul_called", logit_p=mu + theta[d_idx] - b[c_idx], observed=y)
    position_idata = pm.sample(nuts_sampler="nutpie", random_seed=RANDOM_SEED)

players = az.summary(position_idata, var_names=["theta", "b"])
print(f"divergences {int(position_idata.sample_stats['diverging'].sum())}, "
      f"players: min ESS {players.ess_bulk.min():.0f}, max r_hat {players.r_hat.max():.3f}")
az.summary(position_idata, var_names=["mu", "sigma_theta", "sigma_b", "pos_theta", "pos_b"],
           ci_kind="hdi", ci_prob=0.94, round_to=2)

# %% tags=["solution"]
ppost = position_idata.posterior
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, var, title in [(axes[0], "pos_theta", "drawing fouls (theta)"), (axes[1], "pos_b", "avoiding whistles (b)")]:
    draws = ppost[var].stack(sample=("chain", "draw")).values  # (position, sample)
    lo, hi = np.quantile(draws, [0.03, 0.97], axis=1)
    ax.errorbar(draws.mean(1), POSITIONS, xerr=[draws.mean(1) - lo, hi - draws.mean(1)], fmt="o", capsize=3)
    ax.axvline(0, color="k", lw=1)
    ax.set(title=title, xlabel="position mean (logit scale, sum to zero)")
axes[0].invert_yaxis()

gc_theta = ppost["pos_theta"].sel(position="G") - ppost["pos_theta"].sel(position="C")
gc_b = ppost["pos_b"].sel(position="G") - ppost["pos_b"].sel(position="C")
for label, d in [("theta", gc_theta), ("b", gc_b)]:
    print(f"guard - centre, {label}: {float(d.mean()):+.2f}, 94% HDI {az.hdi(d, prob=0.94).values.round(2)}, "
          f"P(guards higher) = {float((d > 0).mean()):.3f}")

print("\nraw call rate by position of the disadvantaged player:")
print(fouls.groupby("disadvantaged_position").foul_called.mean().reindex(POSITIONS).round(3).to_string())

# %% tags=["solution"]
assert h.check("task3", guard_minus_centre_theta=gc_theta.mean(), sigma_b=ppost["sigma_b"].mean())

# %% [markdown] tags=["solution"]
# **Do guards draw more fouls than centres? Per reviewed play, no.** The raw rates are nearly
# equal (24.3% for guards, 23.9% for centres), and once the opponent is accounted for centres
# and centre-forwards are *higher* than guards by 0.17 on the logit scale (94% HDI −0.31 to
# −0.03 for guard minus centre; the probability that guards are higher is under 1%). The
# wings (`G-F`, `F-G`) are lowest. The adjustment changes the picture because guards are
# most often defended by other guards, the most whistle-prone defenders in the league, so
# their raw rate is flattered by their opponents - exactly the matchup effect the team asked
# us to remove.
#
# The $b$ side is far more structured: from guards to centres $b$ rises by 0.8. Position
# absorbs about half of the between-player variance in $b$ ($\sigma_b$ falls from 0.41 to
# 0.28), and almost none in $\theta$ (0.30 to 0.29).
#
# **Caveat.** A "play" here is an incident the league chose to review, not a possession or a
# minute. Big men are involved in screens, box-outs and rim contests on almost every trip,
# many of which are reviewed and ruled clean, so a centre's large $b$ plausibly says more
# about *which incidents get listed* than about his discipline. Read $b$ as "whistles per
# reviewed incident", and compare defenders **within** a position - which is what the
# hierarchy now does for us: each player is shrunk towards players like him.

# %% [markdown]
# ## Task 4 · Check it where it could fail
#
# The model is additive on the logit scale: a guard's skill is the same whether a guard or a
# centre is defending him. That is a strong claim with 49 position pairings to test it on.
#
# **Deliver** a posterior predictive check of the foul-call rate **for every pair of
# (disadvantaged position, committing position)**, for the models of Task 2 and Task 3, with
# a verdict: is either model adequate for the team's purpose? Where, if anywhere, does
# additivity fail?

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task4")

# %% tags=["solution"]
pair_labels = fouls.disadvantaged_position + " vs " + fouls.committing_position
pair_idx, pair_names = pd.factorize(pair_labels, sort=True)
pair_n = np.bincount(pair_idx)
pair_obs = np.bincount(pair_idx, weights=y) / pair_n


def replicated_pair_rates(idata, n_draws=200):
    """Simulate whole replicated datasets and return the call rate per position pair: (pairs, draws)."""
    draws = az.extract(idata, var_names=["mu", "theta", "b"], num_samples=n_draws, random_seed=RANDOM_SEED)
    theta = draws["theta"].transpose("disadvantaged", "sample").values
    b = draws["b"].transpose("committing", "sample").values
    p = expit(draws["mu"].values[None, :] + theta[d_idx] - b[c_idx])  # (plays, draws)
    y_rep = rng.random(p.shape) < p
    return np.stack([np.bincount(pair_idx, weights=y_rep[:, k]) for k in range(n_draws)], axis=1) / pair_n[:, None]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, (label, idata) in zip(axes, [("players only", rasch_idata), ("players + position", position_idata)]):
    rep = replicated_pair_rates(idata)
    lo, hi = np.quantile(rep, [0.03, 0.97], axis=1)
    outside = (pair_obs < lo) | (pair_obs > hi)
    order = np.argsort(rep.mean(1))
    xs = np.arange(len(order))
    ax.vlines(xs, lo[order], hi[order], color="C0", alpha=0.6, label="94% predictive interval")
    ax.scatter(xs, pair_obs[order], s=14 + 40 * outside[order], c=np.where(outside[order], "C3", "k"), zorder=3,
               label="observed")
    ax.set(title=f"{label}: {outside.sum()} of {len(pair_names)} outside", xlabel="position pair (sorted by prediction)")
    print(f"{label}: outside the interval -> "
          + ", ".join(f"{pair_names[i]} (n={pair_n[i]}, obs {pair_obs[i]:.3f}, pred {rep[i].mean():.3f})"
                      for i in np.where(outside)[0]))
axes[0].set_ylabel("foul-call rate")
axes[0].legend();

# %% [markdown] tags=["solution"]
# **Players only.** Eight of 49 pairings fall outside their 94% interval, where chance would
# give about three, and the misses have a pattern: the model over-predicts calls when a guard
# or wing is matched with a big (`G vs C`: 13.0% observed, 17.1% predicted) and
# under-predicts `G vs G`. Player effects alone cannot carry position, because the many
# low-data players are shrunk to the league mean rather than to the mean of their position.
#
# **Players + position.** Three misses, which is what chance would give - but they are not
# random. All three have a **centre as the committing player**: `C vs C` (21.2% observed,
# 17.1% predicted), `F-C vs C` (19.7% vs 15.7%) and `G vs C` (13.0% vs 15.5%). Centres are
# whistled more than the additive model says when the opponent is another big, and less
# when he is a guard. That is a genuine interaction: the skills are not purely additive.
#
# For ranking players the position model is adequate: the discrepancies are 2 to 4
# percentage points in 3 cells. For predicting a **specific guard-against-centre matchup**
# it is known to run a couple of points high, which we will need to remember in Task 6.

# %% [markdown]
# ## Task 5 · What shrinkage buys you
#
# **Deliver**
# 1. For every player, the model's estimate of his foul-drawing rate **against an average
#    defender**, plotted against his raw rate, with the number of plays visible in the plot.
# 2. The top ten by raw rate next to the top ten by posterior mean. How many plays does the
#    least-observed player in each list have?
# 3. Two sentences for the intern on what the model did to the 100% players, and why that is
#    not "throwing data away".

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task5")
# h.check("task5", min_plays_in_posterior_top10=...)

# %% tags=["solution"]
theta_draws = az.extract(position_idata, var_names="theta").transpose("sample", "disadvantaged").values
mu_draws = az.extract(position_idata, var_names="mu").values

skill = drawn.copy()
skill["position"] = d_position.values
skill["theta"] = theta_draws.mean(0)
skill["model_rate"] = expit(mu_draws[:, None] + theta_draws).mean(0)  # against b = 0

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(skill.raw_rate, skill.model_rate, c=np.log10(skill.plays), s=12 + skill.plays / 6, alpha=0.7, cmap="viridis")
ax.plot([0, 1], [0, 1], color="k", lw=1, ls=":")
ax.axhline(expit(mu_draws).mean(), color="k", lw=1)
ax.set(xlabel="raw rate of fouls drawn", ylabel="posterior mean rate against an average defender", ylim=(0.1, 0.45))
fig.colorbar(sc, label="log10(plays)")

perfect = skill[skill.raw_rate == 1]
print(f"{len(perfect)} players with a raw rate of 100% (at most {perfect.plays.max()} plays): "
      f"model rates {perfect.model_rate.min():.3f} to {perfect.model_rate.max():.3f}; "
      f"league average {expit(mu_draws).mean():.3f}")

# %% tags=["solution"]
raw_top = skill.sort_values(["raw_rate", "plays"], ascending=False).head(10)
post_top = skill.sort_values("theta", ascending=False).head(10)
side_by_side = pd.DataFrame(
    {
        "raw top 10": raw_top.index, "plays ": raw_top.plays.values, "raw ": raw_top.raw_rate.values.round(2),
        "posterior top 10": post_top.index, "plays": post_top.plays.values, "raw": post_top.raw_rate.values.round(2),
        "model rate": post_top.model_rate.values.round(3),
    },
    index=pd.RangeIndex(1, 11, name="rank"),
)
side_by_side

# %% tags=["solution"]
assert h.check("task5", min_plays_in_posterior_top10=post_top.plays.min())

# %% [markdown] tags=["solution"]
# The raw top ten are ten men with one or two plays each. The posterior top ten have between
# 32 and 616 plays and include John Wall, Stephen Curry and James Harden - names a scout would
# recognise. In the scatter plot, players with hundreds of plays (yellow) stay close to the
# diagonal: the data speak for themselves. Players with a handful of plays (purple) are
# pulled almost all the way to the average for their position, whatever their raw rate.
#
# **For the intern.** The model did not discard the 100% players' data; it weighed them. One
# whistle in one play is what you would expect to see about a quarter of the time from an
# *average* player, so it is almost no evidence of skill, and the posterior moves those
# 28 players to model rates between 21% and 27%, against a league average of 22%. A rate of
# 38% sustained over 261 plays, as for Wall, is strong evidence, and the model believes most
# of it (34% against an average defender).

# %% [markdown]
# ## Task 6 · Rankings the front office can use
#
# A posterior mean is still a point estimate, and the front office will read a ranked list as
# if rank 3 were better than rank 7.
#
# **Deliver**
# 1. For the leading candidates: the posterior probability of being among the **ten best
#    foul-drawers in the league**, and a 94% interval for each player's **rank**. Is anybody
#    in the top ten with probability above one half?
# 2. The same treatment, briefly, for defenders who avoid the whistle.
# 3. A specific matchup: **James Harden has the ball and Rudy Gobert is defending.** The
#    posterior of P(foul called) with an interval - and the same with an average centre in
#    place of Gobert, to show what the defender is worth.
# 4. Three sentences for the front office on how to use (and not use) the list.

# %% tags=["task"]
# YOUR CODE HERE

# %% tags=["task"]
# h.hint("task6")
# h.check("task6", max_p_top10=..., p_harden_vs_gobert=...)

# %% tags=["solution"]
def rank_table(draws, names, plays, top=10):
    """draws: (sample, player), larger is better. Rank 1 = best within each posterior draw."""
    ranks = (-draws).argsort(axis=1).argsort(axis=1) + 1
    table = pd.DataFrame(
        {
            "plays": plays,
            "posterior mean": draws.mean(0),
            f"P(top {top})": (ranks <= top).mean(0),
            "rank lo94": np.quantile(ranks, 0.03, axis=0),
            "rank hi94": np.quantile(ranks, 0.97, axis=0),
        },
        index=names,
    )
    return table.sort_values(f"P(top {top})", ascending=False)


draw_ranks = rank_table(theta_draws, d_names, drawn.plays.reindex(d_names).values)
draw_ranks.head(12).round(3)

# %% tags=["solution"]
# the pessimist's view: whose *worst plausible* rank is best?
draw_ranks.sort_values("rank hi94").head(8).round(3)

# %% tags=["solution"]
# defenders: rank the *within-position* part of b (see the caveat in Task 3)
b_draws = az.extract(position_idata, var_names="b").transpose("sample", "committing").values
pos_b_draws = az.extract(position_idata, var_names="pos_b").transpose("sample", "position").values
b_within = b_draws - pos_b_draws[:, c_pos_idx]
committed_plays = fouls.groupby("committing").size().reindex(c_names).values
avoid_ranks = rank_table(b_within, c_names, committed_plays)
avoid_ranks.insert(0, "position", c_position.reindex(avoid_ranks.index).values)
avoid_ranks.head(8).round(3)

# %% tags=["solution"]
harden = list(d_names).index("James Harden")
gobert = list(c_names).index("Rudy Gobert")
pos_b_centre = pos_b_draws[:, POSITIONS.index("C")]

p_gobert = expit(mu_draws + theta_draws[:, harden] - b_draws[:, gobert])
p_avg_centre = expit(mu_draws + theta_draws[:, harden] - pos_b_centre)
met = fouls[(fouls.disadvantaged == "James Harden") & (fouls.committing == "Rudy Gobert")]
print(f"they met on {len(met)} reviewed plays, {met.foul_called.sum()} fouls called")
for label, p in [("Harden vs Gobert", p_gobert), ("Harden vs an average centre", p_avg_centre)]:
    print(f"{label:<28} P(foul called) = {p.mean():.3f}, 94% interval {np.quantile(p, [0.03, 0.97]).round(3)}")
print(f"P(Gobert is whistled less than an average centre in this matchup) = {np.mean(p_gobert < p_avg_centre):.2f}")

# %% tags=["solution"]
fig, ax = plt.subplots(figsize=(8, 4.5))
leaders = draw_ranks.head(15)[::-1]
ax.hlines(leaders.index, leaders["rank lo94"], leaders["rank hi94"], color="C0", lw=2)
for name, row in leaders.iterrows():
    ax.text(row["rank hi94"] + 5, name, f"P(top 10) = {row['P(top 10)']:.2f}, {int(row.plays)} plays", va="center", fontsize=8)
ax.axvline(10, color="k", ls=":", lw=1)
ax.set(xlabel=f"rank among {len(d_names)} players (94% interval)", xlim=(0, 900), title="Where could each leader really rank?");

# %% tags=["solution"]
assert h.check("task6", max_p_top10=draw_ranks["P(top 10)"].max(), p_harden_vs_gobert=p_gobert.mean())

# %% [markdown] tags=["solution"]
# **Is anybody in the top ten with probability above one half? No.** The most likely member is
# Kenneth Faried at 0.44, on 49 plays; five players are above 0.3. The rank intervals are
# enormous - Faried's runs from 1st to 167th, Chalmers' from 1st to 228th. The ordering of
# posterior means in Task 5 was a fragile summary of this.
#
# **Which summary you rank by is a decision, not a detail.** $P(\text{top 10})$ rewards
# uncertainty as well as skill: Matt Barnes, on 25 plays, gets 0.22 with a rank interval of 1
# to 368, while James Harden, on 616 plays, gets 0.03 although he is *certainly* very good
# (11th to 99th). The pessimist's table - sorted by worst plausible rank - is the one to use
# if the team wants a dependable foul-drawer rather than a lottery ticket: John Wall (2nd to
# 77th), then Harden and Curry. Wall and Dwight Howard are the only players in the top five of
# both lists.
#
# **Defenders** are ranked on the within-position part of $b$, so that guards are compared
# with guards. Nobody stands out: the best candidates (Alex Caruso, Robin Lopez, Taj Gibson)
# have a probability of about 0.3 of being in the top ten, and even Lopez, on 196 plays,
# could plausibly rank anywhere from 1st to 216th of 789. The data separate defenders within a
# position much less sharply than the raw rates suggest.
#
# **The matchup.** Harden against Gobert: a foul is called on 24.5% of reviewed plays, 94%
# interval 20% to 30%. Against an average centre it would be 22.4%, and the probability that
# Gobert is whistled *less* than an average centre is only 0.19 - in these reports he is
# probably slightly more whistle-prone than his peers. The four plays on which the two actually
# met (two fouls) tell us almost nothing by themselves; the model borrows Harden's 616 plays
# and Gobert's 395. Remember Task 4, though: the additive model ran about 2.5 points high for
# guards against centres, so shade this estimate down.
#
# **For the front office.** (1) Read the lists as tiers, not ranks: the data cannot
# distinguish 3rd from 30th. (2) Pick the summary that matches the decision - $P(\text{top
# 10})$ for upside, worst plausible rank for reliability - and distrust any candidate with
# fewer than about fifty plays. (3) These are skills **per reviewed incident in the last two
# minutes of close games**, whistles included, referees' habits included. They say nothing
# about how often a player gets into such incidents, which for free throws per game matters
# at least as much.

# %% [markdown]
# ## Going further
#
# - **One player, two skills.** Most players appear in both roles. Give each player a
#   bivariate effect $(\theta_i, b_i)$ with an `LKJCholeskyCov` prior. Are good foul-drawers
#   also whistle-prone defenders?
# - **Matchup interactions.** Task 4 tests additivity only at the position level. Add a
#   position-pair interaction (`ZeroSumNormal` over two dims) and compare with LOO. Is it
#   worth 49 more parameters?
# - **Referee error instead of whistles.** Refit with the outcome "the call was wrong"
#   (`IC` or `INC`). Do some players get the benefit of the doubt?
# - **Does skill drift?** The file has no dates, but the reports do. Get the season for each
#   play from the original L2M data and let $\theta$ follow a random walk across seasons.

# %%
h.progress()
