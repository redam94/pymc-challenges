"""Hand-written copy for the website built by tools/build_site.py.

Per-notebook titles, datasets and techniques come from the README tables; this file holds
the series grouping and the prose of the index, guide, challenges and about pages.
"""

import re
from html import escape
from itertools import cycle

SITE_NAME = "Bayesian Modelling in Practice"
TAGLINE = ("Worked examples and real-data challenges in Bayesian data analysis with PyMC: "
           "from the first prior predictive check to decisions made under uncertainty.")
FOOTER = ("Worked examples and challenges in Bayesian data analysis with PyMC 6, ArviZ 1 and PyTensor 3. "
          "Datasets are real and public, sources cited in each notebook; the few simulated systems say so.")


SEARCH_MARK = ('<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="6.8" cy="6.8" r="4.6" fill="none" '
               'stroke="currentColor" stroke-width="1.5"/><path d="M10.3 10.3 14 14" stroke="currentColor" '
               'stroke-width="1.5" stroke-linecap="round"/></svg>')

# Fifteen clusters, by what the reader is trying to do. Every example sits in exactly one; a series
# shows its examples in the order of `ids`. Each key has an icon, tools/site_assets/img/icon-<key>.webp.
SERIES = [
    dict(key="foundations", title="Foundations",
         ids=["E01", "E02", "E03", "E04", "E05", "E06"],
         intro="The core of applied Bayesian work: writing a model down, checking what the priors claim, reading"
               " a posterior, pooling information across groups, comparing models and handling censored data. "
               "Every challenge assumes these."),
    dict(key="data-work", title="Data, posteriors and reporting",
         ids=["D01", "D02", "D03", "E16", "E42", "E43", "E44"],
         intro="Most models that refuse to sample are data problems in disguise. These cover the craft around "
               "the model: preparing arrays, treating missingness and measurement error as part of the model, "
               "working with posteriors as labelled arrays, and showing uncertainty so that a reader can act on "
               "it."),
    dict(key="inference", title="Inference and computation",
         ids=["E09", "E12", "E14", "E62", "E86"],
         intro="What to do when plain NUTS is not enough: your own JAX code with gradients inside a model, "
               "samplers and approximations beyond the default, inference without a likelihood, ensembles on a "
               "GPU, and how to tell when a model cannot answer the question at all."),
    dict(key="regression", title="Flexible regression",
         ids=["E15", "E17", "E25", "E27", "E29", "E83"],
         intro="Regression when a straight line and a handful of predictors will not do: additive trees, neural "
               "networks, priors that pick a few predictors out of many, quantiles in place of means, studies "
               "pooled by meta-analysis, and repeated measurements modelled jointly with survival."),
    dict(key="outcomes", title="Rankings, choices and other unusual outcomes",
         ids=["E22", "E67", "E68", "E70", "E71", "E81"],
         intro="Likelihoods for data that are not a plain number: head-to-head results, choices among "
               "alternatives, shares of a whole, response times, directions on a circle, and many outcomes that "
               "move together."),
    dict(key="latent", title="Latent groups, topics and partitions",
         ids=["E24", "E26", "E28", "E30", "E31", "E69", "E72", "E73", "E124"],
         intro="Models where the structure itself is unobserved: hidden states, mixtures with an unknown number "
               "of components, random partitions and binary features, topics, raters with no gold standard, and "
               "animals that were present but never detected."),
    dict(key="dependence", title="Dependence in space, time and graphs",
         ids=["E07", "E13", "E18", "E19", "E21", "E23", "E65", "E96", "E97", "E98"],
         intro="Observations that are not independent because of where, when or how they are connected: maps and"
               " areal neighbours, state-space models, self-exciting events, networks, phylogenies, and several "
               "imperfect instruments fused into one estimate."),
    dict(key="causal", title="Causal inference",
         ids=["E20", "E34", "E35", "E63", "E66", "E101", "E102", "E103", "E107"],
         intro="From association to what would have happened otherwise: quasi-experiments, synthetic control, "
               "mediation, causal discovery, online experiments, propensity scores graded against an experiment,"
               " and how much a hidden confounder could change the answer."),
    dict(key="decisions", title="Decisions, design and value of information",
         ids=["E51", "E64", "E82", "E87", "E89", "E90", "E91", "E92", "E99", "E100", "E120", "E121", "E122", "E123"],
         intro="The posterior as the input to a choice: which experiment to run next, where to put the next "
               "sensor, what dose to give, whether a bigger model is worth its price, when acting now beats "
               "waiting for better data, and how many agents to put on the phones."),
    dict(key="physics", title="Physics, astronomy and dynamical systems",
         ids=["E08", "E10", "E11", "E46", "E47", "E48", "E49", "E78", "E79", "E80"],
         intro="When theory supplies the equations: the expansion of the universe from supernovae, a black-hole "
               "ringdown in real LIGO strain, planets in Kepler light curves, the cosmic microwave background "
               "and a gravitational lens fitted pixel by pixel, a neural network inside a differential equation,"
               " chaotic dynamics, and jump-diffusions for market crashes."),
    dict(key="colour", title="Colour, light and images",
         ids=["E74", "E75", "E76", "E77"],
         intro="When the estimate is a colour, a spectrum, a region or a word. How the dyes of Japanese "
               "woodblock prints fade, and what a 1766 Harunobu print looked like when new; why two paints can "
               "match in the shop and differ at home (metamers), inverted from colour to spectrum with a prior "
               "learned from 1,269 Munsell chips; segmenting photographs with an unknown number of regions and "
               "checking the result, and its confidence, against the people who outlined them; and how 110 "
               "languages divide colour space, speaker by speaker - each with uncertainty displays that still "
               "work when colour cannot be used to show the uncertainty."),
    dict(key='biology', title='Cell biology and biophysics',
         ids=['E50', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60', 'E61', 'E88'],
         intro="Mechanistic models of living cells, fitted to what biologists measure: signals that travel "
               "through the membrane and the cytosol, the fluctuation test that showed mutations arise before "
               "selection, why a neuron fires (conductance models fitted to real patch-clamp recordings), "
               "transcriptional bursting read from allele-resolved single-cell RNA counts, and how bacteria "
               "control their size, measured through noisy segmentation, and single molecules tracked in living "
               "nuclei, and super-resolution (STORM) microscopy: localising, detecting and counting single "
               "blinking molecules below the diffraction limit, and kinesin's steps counted from MINFLUX traces,"
               " the forces cells exert on their substrate (traction force microscopy), and decisions from "
               "dynamics: Turing patterns and kinetic proofreading in T cells. See also E50 (reaction-diffusion "
               "in tissues). Also here: cell invasion in a scratch assay, and a neural localiser tried on a full"
               " super-resolution movie."),
    dict(key="media", title="Marketing and media",
         ids=["E32", "E33", "E45", "E84", "E85", "E104", "E105", "E106"],
         intro="Measurement for marketing decisions: a custom likelihood for spend that chases demand, "
               "hierarchical media mix models and a budget choice under uncertainty, one family for every "
               "response curve, who competes with whom, and when to spend before a sell-out or a release."),
    dict(key="elections", title="Elections and public opinion",
         ids=["E36", "E40", "E94", "E95", "E118", "E119"],
         intro="Polls as noisy, partial and sometimes biased measurements: poststratifying an unrepresentative "
               "survey, aggregating polls into a forecast, combining polls with fundamentals and money in the US"
               " midterms, updating on election night, and what the data can and cannot say about ideology."),
    dict(key="public", title="Questions people actually ask",
         ids=["E37", "E38", "E39", "E41", "E108", "E109", "E110", "E111", "E112", "E113", "E114", "E115", "E116", "E117"],
         intro="Current methods explained from the question down: did warming load the dice for this heatwave, "
               "is the epidemic growing right now, how old is this object, is my child on track, is the 100-year"
               " flood still a 100-year flood. Each starts from something a newspaper reader would ask and "
               "answers it in their terms."),
]
SERIES_BY_KEY = {s["key"]: s for s in SERIES}
# The examples under "Recently added" on the home page, newest first. Put a new example at the front.
# recent_ids() adds any example numbered above everything listed here, so the section cannot fall
# behind, and recent_intro() writes the sentence above the cards: there is no copy to keep in step.
RECENT = ["E124", "E123","E122", "E121", "E120","E119", "E118", "E117", "E116", "E115", "E114", "E113", "E112", "E111", "E110", "E109", "E108", "E107", "E106", "E105", "E104", "E103", "E102", "E101", "E100", "E99", "E98", "E97", "E96", "E92", "E95", "E94", "E90", "E88", "E91", "E87", "E89"]


def _number(eid: str) -> int:
    return int(eid[1:])


def recent_ids(ctx) -> list[str]:
    """RECENT, led by every E example numbered above everything in it (highest first)."""
    newest = max(_number(i) for i in RECENT)
    unlisted = sorted((e.id for e in ctx["examples"] if e.id.startswith("E") and _number(e.id) > newest),
                      key=_number, reverse=True)
    if unlisted:
        print(f"  note: {', '.join(unlisted)} not in site_content.RECENT; shown first under 'Recently added'. "
              "Put every id named here at the front of RECENT.")
    return unlisted + RECENT


def recent_intro(ctx, ids: list[str]) -> str:
    """One sentence naming the three series the newest examples belong to, each linked to its section."""
    keys = []
    for i in ids:
        e = ctx["by_id"].get(i)
        if e and e.series and e.series not in keys:
            keys.append(e.series)
        if len(keys) == 3:
            break
    links = [f'<a href="examples.html#{k}">{SERIES_BY_KEY[k]["title"]}</a>' for k in keys]
    if not links:
        return "Newest first."
    # series titles contain commas, so the list is separated by semicolons
    listed = links[0] if len(links) == 1 else "; ".join(links[:-1]) + "; and " + links[-1]
    return f"Newest first. The latest additions are in {listed}."


# The figure shown on each example's card: the k-th image the notebook outputs (1-based, in notebook
# order), or a file saved next to the notebook. Chosen by eye from contact sheets of every figure;
# a notebook missing from the map gets its largest landscape figure.
CARD_FIGURE = {
    "D01": 6, "D02": 5, "D03": 9, "E01": 8, "E02": 7, "E03": 11, "E04": 8, "E05": 5, "E06": 6, "E07": 15,
    "E08": 11, "E09": 7, "E10": 4, "E11": 2, "E12": 1, "E13": 8, "E14": 1, "E15": 8, "E16": 2, "E17": 5,
    "E18": 7, "E19": 7, "E20": 2, "E21": 4, "E22": 2, "E23": 2, "E24": 4, "E25": 6, "E26": 4, "E27": 4,
    "E28": 3, "E29": 8, "E30": 2, "E31": 13, "E32": 6, "E33": 8, "E34": 8, "E35": 2, "E36": 5, "E37": 7,
    "E38": 5, "E39": 6, "E40": 11, "E41": 6, "E42": 2, "E43": 1, "E44": 8, "E45": 8, "E46": 1, "E47": 1,
    "E48": 2, "E49": 4, "E50": 2, "E51": 1, "E52": 3, "E53": 1, "E54": 3, "E55": 1, "E56": 3, "E57": 8,
    "E58": 7, "E59": 2, "E60": 7, "E61": 2, "E62": 3, "E63": 6, "E64": 12, "E65": 2, "E66": 5, "E67": 6,
    "E68": 9, "E69": 8, "E70": 6, "E71": 1, "E72": 14, "E73": 2, "E74": 8, "E75": 7, "E76": 1, "E77": 10,
    "E78": 8, "E79": 8, "E80": 10, "E81": 11, "E82": 1, "E83": 6, "E84": 4, "E85": 4, "E86": 1, "E87": 2,
    "E88": 18, "E89": 8, "E90": 1, "E91": 10, "E92": 5, "E94": 15, "E95": "E95_election_night.gif", "E99": 11, "E100": 6, "E101": 3, "E102": 2, "E103": 4, "E104": 7, "E105": 5, "E106": 6, "E107": 3,
    "E108": "E108_usual_salt_intake.gif", "E109": 1, "E110": 1, "E111": 1, "E112": 1, "E113": 1, "E114": 1, "E115": 1,
    "E116": 1, "E117": 1, "E118": 9, "E119": 4, "E120": 10,
    "E121": 8, "E122": 2, "E123": 9, "E124": 5,
}


def _link(ctx, eid, text=None):
    e = ctx["by_id"].get(eid)
    if not e:
        return escape(text or eid)
    return f'<a href="{e.url}">{escape(text) if text else eid}</a>'


# --------------------------------------------------------------------------- workflow figure

# name, what happens, the example that shows it, whether the step is a check that can send you back
WORKFLOW_STEPS = [
    ("Question", "What decision or claim will the model inform?", "E01", False),
    ("Generative model", "A story of how the data came to be, written as distributions.", "E01", False),
    ("Prior predictive", "Simulate before fitting: do the priors claim anything absurd?", "D03", True),
    ("Inference", "Sample the posterior, or approximate it with care.", "E12", False),
    ("Diagnostics", "Did the computation work? Divergences, R-hat, effective sample size.", "E02", True),
    ("Posterior predictive", "Can the fitted model reproduce the features of the data you care about?", "E03", True),
    ("Decision", "Carry the whole posterior through to the quantity someone acts on.", "E33", False),
]


def workflow(ctx) -> str:
    """The seven steps as a list; assets/figures.js turns it into a track you can step through."""
    items = "".join(
        f'<li class="flow-panel"{" data-check" if check else ""}><h3>{name}</h3><p>{desc}</p>'
        f"<p>See {_link(ctx, eid)}.</p></li>"
        for name, desc, eid, check in WORKFLOW_STEPS
    )
    return f'<div class="flow" data-flow><ol class="flow-panels">{items}</ol></div>'


# Small line figures for the tiles; they draw themselves when scrolled into view (.fig[data-reveal]).
TREE_FIG = (
    '<svg class="fig" data-reveal viewBox="0 0 360 150" role="img" '
    'aria-label="One question about your data branching to three candidate models, one of them picked out.">'
    '<path class="f-muted" data-draw pathLength="1" d="M40 75 H120"/>'
    '<path class="f-muted" data-draw pathLength="1" style="--d:1" d="M120 75 C170 75 170 30 220 30 H300"/>'
    '<path class="f-accent" data-draw pathLength="1" style="--d:2" d="M120 75 H300"/>'
    '<path class="f-muted" data-draw pathLength="1" style="--d:3" d="M120 75 C170 75 170 120 220 120 H300"/>'
    '<circle class="f-fill-line" cx="40" cy="75" r="6"/>'
    '<circle class="f-fill-muted" data-pop style="--d:1" cx="306" cy="30" r="5"/>'
    '<circle class="f-fill-accent" data-pop style="--d:2" cx="308" cy="75" r="7"/>'
    '<circle class="f-fill-muted" data-pop style="--d:3" cx="306" cy="120" r="5"/></svg>'
)
LADDER_FIG = (
    '<svg class="fig" data-reveal viewBox="0 0 360 150" role="img" '
    'aria-label="Three hints as rising steps: a nudge, an approach, then a skeleton.">'
    '<path class="f-accent" data-draw pathLength="1" d="M36 126 H132 V86 H228 V46 H324"/>'
    '<text class="f-text f-text-muted" data-pop x="36" y="114">Nudge</text>'
    '<text class="f-text f-text-muted" data-pop style="--d:2" x="132" y="74">Approach</text>'
    '<text class="f-text" data-pop style="--d:4" x="228" y="34">Skeleton</text></svg>'
)


# --------------------------------------------------------------------------- pages


def _bands(body: str) -> str:
    """Turn each <section class="essay ..."> into a full-width band, alternating white and mist."""
    tones = cycle(["bg-white", "bg-mist"])

    def band(m):
        inner = f'<div class="{m.group(1).strip()}">{m.group(2)}</div>' if m.group(1).strip() else m.group(2)
        return f'<section class="group {next(tones)} on-light"><div class="wrap">{inner}</div></section>'

    return re.sub(r'<section class="essay([^"]*)">(.*?)</section>', band, body, flags=re.S)


def home(ctx):
    n_ex, n_ch = len(ctx["examples"]), len(ctx["challenges"])
    L = lambda i, t=None: _link(ctx, i, t)  # noqa: E731
    first = ctx["by_id"]["E01"].url
    featured = "".join(ctx["card"](ctx["by_id"][i]) for i in ("E10", "E37", "E40", "E16") if i in ctx["by_id"])
    recent_list = recent_ids(ctx)
    recent = "".join(ctx["card"](ctx["by_id"][i]) for i in recent_list if i in ctx["by_id"])
    body = f"""
<section class="hero on-dark bg-black">
  <div class="wrap hero-copy">
    <h1>Bayesian data analysis, one real problem at a time.</h1>
    <p class="hero-lede">{n_ex} worked examples and {n_ch} challenges, each a complete analysis in PyMC, almost all
    of public datasets. Every page shows the model, the code, the checks that passed and the ones that did not, and
    what the posterior means for the question that was asked.</p>
    <p class="actions">
      <a class="btn" href="{first}">Start with the workflow</a>
      <a class="link" href="guide.html">Find a model for your problem</a>
      <a class="link" href="challenges.html">Test yourself</a>
    </p>
  </div>
  <figure class="wrap sampler" data-sampler>
    <p class="sampler-readout" data-readout aria-live="off">Four chains of Hamiltonian Monte Carlo</p>
    <canvas role="img" aria-label="Four Markov chains exploring a curved, banana-shaped posterior. A histogram of
    their draws along the top fills in toward the exact marginal density."></canvas>
    <figcaption>
      <span>Four chains of Hamiltonian Monte Carlo explore a curved posterior. Along the top, a histogram of their
      draws fills in toward the exact marginal, dashed. Click the plot to start all four chains from that point.</span>
      <button type="button" class="pill" data-restart>Restart the chains</button>
    </figcaption>
  </figure>
</section>

<section class="flagship bg-mist on-light" id="why">
  <div class="wrap">
    <h2>Why model this way</h2>
    <p class="flagship-tag">A Bayesian analysis returns a distribution over everything unknown, and each
    observation sharpens it.</p>
    <div class="updater" data-updater>
      <svg viewBox="0 0 720 300" role="img" aria-label="A prior distribution for an unknown rate narrowing into a
      posterior as observations arrive, with its 50 and 94 percent intervals shaded."></svg>
      <p class="updater-readout" data-readout aria-live="polite">An unknown rate starts from a prior and is updated
      by every success and failure observed.</p>
      <p class="updater-controls">
        <button type="button" class="pill pill-accent" data-act="success">Observe a success</button>
        <button type="button" class="pill" data-act="failure">Observe a failure</button>
        <button type="button" class="pill" data-act="ten">Add 10 at random</button>
        <button type="button" class="pill" data-act="reset">Start over</button>
      </p>
    </div>
    <div class="features">
      <div>
        <h3>Uncertainty is the result</h3>
        <p>A Bayesian analysis returns a distribution over everything unknown, $p(\\theta \\mid y) \\propto
        p(y \\mid \\theta)\\,p(\\theta)$. Any quantity you compute from the parameters &mdash; a forecast, a
        ranking, a probability that a threshold is crossed &mdash; comes with its uncertainty attached,
        without a separate derivation.</p>
      </div>
      <div>
        <h3>A model is a story about the data</h3>
        <p>Instead of choosing a test from a menu, you write down how the data could have been generated:
        censoring, missingness, measurement error, groups, space, time, physics. The same language covers
        a regression and a gravitational-wave ringdown, so you learn one way of thinking, not a list of
        procedures.</p>
      </div>
      <div>
        <h3>Checking is part of the method</h3>
        <p>Priors are simulated before they are trusted, samplers are diagnosed, and fitted models are asked
        to reproduce the data. When a model fails, these pages say so and show the repair. Seeing the
        failures is how you learn to spot them in your own work.</p>
      </div>
    </div>
  </div>
</section>

<section class="group bg-white on-light" id="workflow">
  <div class="wrap">
    <h2 class="group-title">The workflow every page follows</h2>
    <p class="group-lede">The examples are not a tour of an API. Each one is a complete iteration of the Bayesian
    workflow of Gelman et al. (2020): a question, a generative model, checks before and after fitting,
    and a conclusion stated on the scale the question was asked in.</p>
    {workflow(ctx)}
  </div>
</section>

<section class="statement on-dark bg-black">
  <div class="wrap">
    <p>Every failure is kept in: divergent chains, miscalibrated intervals, wrong answers.</p>
  </div>
</section>

<section class="group bg-mist on-light" id="paths">
  <div class="wrap">
    <h2 class="group-title">Three ways in</h2>
    <div class="tiles">
      <div class="tile bg-white on-light">
        <div class="tile-copy">
          <h3>New to Bayesian modelling</h3>
          <p class="tile-desc">Read the foundations in order. They assume some Python and some regression,
          nothing more.</p>
          <ol>
            <li>{L('E01', 'The Bayesian workflow')}</li>
            <li>{L('E02', 'Hierarchical models')}</li>
            <li>{L('E03', 'GLMs and model comparison')}</li>
            <li>{L('D03', 'An analysis end to end')}</li>
          </ol>
        </div>
      </div>
      <div class="tile on-dark bg-black">
        <div class="tile-copy">
          <h3>A practitioner with a problem</h3>
          <p class="tile-desc">Start from your data, not from a method. The guide maps outcome types, dependence
          structures and questions to the example that treats them.</p>
          <p class="actions"><a class="link" href="guide.html">Choose a model</a></p>
        </div>
        {TREE_FIG}
      </div>
      <div class="tile bg-white on-light">
        <div class="tile-copy">
          <h3>Ready to test yourself</h3>
          <p class="tile-desc">{n_ch} open-ended problems on real data. The brief says what to deliver, never how;
          the first model you try will break, and finding out why is the exercise.</p>
          <p class="actions"><a class="link" href="challenges.html">See the challenges</a></p>
        </div>
        {LADDER_FIG}
      </div>
    </div>
  </div>
</section>

<section class="group bg-white on-light" id="recent">
  <div class="wrap">
    <h2 class="group-title">Recently added</h2>
    <p class="group-lede">{recent_intro(ctx, recent_list)}</p>
    <div class="rail" data-rail><div class="cards rail-track">{recent}</div></div>
  </div>
</section>

<section class="group bg-mist on-light" id="featured">
  <div class="wrap">
    <h2 class="group-title">A few of the case studies</h2>
    <div class="cards">{featured}</div>
    <p class="group-more"><a href="examples.html">All {n_ex} examples</a></p>
  </div>
</section>

<section class="closing on-dark bg-black">
  <div class="wrap">
    <h2>Start with one question.</h2>
    <p class="closing-lede">The first example walks through the whole workflow once. Every page after it reuses
    the same steps.</p>
    <p class="actions">
      <a class="btn" href="{first}">Open the first example</a>
      <a class="link" href="examples.html">Browse all {n_ex} examples</a>
    </p>
  </div>
</section>
"""
    return (SITE_NAME, "home", body, TAGLINE)


def examples_page(ctx):
    body = f"""
<header class="page-head on-dark bg-black backdrop-archive">
  <div class="wrap">
    <h1>Worked examples</h1>
    <p class="lede">Each example is a narrated, executed notebook: the question, the data, the model written
    in mathematics and in PyMC, every check, and what the result means. Read them in the browser, then run
    and modify them locally.</p>
  </div>
</header>
<section class="group bg-mist on-light">
  <div class="wrap">
    <h2 class="group-title">Browse by series</h2>
    <p class="group-lede">The foundations come first; after that, each series stands on its own.</p>
    {ctx['series_index']}
  </div>
</section>
<section class="group bg-white on-light">
  <div class="wrap">
    <div class="filter" data-filter role="search" hidden>
      <input type="search" aria-label="Filter examples" placeholder="Filter by topic, data or technique"
             autocomplete="off" spellcheck="false">
      <p class="filter-count" aria-live="polite"></p>
    </div>
    {ctx['series_sections']}
  </div>
</section>
"""
    return (f"Examples — {SITE_NAME}", "examples", body,
            "Worked, executed examples of Bayesian models in PyMC on real data.")


def challenges_page(ctx):
    body = f"""
<header class="page-head on-dark bg-black backdrop-rock">
  <div class="wrap">
    <h1>Challenges</h1>
    <p class="lede">{len(ctx['challenges'])} problems on real data, each ending in a decision someone actually has
    to make. In every one, the obvious first model breaks in a way you have to diagnose. The brief tells you what
    to deliver, never how.</p>
  </div>
</header>

<section class="essay">
  <h2 id="how">How a challenge works</h2>
  <div class="features">
    <div><h3>A brief and a set of tasks</h3><p>Each challenge page describes the setting, the data and a
    sequence of tasks with concrete deliverables &mdash; a plot, a number, a recommendation. Download the
    notebook and work in the empty cells.</p></div>
    <div><h3>A ladder of hints</h3><p>Every task has three hints, revealed one at a time: a <em>nudge</em>
    (where to look), an <em>approach</em> (the modelling idea and the API) and a <em>skeleton</em> (code with
    the key parts blank). A good target is to finish on nudges alone.</p></div>
    <div><h3>Checks and a solution</h3><p>Numeric self-checks with generous ranges tell you whether your
    model is in the right place. A reference solution, fully executed and discussed, is there for when you
    have finished &mdash; or are truly stuck.</p></div>
  </div>
  <p class="narrow">They are roughly in order of difficulty but independent, so pick the one closest to your
  own work. The descriptions are deliberately vague about <em>what</em> goes wrong: finding out is the challenge.</p>
</section>

<section class="essay">
  <h2 id="list">The challenges</h2>
  {ctx['challenge_table']}
</section>

<section class="essay">
  <h2 id="local">Working locally</h2>
  <p class="narrow">In the notebook, a helper object tracks your hints and checks:</p>
  <div class="code"><pre><code>h.tasks()                       # the tasks, and how many hints / checks each has
h.hint("task2")                 # reveal the next hint for task 2
h.check("task2", sigma_angle_deg=1.5)   # compare your number with the reference
h.progress()                    # hints used, checks passed</code></pre></div>
  <p class="narrow">See <a href="about.html#run">About</a> for setting up the environment.</p>
</section>
"""
    return (f"Challenges — {SITE_NAME}", "challenges", _bands(body),
            "Real-data Bayesian modelling challenges with tiered hints and reference solutions.")


# The small drawing beside each option of the guide's model finder: the first pattern found in the
# row's first cell wins, and a row that matches none gets its table's default. The drawings
# themselves are GLYPHS in assets/figures.js.
GLYPH_RULES = [
    ("binary", r"yes/no|0/1|two-way"),
    ("ordinal", r"ordered categories|number of steps"),
    ("survival", r"time until|waiting times"),
    ("tail", r"largest value|jackpot|odds of an event"),
    ("compass", r"directions"),
    ("network", r"network of ties|causal graph"),
    ("ranking", r"who beat|variant wins|few matter"),
    ("map", r"\bmap\b|points in space|spread and react|actuators and sensors"),
    ("parts", r"shares of a whole|partition|subgroups|many features per item"),
    ("image", r"image|colour|spectr|how many sources"),
    ("tree", r"in groups|lineages|patients who differ"),
    ("series", r"series in time|still arriving|chaotic|jumps|trajectories|trigger more events|spiking|move together"),
    ("gaps", r"missing|missed|known error|measured with error|disagree|unrepresentative|published summaries|"
             r"polls in some|cannot answer"),
    ("dag", r"without the intervention|mediator|policy changed|cause the outcome"),
    ("fork", r"choice among|purchases|which experiment|what should we do|worth its price|whether to buy|better alarm"),
    ("counts", r"counts|a sum over"),
    ("curve", r"equation|smooth surface|non-linear|response curve|inverse problem|simulate|likelihood fits|centile|"
              r"compartments"),
    ("bell", r"continuous|error bars"),
]


def _glyph(situation: str, default: str) -> str:
    text = situation.lower()
    return next((name for name, pattern in GLYPH_RULES if re.search(pattern, text)), default)


def guide_page(ctx):
    L = lambda i, t=None: _link(ctx, i, t)  # noqa: E731
    n_ex, n_ch = len(ctx["examples"]), len(ctx["challenges"])

    rows = list  # each table is a list of (situation, what to consider, worked in)

    outcomes = rows([
        ("A continuous measurement", "Normal; Student-t if there are outliers; skew-normal or a distributional "
         "model if the spread or shape changes", f"{L('E01')}, {L('E41')}, {L('E43')}"),
        ("Yes/no, or successes out of trials", "Bernoulli / Binomial with a logit link", f"{L('E03')}, {L('C01')}"),
        ("Counts", "Poisson, or negative binomial when counts are over-dispersed", f"{L('E04')}, {L('E33')}, {L('E18')}"),
        ("Ordered categories (Likert scales)", "Ordered logistic, with cut-points as parameters", f"{L('C05')}"),
        ("Time until an event, some not yet observed", "Weibull or other survival models with <code>pm.Censored</code>",
         f"{L('E06')}, {L('C04')}, {L('E25')}"),
        ("The largest value in each year or block", "Generalised extreme value distribution", f"{L('E37')}"),
        ("Points in space and time", "A (log-Gaussian) Cox process", f"{L('E07')}, {L('E86')}"),
        ("A partition or a vocabulary", "Chinese restaurant / Pitman–Yor process", f"{L('E30')}"),
        ("A network of ties", "Latent-space and stochastic block models", f"{L('E21')}"),
        ("Who beat whom, or a whole finishing order", "Bradley-Terry, dynamic skills, Plackett-Luce", f"{L('E67')}"),
        ("A choice among alternatives", "Conditional and mixed logit; willingness to pay", f"{L('E68')}, {L('C15')}"),
        ("Repeated purchases: who competes with whom", "Factor-structured hierarchical logit (a choice map)",
         f"{L('E84')}, {L('C15')}"),
        ("A household x product 0/1 table (bought it or not)", "Bernoulli-logit latent factor model",
         f"{L('E85')}"),
        ("Directions: wind, angles, time of day", "Von Mises and circular densities, joint with speed",
         f"{L('E81')}, {L('C12')}"),
        ("Which experiment to run next", "Gaussian-process / bounded-yield surrogates and the value of information",
         f"{L('E82')}, {L('C13')}"),
        ("Where to measure next, on a map", "A marginal Gaussian process and the expected value of sample information",
         f"{L('E89')}, {L('E92')}"),
        ("Whether an exact model is worth its price, on a big map", "An exact GP against Vecchia and local windows, scored by region",
         f"{L('E92')}"),
        ("A sum over many homes on one shared day (load, delivery)", "A closed-form fleet, event means and a day-level model",
         f"{L('E91')}"),
        ("What a big model will cost, and whether to buy it", "Timed gradients, a price table and a held-out test of each model in a ladder",
         f"{L('E87')}"),
        ("Whether a trained network's error bars hold on real data", "Paired-frame tests, a population replicate and a calibration certificate",
         f"{L('E88')}"),
        ("Whether a better forecast makes a better alarm at a threshold", "Decision curves with a cluster bootstrap, lateness and release rules",
         f"{L('E90')}"),
        ("How many servers, agents or beds for random demand (a queue)", "Arrival, service and patience models feeding a queue simulator, and a staffing frontier",
         f"{L('E120')}"),
        ("A rare event to detect from a score, where a false alarm costs more than a miss (cheating, fraud, screening)",
         "Measure the score's spread on ordinary cases, put the base rate in explicitly, and draw the line on a probability with a cost ratio",
         f"{L('E121')}"),
        ("Vehicles on a line that bunch up (trajectories that interact)",
         "Segment-time and dwell models feeding a simulator of the whole line, replayed on unseen days, and a control rule judged in rider time",
         f"{L('E122')}"),
        ("Proportions of sources in a mixture, from tracers (diet from isotopes, sediment fingerprinting)",
         "A mixing model with source and discrimination uncertainty carried through, a prior flat on the question, and a test against a known mixture",
         f"{L('E123')}"),
        ("A centile, not the mean (growth charts, thresholds)", "Bayesian quantile regression with calibration checks",
         f"{L('E83')}, {L('C14')}"),
        ("A quick two-way choice and its response time", "Drift diffusion model (Wiener first-passage time)",
         f"{L('E71')}"),
        ("Shares of a whole, or counts split into parts", "Dirichlet, logistic-normal and Dirichlet-multinomial models",
         f"{L('E70')}"),
        ("Counts of many features per item (words, colours, pages, products)", "Topic models (LDA) and covariate topic models",
         f"{L('E72')}, {L('E73')}"),
        ("Colours or spectra that change with exposure", "Kubelka-Munk mixing, fading kinetics rendered through the "
         "CIE observer", f"{L('E74')}"),
        ("A spectrum from a colour measurement", "Bayesian inversion with a learned spectral prior; metamers",
         f"{L('E76')}"),
        ("Regions of an image, number unknown", "Dirichlet-process mixture with a Potts spatial prior", f"{L('E75')}"),
        ("Names people give to colours", "Softmax categories over CIELAB with speaker-level effects", f"{L('E77')}"),
    ])
    structure = rows([
        ("Observations in groups (schools, counties, markets)", "Hierarchical model with partial pooling",
         f"{L('E02')}, {L('E33')}, {L('E36')}"),
        ("Neighbouring regions on a map", "ICAR / BYM2 areal models", f"{L('E18')}"),
        ("A smooth surface or curve", "Gaussian process (HSGP for speed), splines", f"{L('E05')}, {L('E07')}, {L('C08')}"),
        ("A series in time", "State-space models, random walks, hidden Markov models", f"{L('E13')}, {L('E24')}, {L('C07')}"),
        ("Related species or lineages", "Phylogenetic covariance", f"{L('E23')}"),
        ("Many outcomes that move together", "Full covariance, factor models, copulas", f"{L('E22')}"),
        ("Unobserved subgroups", "Mixtures and Dirichlet-process mixtures", f"{L('C06')}, {L('E26')}, {L('C11')}"),
    ])
    imperfect = rows([
        ("Values are missing", "Model the missingness jointly instead of dropping rows", f"{L('D02')}"),
        ("Measurements carry known error", "Measurement-error model; the error is data", f"{L('D02')}, {L('E17')}"),
        ("Species or cases can be missed", "Occupancy, N-mixture and capture–recapture models", f"{L('E28')}"),
        ("Raters disagree and there is no gold standard", "Latent class and Dawid-Skene models", f"{L('E69')}"),
        ("Two files with no shared key: which records are the same thing?",
         "An agreement-pattern mixture, a posterior over one-to-one linkages, and a regression that sums over candidate links",
         f"{L('E124')}"),
        ("Recent data are still arriving", "Nowcasting the reporting delay", f"{L('E38')}, {L('E90')}, {L('E94')}"),
        ("Polls in some races, none in most", "Fundamentals learned on earlier cycles, polls where they exist, a shared error",
         f"{L('E40')}, {L('E94')}"),
        ("The survey sample is unrepresentative", "Multilevel regression and poststratification", f"{L('E36')}"),
        ("Only published summaries, not individuals", "Meta-analysis on sufficient statistics", f"{L('E17')}"),
    ])
    form = rows([
        ("Theory gives the equation", "Put the physics in the mean function", f"{L('E08')}, {L('C01')}"),
        ("The system follows a differential equation", "ODE inside the model, gradients via JAX", f"{L('E09')}, {L('E11')}"),
        ("You can simulate but not write a likelihood", "Simulation-based inference: ABC, SMC, synthetic likelihood", f"{L('E14')}"),
        ("Unknown non-linear effects of many inputs", "BART, Bayesian neural networks, GPs", f"{L('E15')}, {L('E29')}, {L('E05')}"),
        ("Many candidate predictors, few matter", "Horseshoe and spike-and-slab priors", f"{L('E27')}"),
        ("No standard likelihood fits the process", "Derive and test your own", f"{L('E32')}"),
        ("You do not know which shape a response curve has", "A flexible family (Weibull transform) written in JAX",
         f"{L('E45')}"),
        ("The dynamics are chaotic", "Latent states with process noise, features not paths, control under the posterior",
         f"{L('E46')}, {L('E47')}, {L('E48')}"),
        ("The process jumps as well as diffuses", "Jump-diffusion SDE with the jump count summed out; SV with jumps",
         f"{L('E49')}"),
        ("Things spread and react in space", "Reaction-diffusion PDE by the method of lines, or exactly via an eigenbasis",
         f"{L('E50')}"),
        ("Transport runs in two coupled compartments (surface and bulk)",
         "Bulk-surface PDE solved exactly per Fourier mode; compare with the 1-D shortcut models",
         f"{L('E52')}"),
        ("Counts have a heavy jackpot tail with no closed-form likelihood",
         "Compound-Poisson pmf by a triangular solve inside PyTensor; censored top class",
         f"{L('E53')}"),
        ("A stiff, spiking simulator whose trace likelihood is rugged",
         "Fit summary features with a simulator likelihood (SMC); check sloppy parameter directions",
         f"{L('E54')}"),
        ("Counts come from a stochastic reaction network",
         "Chemical master equation by finite state projection; ask what snapshots identify",
         f"{L('E55')}"),
        ("A predictor is measured with error (regression dilution)",
         "Errors-in-variables with the noise identified from the data's own correlation structure",
         f"{L('E56')}"),
        ("Trajectories blur motion with measurement error",
         "Exact track likelihood (MA(1) via a sine eigenbasis); state mixtures and switching HMMs",
         f"{L('E57')}"),
        ("How many sources are in the data, and where?",
         "Count as model selection (Laplace / SMC evidence) with a pixelated physical forward model",
         f"{L('E58')}"),
        ("Waiting times hide an unknown number of sub-steps",
         "Hypoexponential dwell models with ordered rates, LOO over the number of steps; lattice HMM for step detection",
         f"{L('E59')}"),
        ("An ill-posed inverse problem (deconvolution)",
         "Regularisation as a prior: marginal likelihood for its strength, GP spectra, sparse horseshoe priors",
         f"{L('E60')}"),
        ("A discrete structural parameter (number of steps) is weakly identified",
         "Marginalise it with logsumexp and parameterise by what the data identify",
         f"{L('E61')}, {L('E59')}"),
        ("Doses, concentrations and patients who differ",
         "Hierarchical compartment models; flip-flop checks with multi-start chains; Bayesian therapeutic drug monitoring",
         f"{L('E64')}"),
        ("Events trigger more events (aftershocks, cascades)",
         "Hawkes / ETAS likelihood with a closed-form compensator; incompleteness in the likelihood; cascade forecasts",
         f"{L('E65')}"),
        ("You must decide where to put actuators and sensors", "Design under the posterior: greedy placement, Monte Carlo closed-loop evaluation",
         f"{L('E51')}"),
    ])
    causal = rows([
        ("What would have happened without the intervention?", "g-computation with <code>pm.do</code>; synthetic control",
         f"{L('C09')}, {L('E34')}, {L('E15')}"),
        ("Does the effect travel through a mediator?", "Causal mediation", f"{L('E35')}"),
        ("What is the causal graph?", "Structure learning for graphs and DAGs", f"{L('E20')}, {L('E19')}"),
        ("How much did a driver change the odds of an event?", "Attribution with non-stationary extremes", f"{L('E37')}"),
        ("What should we do?", "Optimise expected utility over posterior draws", f"{L('E33')}, {L('C04')}, {L('C07')}"),
        ("A policy changed for some units, at a threshold, or through an instrument",
         "Difference-in-differences, regression discontinuity, IV; the identifying assumption as a prior",
         f"{L('E63')}"),
        ("Does spending cause the outcome, or follow it?", "Within-pair first differences (rematches) with hierarchical slopes",
         f"{L('E94')}"),
        ("Which variant wins, and when can I stop the test?",
         "Hierarchical effect sizes as the prior, expected-loss stopping, Thompson sampling",
         f"{L('E66')}"),
        ("The data cannot answer the question as asked",
         "Swap priors, run a fake-data check, then answer the decision or the nearest supported question",
         f"{L('E62')}"),
    ])

    def table(title, anchor, lead, items, step, default):
        """One question as a table. `step` and each row's data-g feed the model finder above the tables."""
        body = "".join(f'<tr data-g="{_glyph(a, default)}"><td>{a}</td><td>{b}</td><td>{c}</td></tr>'
                       for a, b, c in items)
        return (f'<section class="essay"><h2 id="{anchor}">{title}</h2><p class="narrow">{lead}</p>'
                f'<div class="table-wrap"><table class="guide-table" data-step="{step}"><thead><tr>'
                f"<th>If your data or question is…</th><th>Consider</th><th>Worked in</th></tr></thead>"
                f"<tbody>{body}</tbody></table></div></section>")

    body = f"""
<header class="page-head on-dark bg-black backdrop-compass">
  <div class="wrap">
  <h1>From your problem to a model</h1>
  <p class="lede">Bayesian models are built, not selected. The questions below are the ones to ask of
  your own data, in roughly the order they matter. Each row points to a worked example where that
  modelling decision is made, checked and, sometimes, got wrong first.</p>
  <button type="button" class="search-bar" data-search-open>{SEARCH_MARK}<span>Search {n_ex} examples and
  {n_ch} challenges</span><kbd>/</kbd></button>
  </div>
</header>

<section class="essay">
  <div data-needs-js>
    <h2 id="finder">Build a model from your answers</h2>
    <p class="narrow">Five questions, in the order they matter. Answer the ones you can: each answer adds a piece
    to the model and the worked examples that show it. The tables further down list the same options in full.</p>
    <div class="finder" data-finder></div>
  </div>
</section>

<section class="essay">
  <h2 id="recipe">A recipe for a first model</h2>
  <ol class="recipe narrow">
    <li><strong>State the estimand.</strong> Write the quantity the decision depends on before writing any
    model: a difference in expected outcomes, a probability of exceeding a threshold, a forecast for
    next week. Everything else is a means to computing it. ({L('D03')}, {L('E62')})</li>
    <li><strong>Identify the unit and the support of the outcome.</strong> One row per what? Is the outcome
    a count, a proportion, a positive quantity, a time with censoring? This chooses the likelihood.
    ({L('D01')})</li>
    <li><strong>Describe how observations are related.</strong> Groups, neighbours, time, a network. This
    chooses the prior structure, and is usually where most of the information is.</li>
    <li><strong>Write down how the data could be imperfect.</strong> Missing, censored, truncated,
    mismeasured, unrepresentative. Modelling the imperfection usually beats cleaning it away. ({L('D02')})</li>
    <li><strong>Choose priors on interpretable scales and simulate them.</strong> If the prior predictive
    distribution produces impossible data, the priors are not "weak", they are wrong. ({L('E01')})</li>
    <li><strong>Fit, then diagnose the computation.</strong> Divergences, R-hat and effective sample size
    are about whether the answer was computed correctly, not whether the model is right. ({L('E02')},
    {L('E12')})</li>
    <li><strong>Criticise the model.</strong> Posterior predictive checks aimed at the features the decision
    depends on, calibration of held-out predictions, and comparison of alternatives by LOO.
    ({L('E03')})</li>
    <li><strong>Report the estimand with its uncertainty,</strong> on the scale of the original question,
    and show it in a way the reader can act on. ({L('E16')})</li>
  </ol>
</section>
{table("1 · What kind of outcome?", "outcome", "The support and the data-generating process of the outcome choose the likelihood.", outcomes, "Outcome", "bell")}
{table("2 · How are the observations related?", "structure", "Dependence between observations is where hierarchical and structured priors earn their keep.", structure, "Structure", "tree")}
{table("3 · Is the data imperfect?", "imperfect", "Imperfections are part of the generative story, and modelling them is often the difference between a biased and an honest answer.", imperfect, "Imperfect data", "gaps")}
{table("4 · What form does the relationship take?", "form", "From a known equation to a flexible function, the choice is how much structure you are willing to assume.", form, "Form", "curve")}
{table("5 · Is the question causal, or a decision?", "causal", "Posterior draws propagate straight into counterfactuals and expected utilities &mdash; if the assumptions that make them causal hold.", causal, "Causal question", "dag")}
"""
    return (f"Choosing a model — {SITE_NAME}", "guide", _bands(body),
            "A practitioner's guide from the shape of your data to a Bayesian model, with worked examples.")


def about_page(ctx):
    repo = ctx.get("repo_url")
    clone = f"git clone {repo}.git\ncd {repo.rsplit('/', 1)[-1]}\n" if repo else ""
    body = f"""
<header class="page-head on-dark bg-black backdrop-notebook">
  <div class="wrap">
    <h1>About these notes</h1>
    <p class="lede">A self-study curriculum in applied Bayesian statistics, written as executable notebooks
    for the current PyMC stack and published here as readable pages.</p>
  </div>
</header>
<section class="essay prose narrow">
  <h2 id="principles">Principles</h2>
  <ul>
    <li><strong>Real data first.</strong> Every dataset is public, with its source cited where it is loaded. A few
    examples study systems for which no public data exist - chaos control in Lorenz models, the design of a
    heated plate, a morphogen gradient - and simulate them from published models and parameters; those pages
    say so in their header.</li>
    <li><strong>Honest results.</strong> Where a model under-covers, fails to mix or gives an inconclusive answer,
    the page says so. Several examples end with a negative result on purpose.</li>
    <li><strong>Current tools.</strong> PyMC 6, ArviZ 1 and PyTensor 3, whose APIs differ from most tutorials
    (posteriors are xarray <code>DataTree</code>s; nutpie is the default sampler when installed). {_link(ctx, 'E01')}
    has a then-versus-now table.</li>
    <li><strong>Decisions, not just posteriors.</strong> Most analyses end with the quantity a reader acts on.</li>
  </ul>
  <h2 id="run">Running the notebooks</h2>
  <p>The environment is managed with <a href="https://docs.astral.sh/uv/">uv</a>:</p>
  <div class="code"><pre><code>{escape(clone)}uv sync                            # PyMC 6, ArviZ 1, nutpie, JAX, JupyterLab, ...
uv run python -m pymc_challenges    # optional: download every dataset into data/
uv run jupyter lab notebooks</code></pre></div>
  <p>Every notebook has been measured to peak below 3 GB of memory. Multi-process samplers copy the model per
  chain; on a small machine prefer nutpie (threads) and keep to one notebook at a time.</p>
  <h2 id="reading">Reading the pages</h2>
  <p>Pages are rendered from the executed notebooks without re-running them, so every number and figure is the
  one the code produced. Use <em>Hide code</em> at the top of a page to read the narrative and figures alone;
  the choice is remembered. Mathematics is typeset with MathJax.</p>
  <h2 id="references">Further reading</h2>
  <ul>
    <li>Gelman, Vehtari, Simpson et al. (2020). <em>Bayesian Workflow.</em> arXiv:2011.01808.</li>
    <li>Gelman, Carlin, Stern, Dunson, Vehtari &amp; Rubin (2013). <em>Bayesian Data Analysis</em>, 3rd ed.</li>
    <li>McElreath (2020). <em>Statistical Rethinking</em>, 2nd ed.</li>
    <li>Martin, Kumar &amp; Lao (2021). <em>Bayesian Modeling and Computation in Python.</em></li>
    <li>The <a href="https://www.pymc.io/">PyMC</a> and <a href="https://python.arviz.org/">ArviZ</a> documentation.</li>
  </ul>
</section>
"""
    return (f"About — {SITE_NAME}", "about", _bands(body),
            "About this Bayesian modelling curriculum and how to run it.")


def pages(ctx):
    return {
        "index.html": home(ctx),
        "examples.html": examples_page(ctx),
        "challenges.html": challenges_page(ctx),
        "guide.html": guide_page(ctx),
        "about.html": about_page(ctx),
    }
