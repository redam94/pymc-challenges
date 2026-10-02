"""Hand-written copy for the website built by tools/build_site.py.

Per-notebook titles, datasets and techniques come from the README tables; this file holds
the series grouping and the prose of the index, guide, challenges and about pages.
"""

from html import escape

SITE_NAME = "Bayesian Modelling in Practice"
TAGLINE = ("Worked examples and real-data challenges in Bayesian data analysis with PyMC: "
           "from the first prior predictive check to decisions made under uncertainty.")
FOOTER = ("Worked examples and challenges in Bayesian data analysis with PyMC 6 · ArviZ 1 · PyTensor 3. "
          "Datasets are real and public, sources cited in each notebook; the few simulated systems say so.")


def _ids(prefix, a, b):
    return [f"{prefix}{i:02d}" for i in range(a, b + 1)]


SERIES = [
    dict(key="foundations", range="E01–E06", title="Foundations", ids=_ids("E", 1, 6),
         intro="The core of applied Bayesian work: writing a model down, checking what the priors "
               "claim, reading a posterior, pooling information across groups, comparing models "
               "and handling censored data. Every challenge assumes these."),
    dict(key="data-work", range="D01–D03", title="Data work", ids=_ids("D", 1, 3),
         intro="Most models that refuse to sample are data problems in disguise. These cover the craft "
               "around the model: preparing arrays, treating missingness and measurement error as part "
               "of the model, and carrying an analysis through to a report."),
    dict(key="science", range="E07–E10, E46–E51, E78–E80", title="Mechanistic and scientific models",
         ids=_ids("E", 7, 10) + _ids("E", 46, 51) + _ids("E", 78, 80),
         intro="When theory supplies the regression function. Earthquakes as a point process in space "
               "and time, the expansion of the universe from supernovae, differential equations with "
               "JAX gradients, a black-hole ringdown in real LIGO strain, chaotic dynamics (fitting, forecasting "
               "and controlling systems that amplify every error, from blowflies to Lorenz's toy atmospheres), "
               "jump-diffusion SDEs for market crashes, and reaction-diffusion PDEs for cell invasion and "
               "signaling gradients, designing heaters, insulation and sensors for a heat-equation plate, and "
               "finding planets in Kepler light curves and proving an interstellar comet is unbound; and cosmology "
               "from whole images: every pixel of a Planck CMB patch, and a Hubble Einstein ring modelled pixel by pixel."),
    dict(key="biology", range="E52–E61", title="Cell biology and biophysics", ids=_ids("E", 52, 61),
         intro="Mechanistic models of living cells, fitted to what biologists measure: signals that travel "
               "through the membrane and the cytosol, the fluctuation test that showed mutations arise before "
               "selection, why a neuron fires (conductance models fitted to real patch-clamp recordings), "
               "transcriptional bursting read from allele-resolved single-cell RNA counts, and how bacteria "
               "control their size, measured through noisy segmentation, and single molecules tracked in "
               "living nuclei, and super-resolution (STORM) microscopy: localising, detecting and counting "
               "single blinking molecules below the diffraction limit, and kinesin's steps counted from "
               "MINFLUX traces, the forces cells exert on their substrate (traction force microscopy), and "
               "decisions from dynamics: Turing patterns and kinetic proofreading in T cells. See also E50 (reaction-diffusion in tissues)."),
    dict(key="failure", range="E62", title="When the model cannot answer", ids=["E62"],
         intro="Most failed analyses do not crash: they answer a question nobody asked, report numbers the "
               "prior chose, or never finish. Three real problems fail in these ways - an MMM that cannot "
               "split credit between channels, a Gaussian process too large to fit, a trial whose hazard "
               "ratio is not the clinic's question - and each is followed by the pivot: a question the data "
               "can answer, a cheaper model checked against the exact one, and an answer audit."),
    dict(key="decisions", range="E63–E66", title="Causal designs, doses, cascades and experiments",
         ids=_ids("E", 63, 66),
         intro="Workhorse models from four fields, each carried through to a decision. Difference-in-"
               "differences, regression discontinuity and instrumental variables on classic natural "
               "experiments, with each identifying assumption written as a prior; population "
               "pharmacokinetics from theophylline and warfarin to a patient's dose after two monitoring "
               "samples; Hawkes and ETAS aftershock forecasts for the 2019 Ridgecrest sequence; and "
               "thousands of real headline A/B tests: winner's curse, peeking, and bandits."),
    dict(key="outcomes", range="E67–E73", title="Rankings, choices, raters, compositions, topics and decisions",
         ids=_ids("E", 67, 73),
         intro="Outcomes that are not one number on a line. Skills estimated from who beat whom in tennis and "
               "from whole Formula 1 finishing orders; what people will pay for, from a panel of stated "
               "choices, with a market simulator; the true label when pathologists or anaesthetists "
               "disagree and nobody knows the answer; data that are parts of a whole; and quick decisions, "
               "where a drift diffusion model explains both the choice and how long it took, and says "
               "how cautious a person should be; and topic models for anything that comes as a bag of "
               "features: the stories behind Upworthy's headlines, the palettes of a century of posters, the "
               "intents behind web visits and the missions behind shopping baskets - and the newer topic models "
               "built for LDA's problems, tested on the same data."),
    dict(key="colour", range="E74–E77", title="Colour, light and images", ids=_ids("E", 74, 77),
         intro="When the estimate is a colour, a spectrum, a region or a word. How the dyes of Japanese woodblock "
               "prints fade, and what a 1766 Harunobu print looked like when new; why two paints can match in "
               "the shop and differ at home (metamers), inverted from colour to spectrum with a prior learned "
               "from 1,269 Munsell chips; segmenting photographs with an unknown number of regions and checking "
               "the result, and its confidence, against the people who outlined them; and how 110 languages "
               "divide colour space, speaker by speaker - each with uncertainty displays that still work when "
               "colour cannot be used to show the uncertainty."),
    dict(key="gaps", range="E81–E83", title="Directions, experiments and quantiles", ids=_ids("E", 81, 83),
         intro="Three model families that need their own tricks. Angles wrap around, so means, priors and links "
               "that work on a line break on a circle (wind at Santa Barbara, ocelots and their prey); Bayesian "
               "optimisation chooses the next reaction on a fully measured chemistry screen and is scored "
               "against 50 real chemists; and quantile regression for birth weights shows why the asymmetric "
               "Laplace posterior is too sure of itself, and how to fix it."),
    dict(key="frontier", range="E11–E17", title="At the research frontier", ids=_ids("E", 11, 17),
         intro="Neural networks inside differential equations, inference beyond plain NUTS, state-space "
               "models, likelihood-free inference, BART, the display of uncertainty and privacy-preserving "
               "meta-analysis. Expect honest negative results next to the successes."),
    dict(key="graphs", range="E18–E23", title="Correlation structures and graphs", ids=_ids("E", 18, 23),
         intro="Dependence that lives on a map, a network or a tree: areal models, learning graphs and "
               "causal structure from data, network outcomes, many correlated outcomes at once, and "
               "regression along a phylogeny."),
    dict(key="families", range="E24–E29", title="More model families", ids=_ids("E", 24, 29),
         intro="Hidden Markov models, joint longitudinal and survival models, Dirichlet-process mixtures, "
               "sparse regression, imperfect detection in ecology and Bayesian neural networks."),
    dict(key="partitions", range="E30–E31", title="Random partitions and random features", ids=_ids("E", 30, 31),
         intro="Exchangeable partitions and latent binary features: Pólya urns, Pitman–Yor processes and "
               "unseen species, and the Indian buffet process."),
    dict(key="media", range="E32–E35, E45, E84–E85", title="Media measurement and market structure",
         ids=_ids("E", 32, 35) + ["E45", "E84", "E85"],
         intro="One open multi-brand e-commerce dataset and five questions a marketing team actually asks: "
               "a custom likelihood for spend that chases demand, a hierarchical marketing-mix model with a "
               "budget decision, a synthetic control, causal mediation, and which shape of response curve "
               "the data support (a Weibull transform written in JAX). Plus market structure from household "
               "panels: which margarines compete, read off a Bayesian choice map, and what that means for a "
               "merger screen; and the same question from the 6-month purchase-incidence tables and partial "
               "demographics most studies actually get."),
    dict(key="everyone", range="E36–E41", title="State of the art, explained for everyone", ids=_ids("E", 36, 41),
         intro="Six model families used in practice today, each with a plain-language opening and a closing "
               "section of uncertainty displays for readers with no statistics: MRP, extreme-event attribution, "
               "epidemic nowcasting, radiocarbon chronologies, election forecasting and growth charts."),
    dict(key="xarray", range="E42–E44", title="Posteriors as labelled arrays", ids=_ids("E", 42, 44),
         intro="A posterior is a labelled array with named chain, draw and model dimensions. These notebooks do "
               "all their data and posterior work by name, and use it for derived quantities, time "
               "aggregation, binning and unusual plots."),
]
SERIES_BY_KEY = {s["key"]: s for s in SERIES}
RECENT = ["E85", "E84", "E83", "E82", "E81", "E80", "E79"]          # newest first, shown on the home page


def _link(ctx, eid, text=None):
    e = ctx["by_id"].get(eid)
    if not e:
        return escape(text or eid)
    return f'<a href="{e.url}">{escape(text) if text else eid}</a>'


# --------------------------------------------------------------------------- workflow figure

WORKFLOW_STEPS = [
    ("Question", "What decision or claim will the model inform?", "E01"),
    ("Generative model", "A story of how the data came to be, written as distributions.", "E01"),
    ("Prior predictive", "Simulate before fitting: do the priors claim anything absurd?", "D03"),
    ("Inference", "Sample the posterior, or approximate it with care.", "E12"),
    ("Diagnostics", "Did the computation work? Divergences, R-hat, effective sample size.", "E02"),
    ("Posterior predictive", "Can the fitted model reproduce the features of the data you care about?", "E03"),
    ("Decision", "Carry the whole posterior through to the quantity someone acts on.", "E33"),
]


def workflow_svg() -> str:
    """A loop of seven steps with a return arrow from the checks back to the model."""
    n, w, gap, h, y = len(WORKFLOW_STEPS), 128, 16, 58, 26
    total = n * w + (n - 1) * gap
    parts = [f'<svg class="workflow" viewBox="0 0 {total + 4} 150" role="img" '
             f'aria-label="The Bayesian workflow: question, generative model, prior predictive check, inference, '
             f'diagnostics, posterior predictive check, decision - with a loop back to revise the model.">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>']
    for i, (name, _, _) in enumerate(WORKFLOW_STEPS):
        x = 2 + i * (w + gap)
        cls = "wf-box wf-check" if name in ("Prior predictive", "Diagnostics", "Posterior predictive") else "wf-box"
        parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>')
        words = name.split(" ")
        lines = [" ".join(words)] if len(name) < 15 else [words[0], " ".join(words[1:])]
        for j, line in enumerate(lines):
            ty = y + h / 2 + (j - (len(lines) - 1) / 2) * 17 + 5
            parts.append(f'<text class="wf-label" x="{x + w / 2}" y="{ty}" text-anchor="middle">{line}</text>')
        if i < n - 1:
            parts.append(f'<line class="wf-arrow" x1="{x + w + 2}" y1="{y + h / 2}" x2="{x + w + gap - 2}" '
                         f'y2="{y + h / 2}" marker-end="url(#arr)"/>')
    # revise loop: from posterior predictive (index 5) back to generative model (index 1)
    x_from = 2 + 5 * (w + gap) + w / 2
    x_to = 2 + 1 * (w + gap) + w / 2
    yb = y + h
    parts.append(f'<path class="wf-loop" d="M{x_from} {yb + 2} V{yb + 38} H{x_to} V{yb + 6}" marker-end="url(#arr)"/>')
    parts.append(f'<text class="wf-note" x="{(x_from + x_to) / 2}" y="{yb + 56}" text-anchor="middle">'
                 f"a check fails → revise the model, and say so</text>")
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- pages


def home(ctx):
    n_ex, n_ch = len(ctx["examples"]), len(ctx["challenges"])
    L = lambda i, t=None: _link(ctx, i, t)  # noqa: E731
    steps = "".join(
        f"<li><strong>{name}.</strong> {desc} <span class=\"muted\">See {L(eid)}.</span></li>"
        for name, desc, eid in WORKFLOW_STEPS
    )
    featured = "".join(ctx["card"](ctx["by_id"][i]) for i in ("E10", "E37", "E40", "E16") if i in ctx["by_id"])
    recent = "".join(ctx["card"](ctx["by_id"][i]) for i in RECENT if i in ctx["by_id"])
    body = f"""
<section class="hero">
  <p class="eyebrow">A course in applied Bayesian statistics, built from real data</p>
  <h1>Learn Bayesian data analysis the way it is actually done: one real problem at a time.</h1>
  <p class="lede">{n_ex} worked examples and {n_ch} challenges, each a complete analysis in PyMC, almost all
  of public datasets. Every page shows the model, the code, the checks that passed and the ones that did not, and
  what the posterior means for the question that was asked.</p>
  <div class="hero-actions">
    <a class="button" href="{ctx['by_id']['E01'].url}">Start with the workflow</a>
    <a class="button button-quiet" href="guide.html">Find a model for your problem</a>
    <a class="button button-quiet" href="challenges.html">Test yourself</a>
  </div>
</section>

<section class="facts" aria-label="At a glance">
  <div><strong>{n_ex}</strong><span>worked examples, executed and narrated</span></div>
  <div><strong>{n_ch}</strong><span>challenges with tiered hints and reference solutions</span></div>
  <div><strong>Real</strong><span>public datasets, from supernovae to election polls; the few simulated systems are labelled</span></div>
  <div><strong>Every</strong><span>failure kept in: divergent chains, miscalibrated intervals, wrong answers</span></div>
</section>

<section class="essay">
  <h2 id="why">Why model this way</h2>
  <div class="three">
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
</section>

<section class="essay">
  <h2 id="workflow">The workflow every page follows</h2>
  <p class="narrow">The examples are not a tour of an API. Each one is a complete iteration of the Bayesian
  workflow of Gelman et al. (2020): a question, a generative model, checks before and after fitting,
  and a conclusion stated on the scale the question was asked in.</p>
  <figure class="figure-wide">{workflow_svg()}</figure>
  <ol class="workflow-list">{steps}</ol>
</section>

<section class="essay">
  <h2 id="paths">Three ways in</h2>
  <div class="paths">
    <div class="path">
      <h3>New to Bayesian modelling</h3>
      <p>Read the foundations in order. They assume some Python and some regression, nothing more.</p>
      <ol>
        <li>{L('E01', 'E01 · The Bayesian workflow')}</li>
        <li>{L('E02', 'E02 · Hierarchical models')}</li>
        <li>{L('E03', 'E03 · GLMs and model comparison')}</li>
        <li>{L('D03', 'D03 · An analysis end to end')}</li>
      </ol>
    </div>
    <div class="path">
      <h3>A practitioner with a problem</h3>
      <p>Start from your data, not from a method. The guide maps outcome types, dependence structures and
      questions to the example that treats them.</p>
      <p><a href="guide.html">Choosing a model →</a></p>
    </div>
    <div class="path">
      <h3>Ready to test yourself</h3>
      <p>Eleven open-ended problems on real data. The brief says what to deliver, never how; the first
      model you try will break, and finding out why is the exercise.</p>
      <p><a href="challenges.html">The challenges →</a></p>
    </div>
  </div>
</section>

<section class="essay">
  <h2 id="recent">Recently added</h2>
  <p class="narrow">Chaos and how to fit, forecast and control it; regime detection with hidden Markov models;
  jump-diffusion SDEs for market crashes; reaction-diffusion in cell biology; optimal placement of heaters and
  sensors for the heat equation; and a flexible response curve for media models, written in JAX.</p>
  <div class="cards">{recent}</div>
</section>

<section class="essay">
  <h2 id="featured">A few of the case studies</h2>
  <div class="cards">{featured}</div>
  <p><a href="examples.html">All {n_ex} examples →</a></p>
</section>
"""
    return (SITE_NAME, "home", body, TAGLINE)


def examples_page(ctx):
    jump = "".join(f'<li><a href="#{s["key"]}">{s["title"]}</a> <span class="muted">{s["range"]}</span></li>'
                   for s in SERIES)
    body = f"""
<header class="page-head">
  <p class="eyebrow">Examples</p>
  <h1>Worked examples</h1>
  <p class="lede">Each example is a narrated, executed notebook: the question, the data, the model written
  in mathematics and in PyMC, every check, and what the result means. Read them in the browser, then run
  and modify them locally. The foundations come first; after that, each series stands on its own.</p>
</header>
<nav class="jump" aria-label="Series"><ol>{jump}</ol></nav>
{ctx['series_sections']}
"""
    return (f"Examples — {SITE_NAME}", "examples", body,
            "Worked, executed examples of Bayesian models in PyMC on real data.")


def challenges_page(ctx):
    body = f"""
<header class="page-head">
  <p class="eyebrow">Challenges</p>
  <h1>Challenges</h1>
  <p class="lede">Eleven problems on real data, each ending in a decision someone actually has to make. In every
  one, the obvious first model breaks in a way you have to diagnose. The brief tells you what to deliver,
  never how.</p>
</header>

<section class="essay">
  <h2 id="how">How a challenge works</h2>
  <div class="three">
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
    return (f"Challenges — {SITE_NAME}", "challenges", body,
            "Real-data Bayesian modelling challenges with tiered hints and reference solutions.")


def guide_page(ctx):
    L = lambda i, t=None: _link(ctx, i, t)  # noqa: E731

    def rows(items):
        return "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in items)

    outcomes = rows([
        ("A continuous measurement", "Normal; Student-t if there are outliers; skew-normal or a distributional "
         "model if the spread or shape changes", f"{L('E01')}, {L('E41')}, {L('E43')}"),
        ("Yes/no, or successes out of trials", "Bernoulli / Binomial with a logit link", f"{L('E03')}, {L('C01')}"),
        ("Counts", "Poisson, or negative binomial when counts are over-dispersed", f"{L('E04')}, {L('E33')}, {L('E18')}"),
        ("Ordered categories (Likert scales)", "Ordered logistic, with cut-points as parameters", f"{L('C05')}"),
        ("Time until an event, some not yet observed", "Weibull or other survival models with <code>pm.Censored</code>",
         f"{L('E06')}, {L('C04')}, {L('E25')}"),
        ("The largest value in each year or block", "Generalised extreme value distribution", f"{L('E37')}"),
        ("Points in space and time", "A (log-Gaussian) Cox process", f"{L('E07')}"),
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
        ("Recent data are still arriving", "Nowcasting the reporting delay", f"{L('E38')}"),
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
        ("Which variant wins, and when can I stop the test?",
         "Hierarchical effect sizes as the prior, expected-loss stopping, Thompson sampling",
         f"{L('E66')}"),
        ("The data cannot answer the question as asked",
         "Swap priors, run a fake-data check, then answer the decision or the nearest supported question",
         f"{L('E62')}"),
    ])

    def table(title, anchor, lead, body):
        return (f'<section class="essay"><h2 id="{anchor}">{title}</h2><p class="narrow">{lead}</p>'
                f'<div class="table-wrap"><table class="guide-table"><thead><tr><th>If your data or question is…</th>'
                f"<th>Consider</th><th>Worked in</th></tr></thead><tbody>{body}</tbody></table></div></section>")

    body = f"""
<header class="page-head">
  <p class="eyebrow">Guide</p>
  <h1>From your problem to a model</h1>
  <p class="lede">Bayesian models are built, not selected. The questions below are the ones to ask of
  your own data, in roughly the order they matter. Each row points to a worked example where that
  modelling decision is made, checked and, sometimes, got wrong first.</p>
</header>

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
{table("1 · What kind of outcome?", "outcome", "The support and the data-generating process of the outcome choose the likelihood.", outcomes)}
{table("2 · How are the observations related?", "structure", "Dependence between observations is where hierarchical and structured priors earn their keep.", structure)}
{table("3 · Is the data imperfect?", "imperfect", "Imperfections are part of the generative story, and modelling them is often the difference between a biased and an honest answer.", imperfect)}
{table("4 · What form does the relationship take?", "form", "From a known equation to a flexible function, the choice is how much structure you are willing to assume.", form)}
{table("5 · Is the question causal, or a decision?", "causal", "Posterior draws propagate straight into counterfactuals and expected utilities &mdash; if the assumptions that make them causal hold.", causal)}
"""
    return (f"Choosing a model — {SITE_NAME}", "guide", body,
            "A practitioner's guide from the shape of your data to a Bayesian model, with worked examples.")


def about_page(ctx):
    repo = ctx.get("repo_url")
    clone = f"git clone {repo}.git\ncd {repo.rsplit('/', 1)[-1]}\n" if repo else ""
    body = f"""
<header class="page-head">
  <p class="eyebrow">About</p>
  <h1>About these notes</h1>
  <p class="lede">A self-study curriculum in applied Bayesian statistics, written as executable notebooks
  for the current PyMC stack and published here as readable pages.</p>
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
    return (f"About — {SITE_NAME}", "about", body, "About this Bayesian modelling curriculum and how to run it.")


def pages(ctx):
    return {
        "index.html": home(ctx),
        "examples.html": examples_page(ctx),
        "challenges.html": challenges_page(ctx),
        "guide.html": guide_page(ctx),
        "about.html": about_page(ctx),
    }
