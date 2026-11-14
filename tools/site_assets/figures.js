// Bayesian Modelling in Practice - the interactive figures on the index pages.
// Each one enhances markup that already reads without it.
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(name, attrs, parent) {
    var node = document.createElementNS(NS, name);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function html(tag, cls, text, parent) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }

  // Call fn(true/false) as the node enters and leaves the viewport.
  function watch(node, fn, margin) {
    if (!("IntersectionObserver" in window)) { fn(true); return; }
    new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { fn(en.isIntersecting); });
    }, { rootMargin: margin || "0px" }).observe(node);
  }

  function once(node, fn, margin) {
    var done = false;
    watch(node, function (on) { if (on && !done) { done = true; fn(); } }, margin);
  }

  function randn() {
    var u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* ------------------------------------------------------------------
     1. Hero: four chains of Hamiltonian Monte Carlo on a curved posterior.
     The target is a "banana": theta1 ~ N(0, S^2), theta2 | theta1 ~ N(B (theta1^2 - S^2), 1),
     so the marginal of theta1 is known exactly and the histogram can be checked against it.
     ------------------------------------------------------------------ */
  function sampler(host) {
    var canvas = host.querySelector("canvas");
    var readout = host.querySelector("[data-readout]");
    var restart = host.querySelector("[data-restart]");
    if (!canvas || !canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    var layer = document.createElement("canvas");  // contours and accepted draws
    var lctx = layer.getContext("2d");

    var S = 1.6, B = 0.35, EPS = 0.2, PER_CHAIN = 400, BINS = 44;
    var X0 = -5.4, X1 = 5.4, Y0 = -3.8, Y1 = 8.8;
    var STARTS = [[-4.6, 7.6], [4.4, -2.6], [-1.2, -3.2], [2.4, 8.2]];
    var COLORS = ["#c3cf8f", "#e8e8ed", "#8f9c62", "#a1a1a6"];
    var W = 0, H = 0, MH = 0, dpr = 1;
    var chains = [], ghosts = [], bins = [], total = 0, visible = false, raf = 0;

    function logp(x, y) { var m = y - B * (x * x - S * S); return -x * x / (2 * S * S) - m * m / 2; }
    function grad(x, y) { var m = y - B * (x * x - S * S); return [-x / (S * S) + 2 * B * x * m, -m]; }
    function px(x) { return (x - X0) / (X1 - X0) * W; }
    function py(y) { return MH + 10 + (Y1 - y) / (Y1 - Y0) * (H - MH - 14); }

    function propose(c) {
      var q = c.q.slice(), p = [randn(), randn()];
      var L = 10 + Math.floor(Math.random() * 14);
      var h0 = -logp(q[0], q[1]) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
      var path = [q.slice()], g = grad(q[0], q[1]);
      p[0] += 0.5 * EPS * g[0]; p[1] += 0.5 * EPS * g[1];
      for (var i = 1; i <= L; i++) {
        q[0] += EPS * p[0]; q[1] += EPS * p[1];
        g = grad(q[0], q[1]);
        if (i < L) { p[0] += EPS * g[0]; p[1] += EPS * g[1]; }
        path.push(q.slice());
      }
      p[0] += 0.5 * EPS * g[0]; p[1] += 0.5 * EPS * g[1];
      var h1 = -logp(q[0], q[1]) + 0.5 * (p[0] * p[0] + p[1] * p[1]);
      c.path = path; c.step = 0; c.accept = Math.log(Math.random()) < h0 - h1;
    }

    function dot(c) {
      lctx.globalAlpha = 0.7;
      lctx.fillStyle = c.color;
      lctx.beginPath();
      lctx.arc(px(c.q[0]), py(c.q[1]), 1.6, 0, 2 * Math.PI);
      lctx.fill();
      lctx.globalAlpha = 1;
    }

    function finish(c) {
      if (c.accept) c.q = c.path[c.path.length - 1].slice();
      else ghosts.push({ path: c.path, life: 1 });
      c.draws.push(c.q.slice());
      total++;
      var b = Math.floor((c.q[0] - X0) / (X1 - X0) * BINS);
      if (b >= 0 && b < BINS) bins[b]++;
      if (c.accept) dot(c);
      if (c.draws.length < PER_CHAIN) propose(c); else c.path = null;
    }

    function advance() {
      var speed = reduced ? 1e9 : total < 60 ? 1 : total < 300 ? 2 : 4;
      chains.forEach(function (c) {
        for (var k = 0; k < speed && c.path; k++) {
          c.step++;
          if (c.step >= c.path.length - 1) finish(c);
        }
      });
    }

    function rhat() {
      var worst = 0;
      for (var d = 0; d < 2; d++) {
        var means = [], vars = [], n = Infinity;
        chains.forEach(function (c) { n = Math.min(n, c.draws.length); });
        n = Math.floor(n / 2);
        if (n < 10) return null;
        chains.forEach(function (c) {
          var xs = c.draws.slice(c.draws.length - n).map(function (q) { return q[d]; });
          var m = xs.reduce(function (a, v) { return a + v; }, 0) / n;
          means.push(m);
          vars.push(xs.reduce(function (a, v) { return a + (v - m) * (v - m); }, 0) / (n - 1));
        });
        var w = vars.reduce(function (a, v) { return a + v; }, 0) / vars.length;
        var mm = means.reduce(function (a, v) { return a + v; }, 0) / means.length;
        var between = means.reduce(function (a, v) { return a + (v - mm) * (v - mm); }, 0) / (means.length - 1);
        worst = Math.max(worst, Math.sqrt(((n - 1) / n * w + between) / w));
      }
      return worst;
    }

    function drawLayer() {
      lctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      lctx.clearRect(0, 0, W, H);
      // level sets of -log p, widest first; each holds 1 - exp(-c) of the mass
      [4.6, 3, 1.5, 0.5].forEach(function (c, i) {
        var xm = S * Math.sqrt(2 * c), n = 90, k, x, r;
        lctx.beginPath();
        for (k = 0; k <= n; k++) {
          x = -xm + 2 * xm * k / n; r = Math.sqrt(Math.max(0, 2 * (c - x * x / (2 * S * S))));
          lctx[k ? "lineTo" : "moveTo"](px(x), py(B * (x * x - S * S) + r));
        }
        for (k = n; k >= 0; k--) {
          x = -xm + 2 * xm * k / n; r = Math.sqrt(Math.max(0, 2 * (c - x * x / (2 * S * S))));
          lctx.lineTo(px(x), py(B * (x * x - S * S) - r));
        }
        lctx.closePath();
        lctx.fillStyle = "rgba(195, 207, 143, 0.045)";
        lctx.fill();
        lctx.strokeStyle = i === 3 ? "#5a5a60" : "#38383b";
        lctx.lineWidth = 1;
        lctx.stroke();
      });
      chains.forEach(function (c) {
        var q = c.q;
        c.draws.forEach(function (d) { c.q = d; dot(c); });
        c.q = q;
      });
    }

    function drawMarginal() {
      var peak = 1 / (S * Math.sqrt(2 * Math.PI)), top = 6, h = MH - top;
      var bw = W / BINS, norm = total * (X1 - X0) / BINS, i, x;
      if (total >= 40) {  // a handful of draws is not yet a histogram
        ctx.beginPath();
        ctx.moveTo(0, MH);
        for (i = 0; i < BINS; i++) {
          var v = MH - Math.min(1.25, bins[i] / norm / peak) * h * 0.8;
          ctx.lineTo(i * bw, v); ctx.lineTo((i + 1) * bw, v);
        }
        ctx.lineTo(W, MH);
        ctx.fillStyle = "rgba(195, 207, 143, 0.22)";
        ctx.fill();
        ctx.strokeStyle = "#c3cf8f";
        ctx.lineWidth = 1.25;
        ctx.stroke();
      }
      ctx.beginPath();
      for (i = 0; i <= 120; i++) {
        x = X0 + (X1 - X0) * i / 120;
        ctx[i ? "lineTo" : "moveTo"](px(x), MH - Math.exp(-x * x / (2 * S * S)) * h * 0.8);
      }
      ctx.setLineDash([5, 5]);
      ctx.strokeStyle = "#a1a1a6";
      ctx.lineWidth = 1.25;
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(0, MH + 0.5); ctx.lineTo(W, MH + 0.5);
      ctx.strokeStyle = "#424245";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    function trace(path, upTo) {
      ctx.beginPath();
      for (var i = 0; i <= upTo; i++) ctx[i ? "lineTo" : "moveTo"](px(path[i][0]), py(path[i][1]));
      ctx.stroke();
    }

    function render() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(layer, 0, 0, W, H);
      drawMarginal();
      ctx.lineJoin = "round";
      // rejected proposals linger for a moment, dashed
      ghosts = ghosts.filter(function (g) { return g.life > 0; });
      ghosts.forEach(function (g) {
        ctx.globalAlpha = g.life * 0.6;
        ctx.strokeStyle = "#86868b";
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        trace(g.path, g.path.length - 1);
        g.life -= 0.05;
      });
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      chains.forEach(function (c) {
        var head = c.path ? c.path[Math.min(c.step, c.path.length - 1)] : c.q;
        if (c.path) {
          ctx.strokeStyle = c.color;
          ctx.lineWidth = 1.5;
          ctx.globalAlpha = 0.85;
          trace(c.path, Math.min(c.step, c.path.length - 1));
          ctx.globalAlpha = 1;
        }
        ctx.beginPath();
        ctx.arc(px(head[0]), py(head[1]), 4, 0, 2 * Math.PI);
        ctx.fillStyle = c.color;
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
      var r = rhat(), done = total >= PER_CHAIN * chains.length;
      readout.textContent = !total ? "Four chains, about to start"
        : total.toLocaleString("en") + " draws from 4 chains" + (r ? ", R-hat " + r.toFixed(2) : ", R-hat not yet defined")
          + (done ? ". Sampling finished." : "");
      return done;
    }

    function tick() {
      raf = 0;
      advance();
      var done = render();
      if (visible && !done && !document.hidden) raf = requestAnimationFrame(tick);
    }

    function start(points) {
      chains = points.map(function (q, i) {
        var c = { q: q.slice(), draws: [], color: COLORS[i], path: null, step: 0 };
        propose(c);
        return c;
      });
      ghosts = []; total = 0; bins = [];
      for (var i = 0; i < BINS; i++) bins.push(0);
      drawLayer();
      if (!raf) raf = requestAnimationFrame(tick);
    }

    function resize() {
      var box = canvas.getBoundingClientRect();
      if (!box.width) return;
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = box.width; H = box.height; MH = Math.round(H * 0.2);
      [canvas, layer].forEach(function (c) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); });
      drawLayer();
      render();
    }

    canvas.addEventListener("click", function (ev) {
      var box = canvas.getBoundingClientRect();
      var x = X0 + (ev.clientX - box.left) / W * (X1 - X0);
      var y = Y1 - (ev.clientY - box.top - MH - 10) / (H - MH - 14) * (Y1 - Y0);
      if (y > Y1) return;  // the marginal strip
      start(STARTS.map(function () { return [x + randn() * 0.15, y + randn() * 0.15]; }));
    });
    if (restart) restart.addEventListener("click", function () { start(STARTS); });
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && visible && !raf) raf = requestAnimationFrame(tick);
    });

    chains = STARTS.map(function (q, i) { return { q: q.slice(), draws: [], color: COLORS[i], path: null, step: 0 }; });
    for (var i = 0; i < BINS; i++) bins.push(0);
    resize();
    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(canvas);
    else window.addEventListener("resize", resize);
    var started = false;
    watch(host, function (on) {
      visible = on;
      if (on && !started) { started = true; start(STARTS); }
      else if (on && !raf) raf = requestAnimationFrame(tick);
    });
  }

  /* ------------------------------------------------------------------
     2. A Beta-Binomial posterior that updates one observation at a time.
     ------------------------------------------------------------------ */
  function updater(host) {
    var svg = host.querySelector("svg");
    var readout = host.querySelector("[data-readout]");
    var W = 720, H = 300, L = 24, R = 24, T = 14, BASE = 226, N = 240;
    var A0 = 2, B0 = 2, TRUE_RATE = 0.68;
    var obs = [], shown = { a: A0, b: B0 }, anim = 0, intro = 0;

    var soft = el("path", { "class": "f-soft" }, svg);
    var mid = el("path", { "class": "f-mid" }, svg);
    var prior = el("path", { "class": "f-muted f-dash" }, svg);
    var post = el("path", { "class": "f-accent" }, svg);
    el("line", { "class": "f-line", x1: L, x2: W - R, y1: BASE, y2: BASE }, svg);
    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      var x = L + t * (W - L - R);
      el("line", { "class": "f-line", x1: x, x2: x, y1: BASE, y2: BASE + 5 }, svg);
      var label = el("text", { "class": "f-text f-text-muted", x: x, y: BASE + 24, "text-anchor": "middle" }, svg);
      label.textContent = t === 0 ? "0" : t === 1 ? "1" : String(t);
    });
    var meanMark = el("line", { "class": "f-accent", y1: BASE - 7, y2: BASE + 7 }, svg);
    var dots = el("g", {}, svg);

    function density(a, b) {
      var d = [], peak = -Infinity, sum = 0, i, x;
      for (i = 0; i < N; i++) {
        x = (i + 0.5) / N;
        d.push((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x));
        if (d[i] > peak) peak = d[i];
      }
      for (i = 0; i < N; i++) { d[i] = Math.exp(d[i] - peak); sum += d[i]; }
      for (i = 0; i < N; i++) d[i] *= N / sum;
      return d;
    }

    function quantile(d, p) {
      var acc = 0;
      for (var i = 0; i < N; i++) {
        acc += d[i] / N;
        if (acc >= p) return (i + 0.5) / N;
      }
      return 1;
    }

    function draw(a, b) {
      var d = density(a, b), d0 = density(A0, B0);
      var top = Math.max(Math.max.apply(null, d), Math.max.apply(null, d0)) * 1.06;
      var X = function (t) { return L + t * (W - L - R); };
      var Y = function (v) { return BASE - v / top * (BASE - T); };
      var line = function (dd) {
        return dd.map(function (v, i) { return (i ? "L" : "M") + X((i + 0.5) / N).toFixed(1) + " " + Y(v).toFixed(1); }).join("");
      };
      var area = function (lo, hi) {
        var s = "M" + X(lo).toFixed(1) + " " + BASE;
        for (var i = 0; i < N; i++) {
          var t = (i + 0.5) / N;
          if (t >= lo && t <= hi) s += "L" + X(t).toFixed(1) + " " + Y(d[i]).toFixed(1);
        }
        return s + "L" + X(hi).toFixed(1) + " " + BASE + "Z";
      };
      prior.setAttribute("d", line(d0));
      post.setAttribute("d", line(d));
      soft.setAttribute("d", area(quantile(d, 0.03), quantile(d, 0.97)));
      mid.setAttribute("d", area(quantile(d, 0.25), quantile(d, 0.75)));
      var m = X(a / (a + b));
      meanMark.setAttribute("x1", m); meanMark.setAttribute("x2", m);
    }

    function describe() {
      var s = obs.filter(Boolean).length, n = obs.length;
      var d = density(A0 + s, B0 + n - s);
      if (!n) { readout.textContent = "No data yet. The dashed curve is the prior, Beta(2, 2): rates near one half are a little more plausible than rates near 0 or 1."; return; }
      readout.textContent = s + (s === 1 ? " success" : " successes") + " in " + n + (n === 1 ? " trial" : " trials")
        + ". The posterior mean is " + ((A0 + s) / (A0 + B0 + n)).toFixed(2) + ", and 94% of the posterior lies between "
        + quantile(d, 0.03).toFixed(2) + " and " + quantile(d, 0.97).toFixed(2) + ".";
    }

    function drawDots() {
      while (dots.firstChild) dots.removeChild(dots.firstChild);
      var recent = obs.slice(-48), x0 = W / 2 - (recent.length - 1) * 7;
      recent.forEach(function (o, i) {
        el("circle", { cx: x0 + i * 14, cy: BASE + 52, r: 4.5, "class": o ? "f-fill-accent" : "f-muted" }, dots);
      });
    }

    function update() {
      var s = obs.filter(Boolean).length, target = { a: A0 + s, b: B0 + obs.length - s };
      var from = { a: shown.a, b: shown.b }, t0 = 0;
      describe();
      drawDots();
      cancelAnimationFrame(anim);
      if (reduced) { shown = target; draw(target.a, target.b); return; }
      (function frame(now) {
        if (!t0) t0 = now;
        var k = Math.min(1, (now - t0) / 420);
        k = 1 - Math.pow(1 - k, 3);
        shown = { a: from.a + (target.a - from.a) * k, b: from.b + (target.b - from.b) * k };
        draw(shown.a, shown.b);
        if (k < 1) anim = requestAnimationFrame(frame);
      })(performance.now());
    }

    host.addEventListener("click", function (ev) {
      var act = ev.target.getAttribute && ev.target.getAttribute("data-act");
      if (!act) return;
      clearInterval(intro);
      if (act === "success") obs.push(1);
      else if (act === "failure") obs.push(0);
      else if (act === "ten") for (var i = 0; i < 10; i++) obs.push(Math.random() < TRUE_RATE ? 1 : 0);
      else obs = [];
      update();
    });

    draw(A0, B0);
    describe();
    // the one staged moment: eight observations arrive as the figure first scrolls in
    once(host, function () {
      if (reduced) { for (var i = 0; i < 8; i++) obs.push(Math.random() < TRUE_RATE ? 1 : 0); update(); return; }
      var left = 8;
      intro = setInterval(function () {
        obs.push(Math.random() < TRUE_RATE ? 1 : 0);
        update();
        if (--left <= 0) clearInterval(intro);
      }, 420);
    }, "-25% 0px");
  }

  /* ------------------------------------------------------------------
     3. The workflow as a track: one step at a time, and a way back when a check fails.
     ------------------------------------------------------------------ */
  function flow(host) {
    var panels = Array.prototype.slice.call(host.querySelectorAll(".flow-panel"));
    if (!panels.length) return;
    var list = host.querySelector(".flow-panels");
    var track = html("ol", "flow-track");
    var steps = panels.map(function (panel, i) {
      var name = panel.querySelector("h3").textContent;
      var btn = html("button", "flow-step", "", html("li", "", "", track));
      btn.type = "button";
      btn.setAttribute("aria-label", "Step " + (i + 1) + ": " + name);
      if (panel.hasAttribute("data-check")) btn.setAttribute("data-check", "");
      html("b", "", String(i + 1), html("span", "flow-dot", "", btn));
      html("span", "", name, btn);
      btn.addEventListener("click", function () { select(i, ""); });
      return btn;
    });
    host.insertBefore(track, list);
    var controls = html("div", "flow-controls", "", host);
    var fail = html("button", "pill", "This check fails", controls);
    var next = html("button", "pill pill-accent", "Next step", controls);
    var note = html("p", "flow-note", "", host);
    fail.type = next.type = "button";
    note.setAttribute("aria-live", "polite");
    var active = 0;

    function select(i, message) {
      active = i;
      steps.forEach(function (b, k) {
        b.classList.toggle("is-active", k === i);
        b.classList.toggle("is-done", k < i);
        if (k === i) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
      });
      panels.forEach(function (p, k) { p.classList.toggle("is-active", k === i); });
      host.style.setProperty("--progress", i / (panels.length - 1));
      fail.hidden = !panels[i].hasAttribute("data-check");
      next.textContent = i === panels.length - 1 ? "Start again" : "Next step";
      note.textContent = message;
    }

    next.addEventListener("click", function () { select((active + 1) % panels.length, ""); });
    fail.addEventListener("click", function () {
      host.classList.add("is-revising");
      select(1, "Back to the model. Revise it, and say what changed.");
      setTimeout(function () { host.classList.remove("is-revising"); }, 1400);
    });
    host.classList.add("is-live");
    select(0, "");
  }

  function init() {
    document.querySelectorAll("[data-sampler]").forEach(sampler);
    document.querySelectorAll("[data-updater]").forEach(updater);
    document.querySelectorAll("[data-flow]").forEach(flow);
    document.querySelectorAll(".fig[data-reveal]").forEach(function (fig) {
      once(fig, function () { fig.classList.add("is-in"); }, "-15% 0px");
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
