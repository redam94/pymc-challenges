// Bayesian Modelling in Practice - small progressive enhancements. The pages work without it.
(function () {
  var root = document.documentElement;

  function store(key, value) { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch (e) {} }

  // theme
  var themeBtn = document.querySelector(".theme-toggle");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var dark = root.dataset.theme ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    store("theme", root.dataset.theme);
  });

  // hide / show code
  var codeBtn = document.querySelector(".code-toggle");
  function syncCodeBtn() { if (codeBtn) codeBtn.textContent = root.classList.contains("hide-code") ? "Show code" : "Hide code"; }
  if (codeBtn) {
    syncCodeBtn();
    codeBtn.addEventListener("click", function () {
      root.classList.toggle("hide-code");
      store("hideCode", root.classList.contains("hide-code") ? "1" : null);
      syncCodeBtn();
    });
  }

  // long code cells start collapsed
  document.querySelectorAll(".code.long").forEach(function (block) {
    var n = block.textContent.split("\n").length;
    block.classList.add("collapsed");
    var btn = document.createElement("button");
    btn.type = "button"; btn.className = "expand-code"; btn.textContent = "Show all " + n + " lines";
    btn.addEventListener("click", function () {
      var c = block.classList.toggle("collapsed");
      btn.textContent = c ? "Show all " + n + " lines" : "Collapse";
    });
    block.after(btn);
  });

  // plotly figures (data stored next to the div as JSON)
  var plots = document.querySelectorAll("script[data-plotly]");
  if (plots.length && window.Plotly) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        io.unobserve(en.target);
        var spec = JSON.parse(document.querySelector('script[data-plotly="' + en.target.id + '"]').textContent);
        var layout = Object.assign({}, spec.layout || {}, { autosize: true });
        delete layout.width;
        Plotly.newPlot(en.target, spec.data || [], layout, { responsive: true, displaylogo: false });
      });
    }, { rootMargin: "400px" });
    plots.forEach(function (s) { var el = document.getElementById(s.dataset.plotly); if (el) io.observe(el); });
  }

  // reading progress under the navigation bar (notebook pages)
  var bar = document.querySelector(".read-progress");
  if (bar) {
    var onProgress = function () {
      var max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = "scaleX(" + (max > 0 ? Math.min(1, window.scrollY / max) : 0) + ")";
    };
    window.addEventListener("scroll", onProgress, { passive: true });
    window.addEventListener("resize", onProgress);
    onProgress();
  }

  // examples page: filter the cards by any words in their title, data or techniques
  var filter = document.querySelector("[data-filter]");
  if (filter) {
    var input = filter.querySelector("input"), count = filter.querySelector(".filter-count");
    var cards = Array.prototype.slice.call(document.querySelectorAll(".series .card"));
    var sections = Array.prototype.slice.call(document.querySelectorAll(".series"));
    var haystack = cards.map(function (c) { return c.textContent.toLowerCase(); });
    var applyFilter = function () {
      var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean), shown = 0;
      cards.forEach(function (c, i) {
        var ok = terms.every(function (t) { return haystack[i].indexOf(t) >= 0; });
        c.hidden = !ok;
        if (ok) shown++;
      });
      sections.forEach(function (s) { s.hidden = !s.querySelector(".card:not([hidden])"); });
      count.textContent = !terms.length ? cards.length + " examples in " + sections.length + " series"
        : shown ? shown + " of " + cards.length + " examples match"
        : "No example matches. Try fewer or shorter words.";
    };
    filter.hidden = false;
    input.addEventListener("input", applyFilter);
    applyFilter();
  }

  // a sideways rail of cards gets previous and next buttons
  document.querySelectorAll("[data-rail]").forEach(function (rail) {
    var track = rail.querySelector(".rail-track");
    var nav = document.createElement("div");
    nav.className = "rail-nav";
    var buttons = [["Previous", "←", -1], ["Next", "→", 1]].map(function (spec) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = spec[1]; b.setAttribute("aria-label", spec[0]);
      b.addEventListener("click", function () {
        track.scrollBy({ left: spec[2] * track.clientWidth * 0.85, behavior: "smooth" });
      });
      nav.appendChild(b);
      return b;
    });
    var sync = function () {
      buttons[0].disabled = track.scrollLeft < 4;
      buttons[1].disabled = track.scrollLeft > track.scrollWidth - track.clientWidth - 4;
    };
    track.addEventListener("scroll", sync, { passive: true });
    rail.appendChild(nav);
    sync();
  });

  // table of contents: highlight the current section
  var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
  if (links.length) {
    var targets = links.map(function (a) { return document.getElementById(decodeURIComponent(a.hash.slice(1))); });
    var onScroll = function () {
      var y = window.scrollY + 120, current = 0;
      targets.forEach(function (t, i) { if (t && t.offsetTop <= y) current = i; });
      links.forEach(function (a, i) { a.classList.toggle("active", i === current); });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    if (window.matchMedia("(max-width: 64rem)").matches) {
      var d = document.querySelector(".toc details"); if (d) d.removeAttribute("open");
    }
  }
})();

// size an animation iframe to its content (same origin)
function fitFrame(frame) {
  try {
    var doc = frame.contentDocument;
    var fit = function () { frame.style.height = (doc.body.offsetHeight + 8) + "px"; };
    fit();
    doc.querySelectorAll("img").forEach(function (img) { img.addEventListener("load", fit); });
    setTimeout(fit, 400);
  } catch (e) {}
}
