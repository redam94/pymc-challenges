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
