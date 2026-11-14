// Bayesian Modelling in Practice - small progressive enhancements. The pages work without it.
(function () {
  var root = document.documentElement;

  function store(key, value) { try { value === null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch (e) {} }


  // where the site's root is, read from this script's own URL (pages live at two depths)
  var ROOT = ((document.currentScript && document.currentScript.src) || "").replace(/assets\/site\.js.*$/, "");
  var indexPromise = null;
  function siteIndex() {  // every example and challenge: id, kind, title, data, tech, url, img, series, stars
    if (!indexPromise) {
      indexPromise = fetch(ROOT + "assets/search.json").then(function (r) { return r.json(); })
        .catch(function () { indexPromise = null; return []; });
    }
    return indexPromise;
  }
  window.BMP = { root: ROOT, index: siteIndex };

  // search: one dialog, opened from the navigation bar, a search bar on the page, or the "/" key
  var openers = document.querySelectorAll("[data-search-open]");
  if (openers.length && window.HTMLDialogElement) {
    var dialog = document.createElement("dialog");
    dialog.className = "search-dialog";
    dialog.setAttribute("aria-label", "Search examples and challenges");
    dialog.innerHTML = '<div class="search-field">' + openers[0].querySelector("svg").outerHTML +
      '<input type="search" placeholder="Search examples and challenges" aria-label="Search examples and challenges"' +
      ' autocomplete="off" spellcheck="false"></div><ul class="search-results"></ul>' +
      '<p class="search-note" aria-live="polite"></p>';
    document.body.appendChild(dialog);
    var sInput = dialog.querySelector("input"), sList = dialog.querySelector("ul"), sNote = dialog.querySelector("p");
    var entries = [], current = -1;

    var scoreOf = function (it, terms) {
      var total = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i], s = 0;
        if (it._id === t) s = 12;
        else if (it._title.indexOf(t) >= 0) s = new RegExp("(^|[^a-z0-9])" + t.replace(/[^a-z0-9]/g, "\\$&")).test(it._title) ? 6 : 4;
        else if (it._series.indexOf(t) >= 0) s = 2;
        else if (it._data.indexOf(t) >= 0) s = 1.5;
        else if (it._tech.indexOf(t) >= 0 || it._id.indexOf(t) === 0) s = 1;
        if (!s) return 0;  // every word must be found somewhere
        total += s;
      }
      return total;
    };

    var mark = function (i) {
      var rows = sList.children;
      if (!rows.length) return;
      current = (i + rows.length) % rows.length;
      for (var k = 0; k < rows.length; k++) rows[k].classList.toggle("is-active", k === current);
      rows[current].scrollIntoView({ block: "nearest" });
    };

    var runSearch = function () {
      var terms = sInput.value.toLowerCase().split(/\s+/).filter(Boolean);
      sList.textContent = ""; current = -1;
      if (!entries.length) { sNote.textContent = "Loading the list of notebooks."; return; }
      if (!terms.length) {
        sNote.textContent = "Search " + entries.length + " notebooks by model, dataset or topic. Try “survival”, " +
          "“polls” or “Gaussian process”.";
        return;
      }
      var hits = entries.map(function (it, i) { return [scoreOf(it, terms), i]; })
        .filter(function (h) { return h[0] > 0; })
        .sort(function (a, b) { return b[0] - a[0] || a[1] - b[1]; });
      hits.slice(0, 40).forEach(function (h) {
        var it = entries[h[1]], li = document.createElement("li"), a = document.createElement("a");
        a.className = "search-hit"; a.href = ROOT + it.url;
        var thumb;
        if (it.img) { thumb = document.createElement("img"); thumb.src = ROOT + it.img; thumb.alt = ""; thumb.loading = "lazy"; }
        else { thumb = document.createElement("span"); thumb.className = "hit-stars"; thumb.textContent = "★★★★★".slice(0, it.stars || 0); }
        var title = document.createElement("b"), id = document.createElement("span"), sub = document.createElement("small");
        id.className = "hit-id"; id.textContent = it.id;
        title.appendChild(id); title.appendChild(document.createTextNode(it.title));
        sub.textContent = (it.kind === "challenge" ? "Challenge, difficulty " + it.stars + " of 5. " : it.series ? it.series + ". " : "") + it.data;
        a.appendChild(thumb); a.appendChild(title); a.appendChild(sub);
        li.appendChild(a); sList.appendChild(li);
      });
      sNote.textContent = !hits.length ? "Nothing matches. Try fewer or shorter words."
        : hits.length > 40 ? "Showing the best 40 of " + hits.length + " matches. Add a word to narrow them."
        : hits.length + (hits.length === 1 ? " match." : " matches.") + " Arrow keys move, Enter opens.";
      mark(0);
    };

    var openSearch = function () {
      if (dialog.open) return;
      sInput.value = "";
      dialog.showModal();
      runSearch();
      siteIndex().then(function (items) {
        if (!entries.length) {
          entries = items.map(function (it) {
            it._id = it.id.toLowerCase(); it._title = it.title.toLowerCase(); it._series = (it.series || "").toLowerCase();
            it._data = it.data.toLowerCase(); it._tech = it.tech.toLowerCase();
            return it;
          });
        }
        runSearch();
      });
    };

    openers.forEach(function (b) { b.addEventListener("click", openSearch); });
    sInput.addEventListener("input", runSearch);
    sInput.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowDown") { ev.preventDefault(); mark(current + 1); }
      else if (ev.key === "ArrowUp") { ev.preventDefault(); mark(current - 1); }
      else if (ev.key === "Enter" && sList.children[current]) { ev.preventDefault(); sList.children[current].firstChild.click(); }
    });
    dialog.addEventListener("click", function (ev) { if (ev.target === dialog) dialog.close(); });  // the backdrop
    document.addEventListener("keydown", function (ev) {
      var el = document.activeElement, typing = el && (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || el.isContentEditable);
      if (ev.key === "/" && !typing && !ev.metaKey && !ev.ctrlKey && !ev.altKey) { ev.preventDefault(); openSearch(); }
    });
  }

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
