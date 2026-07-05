/* FUT Market Desk — interactive dashboard.
   FUT.init()  drives the live app (served by FastAPI): fetches data, wires the
               execution controls / scraper panel / link manager, and polls jobs.
   FUT.render(d) renders a static read-only snapshot (the Artifact preview, where
               window.__FUT_DATA__ is inlined and there is no backend). */
(function () {
  "use strict";

  var MONO = 'ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace';
  var state = { data: null, watch: [], jobs: [], selected: null, live: false, lastRunning: false };

  // ---- helpers ----
  function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function parseT(s) { return new Date(s); }
  function fmtCoins(n, compact) {
    if (n == null) return "–";
    if (compact) {
      if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
      if (Math.abs(n) >= 1e3) return Math.round(n / 1e3) + "K";
      return String(Math.round(n));
    }
    return Math.round(n).toLocaleString("en-US");
  }
  function fmtPct(x) { return x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(1) + "%"; }
  var MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function fmtDay(d) { return MON[d.getUTCMonth()] + " " + d.getUTCDate(); }

  function api(method, path, body) {
    return fetch(path, {
      method: method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (r) {
      if (r.status === 401) { showLogin(); throw new Error("locked"); }
      if (!r.ok) return r.json().then(function (j) { throw new Error(j.detail || r.status); });
      return r.status === 204 ? {} : r.json();
    });
  }

  // ---- login gate ----
  function showLogin() {
    if (document.getElementById("login")) return;
    var ov = el("div", "loginov"); ov.id = "login";
    var box = el("div", "loginbox");
    box.appendChild(el("div", "loginlogo", "🔒"));
    box.appendChild(el("h2", null, "FUT Market Desk"));
    box.appendChild(el("p", "muted", "Enter the access key shown in the terminal where you started the dashboard."));
    var inp = el("input", "urlinput"); inp.type = "password"; inp.placeholder = "Access key";
    inp.autofocus = true;
    var err = el("div", "loginerr", "");
    var btn = el("button", "btn primary", "Unlock");
    function submit() {
      var key = inp.value.trim(); if (!key) return;
      btn.disabled = true; err.textContent = "";
      fetch("/api/login", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: key }) })
        .then(function (r) {
          btn.disabled = false;
          if (r.ok) { document.body.removeChild(ov); probe(); }
          else { err.textContent = "Invalid key — try again."; inp.select(); }
        })
        .catch(function () { btn.disabled = false; err.textContent = "Could not reach the server."; });
    }
    inp.onkeydown = function (e) { if (e.key === "Enter") submit(); };
    btn.onclick = submit;
    box.appendChild(inp); box.appendChild(err); box.appendChild(btn);
    ov.appendChild(box); document.body.appendChild(ov);
    setTimeout(function () { inp.focus(); }, 30);
  }
  function lock() {
    fetch("/api/logout", { method: "POST" }).then(function () { showLogin(); });
  }

  // ============================ live app ============================
  function init() {
    state.live = true;
    var app = document.getElementById("app"); app.innerHTML = "";
    app.appendChild(header());
    app.appendChild(controls());
    app.appendChild(scraperPanel());
    app.appendChild(momentumPanel());
    app.appendChild(watchManager());
    app.appendChild(activityPanel());
    app.appendChild(el("div", null, "")).id = "readonly";
    window.addEventListener("resize", debounce(drawChart, 120));
    new MutationObserver(function (m) {
      for (var i = 0; i < m.length; i++) if (m[i].attributeName === "data-theme") { drawChart(); break; }
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    probe();
  }

  // Single gated call on load: succeeds → go live; 401 → api() shows the login
  // overlay (so an unauthenticated load makes one request, not a burst).
  function probe() {
    api("GET", "/api/data").then(function (d) {
      state.data = d; renderReadonly(); afterAuth();
    }).catch(function () {});
  }
  function afterAuth() {
    refreshWatch(); refreshMomentum(); pollJobs();
    if (!state._poll) state._poll = setInterval(pollJobs, 1500);
  }
  function refreshMomentum() {
    api("GET", "/api/momentum").then(function (d) {
      state.momentum = d.players; state.momentumUpdated = d.updated_at; renderMomentum();
    }).catch(function () {});
  }
  function refreshAll() { refreshData(); refreshWatch(); pollJobs(); }
  function refreshData() {
    api("GET", "/api/data").then(function (d) { state.data = d; renderReadonly(); })
      .catch(function () {});
  }
  function refreshWatch() {
    api("GET", "/api/watchlist").then(function (d) {
      state.watch = d.players; renderWatch(); renderMomentum();  // keep "Tracked" badges in sync
    }).catch(function () {});
  }

  // ---- header ----
  function header() {
    var h = el("div", "hdr");
    var top = el("div", "hdrtop");
    top.appendChild(el("h1", null, "FUT Market Desk"));
    if (state.live) {
      var lk = el("button", "btn tiny", "Lock"); lk.title = "Log out";
      lk.onclick = lock; top.appendChild(lk);
    }
    h.appendChild(top);
    h.appendChild(el("div", "sub id-src", "connecting…"));
    return h;
  }

  // ---- execution controls ----
  function controls() {
    var wrap = el("div", "controls");
    var defs = [
      ["Scrape all", function () { runJob("/api/collect", {}, "Scraping all players"); }, "primary"],
      ["Run backtest", function () { runJob("/api/backtest", null, "Running backtest"); }, ""],
      ["Recompute signals", function () { runJob("/api/signals", null, "Recomputing signals"); }, ""],
      ["Rebuild features", function () { runJob("/api/build-features", null, "Rebuilding features"); }, ""],
    ];
    defs.forEach(function (d) {
      var b = el("button", "btn " + d[2], d[0]); b.onclick = d[1];
      b.dataset.action = "1"; wrap.appendChild(b);
    });
    return wrap;
  }

  function runJob(path, body, label) {
    setBusy(true);
    toast(label + "…");
    api("POST", path, body).then(function (r) {
      state.watchJob = r.job_id;
    }).catch(function (e) { toast("Error: " + e.message, true); setBusy(false); });
  }
  function setBusy(b) {
    var btns = document.querySelectorAll(".controls .btn");
    Array.prototype.forEach.call(btns, function (x) { x.disabled = b; });
  }

  // ---- scraper engine panel ----
  function scraperPanel() {
    var card = el("div", "card"); card.id = "scraper";
    var head = el("div", "cardhead");
    head.appendChild(el("div", "ctitle", "Scraping engine"));
    var badge = el("span", "statusdot idle", "idle"); badge.id = "scr-badge";
    head.appendChild(badge); card.appendChild(head);
    var row = el("div", "scr-row");
    var start = el("button", "btn primary", "Start continuous"); start.id = "scr-start";
    start.onclick = function () { api("POST", "/api/scraper/start").then(applyScraper); };
    var stop = el("button", "btn danger", "Stop"); stop.id = "scr-stop"; stop.disabled = true;
    stop.onclick = function () { api("POST", "/api/scraper/stop").then(applyScraper); };
    row.appendChild(start); row.appendChild(stop);
    var info = el("div", "scr-info muted"); info.id = "scr-info"; row.appendChild(info);
    card.appendChild(row);
    // active-job progress + log
    var prog = el("div", "prog hidden"); prog.id = "scr-prog";
    prog.innerHTML = '<div class="bar"><i></i></div><div class="proglabel muted"></div>';
    card.appendChild(prog);
    var log = el("pre", "log hidden"); log.id = "scr-log"; card.appendChild(log);
    return card;
  }
  function applyScraper(s) {
    var on = s.running;
    document.getElementById("scr-start").disabled = on;
    document.getElementById("scr-stop").disabled = !on;
    var b = document.getElementById("scr-badge");
    b.className = "statusdot " + (on ? "on" : "idle");
    b.textContent = on ? "running" : "idle";
    document.getElementById("scr-info").textContent =
      "continuous mode " + (on ? "on — scraping every " + s.poll_minutes + " min" : "off");
  }

  // ---- momentum scanner (market movers) ----
  function momentumPanel() {
    var card = el("div", "card"); card.id = "momentum";
    var head = el("div", "cardhead");
    head.appendChild(el("div", "ctitle", "Momentum — market movers"));
    var upd = el("span", "muted mupd"); upd.id = "mom-updated"; head.appendChild(upd);
    var ref = el("button", "btn tiny primary", "Refresh"); ref.id = "mom-refresh";
    ref.style.marginLeft = "auto";
    ref.onclick = function () {
      setMomBusy(true);
      api("POST", "/api/momentum/refresh").catch(function (e) { toast("Error: " + e.message, true); setMomBusy(false); });
    };
    head.appendChild(ref); card.appendChild(head);
    card.appendChild(el("div", "muted mnote",
      "Biggest special-card price movers on fut.gg — a discovery aid. Track one, then confirm with its history &amp; backtest."));
    card.appendChild(el("div", "mlist", "")).id = "mom-list";
    return card;
  }
  function setMomBusy(b) {
    var r = document.getElementById("mom-refresh");
    if (r) { r.disabled = b; r.textContent = b ? "Refreshing…" : "Refresh"; }
  }
  function renderMomentum() {
    var box = document.getElementById("mom-list"); if (!box) return;
    var upd = document.getElementById("mom-updated");
    if (upd) upd.textContent = state.momentumUpdated ? "updated " + ago(state.momentumUpdated) : "";
    box.innerHTML = "";
    var rows = state.momentum || [];
    if (!rows.length) {
      box.appendChild(el("div", "muted empty-sm", "No data yet — hit Refresh to scan the market.")); return;
    }
    var tracked = {};
    (state.watch || []).forEach(function (p) { tracked[p.player_id] = true; });
    var t = el("table", "players mtable");
    t.innerHTML = "<thead><tr><th>#</th><th class='r'>Mom</th><th>Player</th>" +
      "<th class='r'>Price</th><th></th></tr></thead>";
    var body = el("tbody");
    rows.forEach(function (p) {
      var tr = el("tr");
      var sub = (p.rating != null ? p.rating + " " : "") + (p.position || "") +
        (p.rarity ? " · " + p.rarity : "");
      tr.innerHTML =
        "<td class='rtg num'>" + p.rank + "</td>" +
        "<td class='r num mom'>" + p.momentum.toFixed(1) + "</td>" +
        "<td class='pname'>" + p.name + "<div class='wmeta muted'>" + sub + "</div></td>" +
        "<td class='r num'>" + fmtCoins(p.price, true) + "</td>";
      var td = el("td", "r");
      if (tracked[p.player_id]) {
        td.appendChild(el("span", "badge", "Tracked"));
      } else {
        var b = el("button", "btn tiny", "Track");
        b.onclick = function () { trackFromMomentum(p, b); };
        td.appendChild(b);
      }
      tr.appendChild(td); body.appendChild(tr);
    });
    t.appendChild(body); box.appendChild(t);
  }
  function trackFromMomentum(p, btn) {
    btn.disabled = true; btn.textContent = "…";
    api("POST", "/api/watchlist", { url: p.url }).then(function () {
      toast("Tracking " + p.name + " — fetching prices…");
      refreshWatch();
    }).catch(function (e) {
      btn.disabled = false; btn.textContent = "Track"; toast("Error: " + e.message, true);
    });
  }
  function ago(iso) {
    var s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 90) return "just now";
    if (s < 5400) return Math.round(s / 60) + "m ago";
    return Math.round(s / 3600) + "h ago";
  }

  // ---- watchlist / player-link manager ----
  function watchManager() {
    var card = el("div", "card"); card.id = "watch-mgr";
    var head = el("div", "cardhead");
    head.appendChild(el("div", "ctitle", "Tracked player links"));
    card.appendChild(head);
    var form = el("div", "addrow");
    var input = el("input", "urlinput"); input.id = "url-in";
    input.placeholder = "Paste a fut.gg player URL…";
    input.onkeydown = function (e) { if (e.key === "Enter") addPlayer(); };
    var add = el("button", "btn primary", "Add"); add.onclick = addPlayer;
    form.appendChild(input); form.appendChild(add); card.appendChild(form);
    var list = el("div", "watchlist"); list.id = "watch-list"; card.appendChild(list);
    return card;
  }
  function addPlayer() {
    var input = document.getElementById("url-in");
    var url = input.value.trim(); if (!url) return;
    input.disabled = true;
    api("POST", "/api/watchlist", { url: url }).then(function (r) {
      input.value = ""; input.disabled = false;
      toast("Added " + r.added.name + " — fetching prices…");
      state.watchJob = r.job_id; refreshWatch();
    }).catch(function (e) { input.disabled = false; toast("Error: " + e.message, true); });
  }
  function removePlayer(pid, name) {
    if (!window.confirm("Stop tracking " + name + "?")) return;
    api("DELETE", "/api/watchlist/" + encodeURIComponent(pid))
      .then(function () { toast("Removed " + name); refreshWatch(); refreshData(); })
      .catch(function (e) { toast("Error: " + e.message, true); });
  }
  function renderWatch() {
    var list = document.getElementById("watch-list"); if (!list) return;
    list.innerHTML = "";
    if (!state.watch.length) { list.appendChild(el("div", "muted empty-sm", "No players yet — paste a fut.gg URL above.")); return; }
    state.watch.forEach(function (p) {
      var row = el("div", "wrow");
      row.appendChild(el("div", "rtg num", p.rating != null ? String(p.rating) : "–"));
      var mid = el("div", "wmid");
      mid.appendChild(el("div", "wname", p.name));
      mid.appendChild(el("div", "wmeta muted", (p.version || "") + " · " + (p.snapshots || 0) + " snapshots"));
      row.appendChild(mid);
      var scrape = el("button", "btn tiny", "Scrape"); scrape.title = "Scrape just this player";
      scrape.onclick = function () { runJob("/api/collect", { player_id: p.player_id }, "Scraping " + p.name); };
      var rm = el("button", "btn tiny danger", "✕"); rm.title = "Remove";
      rm.onclick = function () { removePlayer(p.player_id, p.name); };
      row.appendChild(scrape); row.appendChild(rm);
      list.appendChild(row);
    });
  }

  // ---- activity / jobs feed ----
  function activityPanel() {
    var card = el("div", "card"); card.id = "activity";
    var head = el("div", "cardhead"); head.appendChild(el("div", "ctitle", "Activity"));
    card.appendChild(head);
    card.appendChild(el("div", "jobs", "")).id = "jobs-list";
    return card;
  }
  function pollJobs() {
    if (!state.live) return;
    api("GET", "/api/scraper/status").then(applyScraper).catch(function () {});
    api("GET", "/api/jobs").then(function (d) {
      state.jobs = d.jobs; renderJobs(d.jobs);
      var running = d.jobs.some(function (j) { return j.status === "running" || j.status === "queued"; });
      setBusy(running);
      // show live progress + log for the running job
      var cur = d.current || (d.jobs[0] && (d.jobs[0].status === "running") ? d.jobs[0].id : null);
      if (cur) showJobProgress(cur); else hideJobProgress();
      // when work finishes, pull fresh data + watchlist + momentum once
      if (state.lastRunning && !running) { refreshData(); refreshWatch(); refreshMomentum(); setMomBusy(false); }
      state.lastRunning = running;
    }).catch(function () {});
  }
  function renderJobs(jobs) {
    var box = document.getElementById("jobs-list"); if (!box) return;
    box.innerHTML = "";
    if (!jobs.length) { box.appendChild(el("div", "muted empty-sm", "No jobs yet.")); return; }
    jobs.slice(0, 8).forEach(function (j) {
      var row = el("div", "jrow");
      row.appendChild(el("span", "jtype", j.type));
      row.appendChild(el("span", "jstatus " + j.status, j.status));
      row.appendChild(el("span", "jdetail muted", j.detail || ""));
      box.appendChild(row);
    });
  }
  function showJobProgress(jobId) {
    api("GET", "/api/jobs/" + jobId).then(function (j) {
      var prog = document.getElementById("scr-prog"), log = document.getElementById("scr-log");
      if (j.status === "running" && j.total) {
        prog.classList.remove("hidden");
        var pct = Math.round((j.progress / j.total) * 100);
        prog.querySelector("i").style.width = pct + "%";
        prog.querySelector(".proglabel").textContent = j.detail || "";
      } else { prog.classList.add("hidden"); }
      if (j.log) { log.classList.remove("hidden"); log.textContent = j.log.trim().split("\n").slice(-6).join("\n"); }
      else { log.classList.add("hidden"); }
    }).catch(function () {});
  }
  function hideJobProgress() {
    var p = document.getElementById("scr-prog"), l = document.getElementById("scr-log");
    if (p) p.classList.add("hidden"); if (l) l.classList.add("hidden");
  }

  // ---- toast ----
  function toast(msg, err) {
    var t = document.getElementById("toast");
    if (!t) { t = el("div", "toast"); t.id = "toast"; document.body.appendChild(t); }
    t.textContent = msg; t.className = "toast show" + (err ? " err" : "");
    clearTimeout(toast._t); toast._t = setTimeout(function () { t.className = "toast"; }, 3200);
  }

  // ============================ read-only view (data + chart) ============================
  function renderReadonly() {
    var host = document.getElementById("readonly") || document.getElementById("app");
    host.innerHTML = "";
    var d = state.data; if (!d) return;
    updateSourceLine(d);
    if (!d.players.length) {
      host.appendChild(el("div", "empty",
        "No price data yet. Add player links above and hit <b>Scrape all</b>."));
      return;
    }
    if (!state.selected || !d.players.some(function (p) { return p.id === state.selected; }))
      state.selected = d.players[0].id;
    host.appendChild(playerTable(d));
    host.appendChild(detail());
    requestAnimationFrame(drawChart);
  }
  function updateSourceLine(d) {
    var s = document.querySelector(".hdr .id-src");
    if (s) s.textContent = "Source " + d.source + " · " + d.platform + " · " +
      d.summary.snapshots.toLocaleString() + " snapshots · " +
      d.summary.buys + " BUY / " + d.summary.sells + " SELL / " + d.summary.holds + " HOLD";
  }

  function playerTable(d) {
    var wrap = el("div", "tablewrap");
    var t = el("table", "players");
    t.innerHTML = "<thead><tr><th>Rtg</th><th>Player</th><th class='r'>Price</th>" +
      "<th class='r'>24h</th><th class='r'>Signal</th></tr></thead>";
    var body = el("tbody");
    d.players.forEach(function (p) {
      var tr = el("tr", p.id === state.selected ? "sel" : ""); tr.dataset.id = p.id;
      var up = (p.pct_change_24h || 0) >= 0;
      tr.innerHTML =
        "<td class='rtg num'>" + (p.rating != null ? p.rating : "–") + "</td>" +
        "<td class='pname'>" + p.name + "</td>" +
        "<td class='r num'>" + fmtCoins(p.price, true) + "</td>" +
        "<td class='r num delta " + (up ? "up" : "down") + "'>" + fmtPct(p.pct_change_24h) + "</td>" +
        "<td class='r'><span class='badge " + p.signal.type + "'>" + p.signal.type + "</span></td>";
      tr.onclick = function () { select(p.id); };
      body.appendChild(tr);
    });
    t.appendChild(body); wrap.appendChild(t);
    return wrap;
  }
  function detail() {
    var p = current();
    var card = el("div", "detail"); card.id = "detail";
    var head = el("div", "dhead");
    head.appendChild(el("div", "dtitle", "<b>" + p.name + "</b><span class='dmeta'>price · last " + spanDays(p) + "</span>"));
    head.appendChild(el("span", "badge " + p.signal.type, p.signal.type));
    card.appendChild(head);
    var wrap = el("div", "chartwrap"); var cv = el("canvas"); wrap.appendChild(cv); card.__cv = cv;
    card.appendChild(wrap);
    card.appendChild(el("div", "why", "<b>Why:</b> " + p.signal.reason + "."));
    return card;
  }
  function spanDays(p) {
    var s = p.series; if (s.length < 2) return "–";
    var days = Math.round((parseT(s[s.length - 1].t) - parseT(s[0].t)) / 864e5);
    return Math.max(1, days) + "d";
  }
  function select(id) {
    state.selected = id;
    var rows = document.querySelectorAll("table.players tbody tr");
    Array.prototype.forEach.call(rows, function (r) { r.classList.toggle("sel", r.dataset.id === id); });
    var old = document.getElementById("detail");
    if (old) { var fresh = detail(); old.parentNode.replaceChild(fresh, old); requestAnimationFrame(drawChart); }
  }
  function current() { return state.data.players.filter(function (p) { return p.id === state.selected; })[0]; }

  function drawChart() {
    var card = document.getElementById("detail"); if (!card || !state.data) return;
    var cv = card.__cv, series = current().series;
    var dpr = window.devicePixelRatio || 1, W = cv.parentElement.clientWidth, H = 240;
    cv.style.width = "100%"; cv.style.height = H + "px";
    cv.width = Math.max(1, Math.round(W * dpr)); cv.height = Math.round(H * dpr);
    var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    if (series.length < 2) return;
    var padL = 8, padR = 56, padT = 14, padB = 24;
    var xs = series.map(function (p) { return parseT(p.t).getTime(); });
    var prices = series.map(function (p) { return p.price; });
    var minT = xs[0], maxT = xs[xs.length - 1];
    var lo = Math.min.apply(null, prices), hi = Math.max.apply(null, prices);
    var pad = (hi - lo) * 0.12 || hi * 0.05; lo = Math.max(0, lo - pad); hi += pad;
    function X(t) { return padL + (t - minT) / (maxT - minT || 1) * (W - padL - padR); }
    function Y(v) { return padT + (1 - (v - lo) / (hi - lo || 1)) * (H - padT - padB); }
    ctx.strokeStyle = cssVar("--grid"); ctx.fillStyle = cssVar("--muted");
    ctx.font = "11px " + MONO; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.lineWidth = 1;
    for (var g = 0; g <= 3; g++) {
      var v = lo + (hi - lo) * g / 3, y = Y(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(fmtCoins(v, true), W - padR + 8, y);
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(fmtDay(new Date(minT)), X(minT) + 12, H - padB + 7);
    ctx.fillText(fmtDay(new Date(maxT)), X(maxT) - 12, H - padB + 7);
    var grad = ctx.createLinearGradient(0, padT, 0, H - padB);
    grad.addColorStop(0, cssVar("--price-fill")); grad.addColorStop(1, "transparent");
    ctx.beginPath(); ctx.moveTo(X(xs[0]), Y(prices[0]));
    for (var k = 1; k < series.length; k++) ctx.lineTo(X(xs[k]), Y(prices[k]));
    ctx.lineTo(X(xs[xs.length - 1]), H - padB); ctx.lineTo(X(xs[0]), H - padB); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    ctx.strokeStyle = cssVar("--price"); ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.moveTo(X(xs[0]), Y(prices[0]));
    for (var m = 1; m < series.length; m++) ctx.lineTo(X(xs[m]), Y(prices[m]));
    ctx.stroke();
    var lx = X(xs[xs.length - 1]), ly = Y(prices[prices.length - 1]);
    ctx.fillStyle = cssVar("--surface"); ctx.beginPath(); ctx.arc(lx, ly, 5, 0, 7); ctx.fill();
    ctx.fillStyle = cssVar("--price"); ctx.beginPath(); ctx.arc(lx, ly, 3.2, 0, 7); ctx.fill();
  }

  // static Artifact-preview entry point (no backend)
  function render(data) {
    state.data = data;
    var app = document.getElementById("app"); app.innerHTML = "";
    app.appendChild(header());
    app.appendChild(el("div", null, "")).id = "readonly";
    renderReadonly();
    window.addEventListener("resize", debounce(drawChart, 120));
  }

  function debounce(fn, ms) { var t; return function () { clearTimeout(t); t = setTimeout(fn, ms); }; }

  window.FUT = { init: init, render: render };
})();
