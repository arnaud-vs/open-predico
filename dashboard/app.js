// open-predico dashboard. Static: draws the JSON that build.py writes to data/. No framework.
"use strict";

const DATA = "data/";
const STEP = 900;
const MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(" ");
const WEEKDAYS = "Mon Tue Wed Thu Fri Sat Sun".split(" ");
const LABEL = { energy: "Energy", price: "Price", wind: "Wind", solar: "Solar" };
const cache = {};
const json = p =>
  (cache[p] ??= fetch(DATA + p).then(r => {
    if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
    return r.json();
  }));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pad = v => String(v).padStart(2, "0");
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const minus = s => String(s).replace(/^-/, "−");
const f1 = v => (v == null || Number.isNaN(v) ? "–" : minus(v.toFixed(1)));
const f3 = v => (v == null || Number.isNaN(v) ? "–" : v.toFixed(3));
const pct = v => (v == null || Number.isNaN(v) ? "–" : `${Math.round(v * 100)}%`);
const mean = a => { let s = 0, n = 0; for (const v of a) if (v != null && !Number.isNaN(v)) { s += v; n++; } return n ? s / n : null; };
const fmtUTC = s => { const d = new Date(s * 1000); return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}, ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`; };
const fmtDay = key => { const [y, m, d] = key.split("-"); return `${+d} ${MON[+m - 1]} ${y}`; };

// Madrid calendar for a UTC second, cached per hour (offsets are whole hours). Berlin shares
// Madrid's offsets, so it serves the German markets too.
const MAD = new Intl.DateTimeFormat("en-CA", { timeZone: "Europe/Madrid", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23", weekday: "short" });
const hourCache = new Map();
function madrid(sec) {
  const h = Math.floor(sec / 3600);
  let v = hourCache.get(h);
  if (!v) {
    const p = Object.fromEntries(MAD.formatToParts(new Date(h * 3600e3)).map(x => [x.type, x.value]));
    v = { day: `${p.year}-${p.month}-${p.day}`, month: `${p.year}-${p.month}`, hour: +p.hour, wd: WEEKDAYS.indexOf(p.weekday) };
    hourCache.set(h, v);
  }
  return v;
}

// --- Charts: a registry so resizes and theme switches redraw every chart --------------------
const charts = new Set();
function mount(el, build, height) {
  const h = () => (typeof height === "function" ? height(el.clientWidth) : height);
  const c = { el, build: () => build(el.clientWidth || 600, h()), inst: null, h };
  c.inst = c.build();
  charts.add(c);
  new ResizeObserver(() => {
    if (!document.body.contains(el)) { c.inst?.destroy(); charts.delete(c); return; }
    if (el.clientWidth && c.inst) c.inst.setSize({ width: el.clientWidth, height: c.h() });
  }).observe(el);
  return c;
}
function redrawAll() {
  for (const c of charts) {
    if (!document.body.contains(c.el)) { charts.delete(c); continue; }
    const x = c.inst?.scales?.x;
    const keep = x && { min: x.min, max: x.max };
    c.inst?.destroy();
    c.inst = c.build();
    if (keep && c.keepX) c.inst.setScale("x", keep);
  }
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
new MutationObserver(redrawAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

function axisBase() {
  return {
    stroke: css("--ink-soft"), font: `11px ${css("--mono")}`, labelFont: `12px ${css("--sans")}`,
    grid: { stroke: css("--grid"), width: 1 }, ticks: { stroke: css("--grid"), width: 1, size: 4 },
  };
}
function timeValues(u, splits, ax, space, incr) {
  return splits.map(s => {
    const d = new Date(s * 1000), date = `${pad(d.getUTCDate())}-${pad(d.getUTCMonth() + 1)}`;
    if (incr >= 86400 * 25) return `${MON[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
    if (incr >= 86400) return date;
    const hm = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
    return d.getUTCHours() === 0 && d.getUTCMinutes() === 0 ? `${hm}\n${date}` : hm;
  });
}
const yValues = (u, s) => s.map(v => minus(Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-GB") : +v.toFixed(2)));
function axes(yLabel, xValues = timeValues, xOpts = {}) {
  const a = axisBase();
  return [{ ...a, values: xValues, space: 64, size: 44, ...xOpts }, { ...a, label: yLabel, labelSize: yLabel ? 20 : 0, size: 58, values: yValues }];
}
const tzUTC = ts => uPlot.tzDate(new Date(ts * 1000), "Etc/UTC");

function tooltip(rows) {
  let tt;
  return {
    hooks: {
      init: u => { tt = document.createElement("div"); tt.className = "tt"; tt.hidden = true; u.over.appendChild(tt); },
      setCursor: u => {
        const { left, top, idx } = u.cursor;
        const html = idx == null || left < 0 ? null : rows(u, idx);
        if (!html) { tt.hidden = true; return; }
        tt.innerHTML = html;
        tt.hidden = false;
        const w = tt.offsetWidth, h = tt.offsetHeight, W = u.over.clientWidth, H = u.over.clientHeight;
        const x = left + 16 + w > W ? Math.max(0, left - 16 - w) : left + 16;
        tt.style.transform = `translate(${x}px, ${Math.min(Math.max(0, top - h / 2), Math.max(0, H - h))}px)`;
      },
    },
  };
}
const ttHTML = (title, rows) =>
  `<div class="t">${esc(title)}</div>` +
  rows.map(([c, k, v, dash]) => `<div class="r"><i style="background:${c};${dash ? "height:0;border-top:3px dotted " + c + ";background:none" : ""}"></i><span>${esc(k)}</span><span>${v}</span></div>`).join("");
// Default tooltip: every visible series with a `tt` label.
const seriesRows = fmtX => (u, idx) => {
  const rows = [];
  u.series.forEach((s, i) => { if (i && s.show && s.tt) { const v = u.data[i][idx]; if (v != null) rows.push([s._color, s.tt, f1(v)]); } });
  return rows.length ? ttHTML(fmtX(u.data[0][idx]), rows) : null;
};
function line(label, color, opts = {}) {
  return { label, tt: label, stroke: color, _color: color, width: 2, points: { show: false }, ...opts };
}
function nowMarker(nowSec) {
  return {
    hooks: {
      drawAxes: u => {
        const { left, top, width, height } = u.bbox, x = Math.max(u.valToPos(nowSec, "x", true), left);
        if (x >= left + width) return;
        u.ctx.save(); u.ctx.fillStyle = css("--future"); u.ctx.fillRect(x, top, left + width - x, height); u.ctx.restore();
      },
      draw: u => {
        const { left, top, width, height } = u.bbox, x = u.valToPos(nowSec, "x", true), dpr = devicePixelRatio || 1;
        if (x < left || x > left + width) return;
        const ctx = u.ctx; ctx.save();
        ctx.strokeStyle = css("--ink-soft"); ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.lineWidth = dpr;
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
        const d = new Date(nowSec * 1000);
        ctx.fillStyle = css("--ink-soft"); ctx.font = `${10 * dpr}px ${css("--mono")}`;
        ctx.fillText(`NOW ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`, x + 6 * dpr, top + 12 * dpr);
        ctx.restore();
      },
    },
  };
}
const chartHeight = w => Math.max(280, Math.min(440, w * 0.45));

// UI bits
function seg(id, options, value) {
  return `<div class="seg" role="group" id="${id}">${options.map(([v, l]) => `<button data-v="${v}" aria-pressed="${v === value}">${esc(l)}</button>`).join("")}</div>`;
}
function onSeg(root, id, fn) {
  const g = root.querySelector(`#${id}`);
  g?.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    g.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
    fn(b.dataset.v);
  });
}
// The selected market (MARKETS, below) and its targets' short names (build.py SHORT).
const mk = () => MARKETS.find(m => m.id === market);
const shortsOf = (m, meta) => m.resources.map(r => meta.short[r]);
const targetSeg = (id, shorts, value) => (shorts.length > 1 ? seg(id, shorts.map(t => [t, LABEL[t]]), value) : "");
const kpi = (k, v, d = "", cls = "") => `<div class="kpi"><span class="k">${esc(k)}</span><span class="v">${v}</span><span class="d ${cls}">${d}</span></div>`;
const relClass = r => (r == null || Math.abs(r - 1) < 5e-4 ? "" : r < 1 ? "good" : "bad");
const relText = r => (r == null ? "no baseline" : `${f3(r)} × baseline`);

// Colours per model family, dash per variant: the same config always looks the same.
const FAMILY = [["slot_climatology", "--ink-soft"], ["t0", "--accent-ink"], ["chronos", "#3b6fd1"], ["timesfm", "#8a63d2"], ["timestral", "#d08a1e"]];
const DASHES = [[], [7, 4], [2, 3], [10, 3, 2, 3], [4, 4], [1, 2]];
let CONFIGS = [];
function configStyle(name) {
  const fam = FAMILY.find(([p]) => name.startsWith(p));
  const color = fam ? (fam[1].startsWith("--") ? css(fam[1]) : fam[1]) : "#c2456b";
  const sibs = CONFIGS.filter(c => (fam ? c.startsWith(fam[0]) : !FAMILY.some(([p]) => c.startsWith(p))));
  return { color, dash: DASHES[Math.max(0, sibs.indexOf(name)) % DASHES.length] };
}

// --- Daily scores (same formulas as src/openpredico/evaluation/metrics.py) ------------------
function dailyScores(x, q10, q50, q90, obs) {
  const days = new Map();
  for (let i = 0; i < x.length; i++) {
    const o = obs[i], m = q50[i];
    if (o == null || m == null) continue;
    const key = madrid(x[i]).day;
    let d = days.get(key);
    if (!d) days.set(key, (d = { key, t: x[i], n: 0, se: 0, ae: 0, wk: 0, cov: 0 }));
    d.n++; d.se += (o - m) ** 2; d.ae += Math.abs(o - m);
    d.wk += q90[i] - q10[i] + 10 * Math.max(0, q10[i] - o) + 10 * Math.max(0, o - q90[i]);
    if (o >= q10[i] && o <= q90[i]) d.cov++;
  }
  return [...days.values()].map(d => ({ key: d.key, t: d.t, n: d.n, rmse: Math.sqrt(d.se / d.n), mae: d.ae / d.n, winkler: d.wk / d.n, coverage: d.cov / d.n }));
}

// --- Live --------------------------------------------------------------------------------
// q*: our continuous forecast as it stood at the gate. s*: what Predico registered as our session
// submission (the values it scores). b*: slot climatology from the same run's inputs.
async function renderLive(root) {
  const [live, meta] = await Promise.all([json("live.json"), json("meta.json")]);
  const m = mk(), shorts = shortsOf(m, meta);
  const head = `<div class="head"><div><h1>Live forecasts</h1><p class="lede">What we sent to Predico for ${esc(m.name)}, published only after each gate (${m.gateLocal}), next to the measured values from ${m.source}.</p></div>`;
  if (!live.available || !shorts.some(s => live.targets[s])) {
    root.innerHTML = `${head}</div><div class="card"><p class="empty">No live forecasts are published yet. The first ones appear here half an hour after the next gate.</p></div>`;
    return;
  }
  let target = shorts[0];
  root.innerHTML = `${head}<div class="controls">${targetSeg("lv-t", shorts, target)}</div></div><div id="lv-body" style="display:grid;gap:18px"></div>`;
  onSeg(root, "lv-t", v => { target = v; draw(); });
  draw();

  function draw() {
    const T = live.targets[target], unit = meta.units[target];
    const n = T.q50.length, x = Array.from({ length: n }, (_, i) => T.start + i * STEP);
    const nul = () => new Array(n).fill(null);
    const [s10, s50, s90] = ["s10", "s50", "s90"].map(k => T[k] ?? nul());
    // Scored = Predico's registered values where it has them, else our continuous forecast.
    const pick = (s, q) => s.map((v, i) => v ?? q[i]);
    const e10 = pick(s10, T.q10), e50 = pick(s50, T.q50), e90 = pick(s90, T.q90);
    const model = dailyScores(x, e10, e50, e90, T.obs);
    const base = new Map(dailyScores(x, T.b10, T.b50, T.b90, T.obs).map(d => [d.key, d]));
    // Day record: registered by Predico, continuous forecast only, or missed.
    const dayInfo = new Map();
    for (let i = 0; i < n; i++) {
      const key = madrid(x[i]).day;
      let d = dayInfo.get(key);
      if (!d) dayInfo.set(key, (d = { key, i0: i, i1: i, rows: 0, sub: 0 }));
      d.i1 = i;
      if (e50[i] != null) d.rows++;
      if (s50[i] != null) d.sub++;
    }
    const record = [...dayInfo.values()];
    const counts = { sub: 0, cont: 0, miss: 0 };
    for (const d of record) d.state = d.rows === 0 ? "miss" : d.sub ? "sub" : "cont";
    record.forEach(d => counts[d.state]++);
    const STATE = { sub: "submitted", cont: "not confirmed", miss: "missed" };
    const envNote = live.demo
      ? `<div class="banner"><b>Demo data.</b> No live forecasts are published yet, so this shows the last 21 backtest days of <span class="mono">${esc(T.model)}</span> in the live format.</div>`
      : T.env === "playground"
        ? `<div class="banner"><b>Qualifying on the Predico playground.</b> Production submissions open after five good days; scores below are playground runs.</div>`
        : "";
    const body = root.querySelector("#lv-body");
    body.innerHTML = `${envNote}
      <div class="card">
        <div class="card-head"><div><h2>Forecast vs measured</h2><p class="sub">Solid: what Predico registered as our submission. Dotted on top: our continuous forecast. ${unit} · times in UTC</p></div>
        <div class="controls">${seg("lv-r", [["7", "7 days"], ["30", "30 days"], ["all", "All"]], "7")}</div></div>
        <div class="chart" id="lv-chart"></div>
        <div class="foot"><div class="legend" id="lv-legend">
          <button data-s="6" aria-pressed="true"><span class="sw" style="border-color:var(--obs)"></span>Observed (${m.source})</button>
          <button data-s="3" aria-pressed="true"><span class="sw thick" style="border-color:var(--sub)"></span>Submitted forecast</button>
          <button data-s="4" aria-pressed="true"><span class="sw dot" style="border-color:var(--cont)"></span>Continuous forecast</button>
          <button data-s="1,2" aria-pressed="true"><span class="sw box"></span>Interval (q10–q90)</button>
          <button data-s="5" aria-pressed="false"><span class="sw dash" style="border-color:var(--base)"></span>Climatology baseline</button>
        </div><span class="hint">drag to zoom · double-click to reset · scores follow the view</span></div>
      </div>
      <div class="card"><div class="card-head"><div><h2>Scores against climatology</h2><p class="sub" id="lv-span"></p></div></div>
        <div class="tablewrap" id="lv-sum"></div>
        <p class="sub">Climatology is the slot climatology of the last 28 days, built from the same run's inputs. Ratio = our score ÷ climatology on the days both have; below 1 beats it. Coverage target is 80%. Lower is better for every other metric.</p>
      </div>
      <div class="card"><div class="card-head"><div><h2>Predico's official scores</h2><p class="sub">How Predico scored our submissions and where we ranked among all forecasters on that challenge. Only our own results are shown.</p></div></div>
        <div id="lv-off"></div></div>
      <div class="grid2">
        <div class="card"><div class="card-head"><div><h2>Submission record</h2><p class="sub">One square per target day (${m.city}). Click a day to show it in the chart.</p></div></div>
          <div class="days" id="lv-days">${record.map(d => `<button class="day ${d.state}" data-k="${d.key}" title="${fmtDay(d.key)}: ${STATE[d.state]}">${+d.key.slice(8)}</button>`).join("")}</div>
          <div class="legend"><span class="chip static"><b class="good">■</b> submitted · ${counts.sub}</span><span class="chip static">⋯ not confirmed · ${counts.cont}</span><span class="chip static">▨ missed · ${counts.miss}</span></div>
        </div>
        <div class="card"><div class="card-head"><div><h2>Model and inputs</h2><p class="sub">Latest run ${esc(T.last_run.slice(0, 16).replace("T", " "))} UTC · ${T.runs} run${T.runs === 1 ? "" : "s"} published</p></div><span class="pill ${T.env === "production" ? "" : "warn"}">${esc(T.env)}</span></div>
          <p><span class="mono">${esc(T.model)}</span>: t0-gamma (The Forecasting Company, Apache-2.0), zero-shot, 90 days of context, with covariates. D+2 to D+10 fall back to slot climatology. <a href="#backtest">Backtest results</a></p>
          <div class="chips">${(T.features.length ? T.features : ["(feature list arrives with the first published run)"]).map(f => `<span class="chip static">${esc(f)}</span>`).join("")}</div>
        </div>
      </div>
      <div class="card"><div class="card-head"><div><h2>Daily scores</h2><p class="sub">Every target day, newest first. Click a row to show it in the chart.</p></div></div>
        <div class="tablewrap" style="max-height:520px" id="lv-daily"></div></div>`;

    const data = [x, e10, e90, s50, T.q50, T.b50, T.obs];
    const span = x[n - 1] - x[0];
    let range = "7";
    const setRange = r => {
      range = r;
      const max = x[n - 1] + STEP, min = r === "all" ? x[0] : Math.max(x[0], max - +r * 86400);
      main.inst.setScale("x", { min, max });
    };
    const shown = [true, true, true, true, true, false, true];
    const main = mount(root.querySelector("#lv-chart"), (w, h) => new uPlot({
      width: w, height: h, tzDate: tzUTC, legend: { show: false },
      scales: { x: { time: true } },
      cursor: { drag: { x: true, y: false }, points: { size: 7, width: 2, fill: css("--surface") }, bind: { dblclick: () => () => { setRange(range); return null; } } },
      series: [{},
        { stroke: css("--accent"), width: 1, dash: [2, 3], points: { show: false } },
        { stroke: css("--accent"), width: 1, dash: [2, 3], points: { show: false } },
        line("Submitted", css("--sub"), { width: 6 }),
        line("Continuous", css("--cont"), { width: 2, dash: [2, 4], cap: "round" }),
        line("Climatology", css("--base"), { width: 1.5, dash: [6, 4] }),
        line("Observed", css("--obs"), { width: 2 }),
      ].map((s, i) => (i ? { ...s, show: shown[i] } : s)),
      bands: [{ series: [2, 1], fill: css("--band") }],
      axes: axes(`${LABEL[target]} (${unit})`),
      hooks: { setScale: [(u, k) => k === "x" && summary(u.scales.x.min, u.scales.x.max)] },
      plugins: [nowMarker(Date.now() / 1000), tooltip((u, i) => {
        const rows = [], on = k => u.series[k].show;
        if (on(6) && T.obs[i] != null) rows.push([css("--obs"), "Observed", `${f1(T.obs[i])} ${unit}`]);
        if (on(3) && s50[i] != null) rows.push([css("--sub"), "Submitted", `${f1(s50[i])} ${unit}`]);
        if (on(4) && T.q50[i] != null) rows.push([css("--cont"), "Continuous", `${f1(T.q50[i])} ${unit}`, true]);
        if (on(1) && e10[i] != null) rows.push([css("--accent"), "q10–q90", `${f1(e10[i])} … ${f1(e90[i])}`]);
        if (on(5) && T.b50[i] != null) rows.push([css("--base"), "Climatology", `${f1(T.b50[i])} ${unit}`]);
        return rows.length ? ttHTML(fmtUTC(x[i]), rows) : null;
      })],
    }, data, root.querySelector("#lv-chart")), chartHeight);
    main.keepX = true;
    setRange(span < 7 * 86400 ? "all" : "7");
    onSeg(root, "lv-r", setRange);
    root.querySelector("#lv-legend").addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      const on = b.getAttribute("aria-pressed") !== "true";
      b.setAttribute("aria-pressed", String(on));
      b.dataset.s.split(",").forEach(i => { shown[+i] = on; main.inst.setSeries(+i, { show: on }); });
    });
    const zoomDay = k => { const d = dayInfo.get(k); main.inst.setScale("x", { min: x[d.i0], max: x[d.i1] + STEP }); };
    root.querySelector("#lv-days").addEventListener("click", e => { const b = e.target.closest("button"); if (b) zoomDay(b.dataset.k); });

    // Summary table: the days in the chart's view; ratios only on days with a baseline.
    const METRICS = [["rmse", "RMSE (q50)", f1, unit], ["mae", "MAE (q50)", f1, unit], ["winkler", "Winkler (q10–q90)", f1, unit], ["coverage", "Coverage (q10–q90)", pct, "target 80%"]];
    const covClass = c => (c == null ? "" : Math.abs(c - 0.8) <= 0.05 ? "good" : "warn");
    function summary(lo, hi) {
      const inView = model.filter(d => d.t >= lo - 3600 && d.t < hi), both = inView.filter(d => base.has(d.key));
      root.querySelector("#lv-span").textContent = inView.length
        ? `${inView.length} scored day${inView.length === 1 ? "" : "s"} in view: ${fmtDay(inView[0].key)} – ${fmtDay(inView[inView.length - 1].key)}`
        : "No measured days in view yet.";
      root.querySelector("#lv-sum").innerHTML = `<table><thead><tr><th>Metric</th><th class="n">Our forecast</th><th class="n">Climatology</th><th class="n">Ratio</th><th>Unit</th></tr></thead><tbody>${METRICS.map(([m, label, f, u]) => {
        const ours = mean(inView.map(d => d[m])), clim = mean(both.map(d => base.get(d.key)[m]));
        const r = m === "coverage" || !both.length ? null : mean(both.map(d => d[m])) / clim;
        return `<tr><td>${label}</td><td class="n ${m === "coverage" ? covClass(ours) : ""}"><b>${f(ours)}</b></td><td class="n muted">${f(clim)}</td><td class="n ${relClass(r)}">${r == null ? "–" : f3(r)}</td><td class="muted">${u}</td></tr>`;
      }).join("")}</tbody></table>`;
    }
    summary(main.inst.scales.x.min, main.inst.scales.x.max);

    // Official scores: one column per variable and metric, value with our rank among N.
    const off = T.official ?? [], offEl = root.querySelector("#lv-off");
    if (!off.length) offEl.innerHTML = `<p class="empty">No official scores yet. Predico scores a day once its measurements are in, usually a day or two after delivery.</p>`;
    else {
      const cols = [...new Set(off.map(r => `${r.variable} ${r.metric}`))].sort();
      const byDay = new Map();
      for (const r of off) (byDay.get(r.day) ?? byDay.set(r.day, {}).get(r.day))[`${r.variable} ${r.metric}`] = r;
      const days = [...byDay.keys()].sort().reverse();
      const rankCell = r => (r ? `<td class="n ${r.rank <= 3 ? `m${r.rank}` : ""}">${f1(r.value)} <span class="muted">·</span> ${r.rank <= 3 ? `<span class="medal">${r.rank}</span>` : r.rank}<span class="muted">/${r.participants}</span></td>` : `<td class="n muted">–</td>`);
      const latest = byDay.get(days[0]);
      offEl.innerHTML = `<div class="kpis">${cols.map(c => { const r = latest[c]; return kpi(`${c} · ${fmtDay(days[0])}`, r ? `${r.rank}<span class="muted" style="font-size:14px"> / ${r.participants}</span>` : "–", r ? `score ${f1(r.value)}` : ""); }).join("")}</div>
        <div class="tablewrap" style="max-height:420px"><table><thead><tr><th>Day</th>${cols.map(c => `<th class="n">${esc(c)} <span class="muted">· rank</span></th>`).join("")}</tr></thead><tbody>${days.map(d => `<tr><td class="mono">${fmtDay(d)}</td>${cols.map(c => rankCell(byDay.get(d)[c])).join("")}</tr>`).join("")}</tbody></table></div>`;
    }

    // Daily table, newest first; days without measurements still show their status.
    const scoredDay = new Map(model.map(d => [d.key, d]));
    const cell = (d, b, m, f) => {
      const v = d?.[m], c = b?.[m], r = m === "coverage" || v == null || c == null ? null : v / c;
      return `<td class="n ${m === "coverage" ? covClass(v) : ""}">${f(v)}</td><td class="n muted">${f(c)}</td>${m === "coverage" ? "" : `<td class="n ${relClass(r)}">${r == null ? "–" : f3(r)}</td>`}`;
    };
    root.querySelector("#lv-daily").innerHTML = `<table><thead>
      <tr><th rowspan="2">Day</th><th rowspan="2">Status</th><th class="n" colspan="3">RMSE (${unit})</th><th class="n" colspan="3">MAE (${unit})</th><th class="n" colspan="3">Winkler (${unit})</th><th class="n" colspan="2">Coverage</th></tr>
      <tr>${["ours", "clim.", "ratio", "ours", "clim.", "ratio", "ours", "clim.", "ratio", "ours", "clim."].map(h => `<th class="n">${h}</th>`).join("")}</tr></thead><tbody>${[...record].reverse().map(r => {
        const d = scoredDay.get(r.key), b = base.get(r.key);
        return `<tr class="click" data-k="${r.key}"><td class="mono">${fmtDay(r.key)}</td><td><span class="pill ${r.state === "sub" ? "" : r.state === "cont" ? "warn" : "muted"}">${STATE[r.state]}</span></td>${d ? cell(d, b, "rmse", f1) + cell(d, b, "mae", f1) + cell(d, b, "winkler", f1) + cell(d, b, "coverage", pct) : `<td class="muted" colspan="11">${r.state === "miss" ? "no forecast" : "awaiting measurements"}</td>`}</tr>`;
      }).join("")}</tbody></table>`;
    root.querySelector("#lv-daily").addEventListener("click", e => { const tr = e.target.closest("tr[data-k]"); if (tr) zoomDay(tr.dataset.k); });
  }
}

// --- Backtest ----------------------------------------------------------------------------
function leaderboard(days, baseline) {
  const rows = {};
  for (const [t, d] of Object.entries(days)) {
    for (const [c, s] of Object.entries(d.configs)) {
      const r = (rows[c] ??= { config: c });
      for (const m of ["rmse", "winkler"]) {
        const a = s[m], b = d.configs[baseline][m];
        let sa = 0, sb = 0, nb = 0;
        for (let i = 0; i < a.length; i++) if (a[i] != null && b[i] != null) { sa += a[i]; sb += b[i]; nb++; }
        r[`${t} ${m}`] = mean(a);
        r[`${t} ${m} rel`] = nb ? sa / sb : null;
      }
      r[`${t} coverage`] = mean(s.coverage);
      r[`${t} days`] = s.rmse.filter(v => v != null).length;
    }
  }
  const out = Object.values(rows);
  for (const r of out) {
    const rel = Object.keys(r).filter(k => k.endsWith(" rel")).map(k => r[k]);
    r.score = rel.every(v => v) ? Math.exp(mean(rel.map(Math.log))) : null;
  }
  out.sort((a, b) => (a.score ?? 9) - (b.score ?? 9));
  out.forEach((r, i) => (r.rank = r.score == null ? null : i + 1));
  return out;
}

async function renderBacktest(root) {
  const [allDays, meta, prof] = await Promise.all([json("days.json"), json("meta.json"), json("profiles.json")]);
  const m = mk(), targets = shortsOf(m, meta).filter(t => allDays[t]);
  if (!targets.length) {
    root.innerHTML = `<div class="head"><div><h1>Backtest</h1></div></div><div class="card"><p class="empty">No backtest of ${esc(m.name)} is published yet.</p></div>`;
    return;
  }
  const days = Object.fromEntries(targets.map(t => [t, allDays[t]]));
  CONFIGS = meta.configs.map(c => c.name).filter(c => targets.some(t => days[t].configs[c]));
  const type = Object.fromEntries(meta.configs.map(c => [c.name, c.type]));
  const lb = leaderboard(days, meta.baseline), t0 = targets[0];
  const top = lb.filter(r => r.config !== meta.live_config && r.config !== meta.baseline).slice(0, 2).map(r => r.config);
  let sel = [meta.live_config, meta.baseline, ...top].filter(c => CONFIGS.includes(c)), types = new Set(), absolute = false;

  root.innerHTML = `
    <div class="head"><div><h1>Backtest</h1><p class="lede">Test year 1 Nov 2025 – 31 Oct 2026. Each forecast is made at 14:00 ${m.city} on D-1 from data published before then, and scored per local day as Predico does. <b>Lower is better.</b> ${m.missingTruth}</p></div></div>
    <div class="card"><div class="card-head"><div><h2>Leaderboard</h2><p class="sub">Scores relative to <span class="mono">${esc(meta.baseline)}</span> (= 1.000) on the days both are scored. Highlighted: the live config.</p></div>
      <div class="controls"><div class="chips" id="bt-types">${["baseline", "foundation model", "ML"].filter(t => CONFIGS.some(c => type[c] === t)).map(t => `<button class="chip" data-v="${t}" aria-pressed="false">${t}</button>`).join("")}</div>
      <label class="tog"><input type="checkbox" id="bt-abs"> absolute scores</label></div></div>
      <div class="tablewrap" id="bt-lb"></div>
      <details><summary class="sub">How the scores work</summary><ul class="sub">
        <li><b>RMSE</b> (Predico's q50 score): √(mean (obs − q50)²) over a local day.</li>
        <li><b>Winkler</b> (Predico's q10/q90 score, α = 0.2): mean of (q90 − q10) + 10·max(0, q10 − obs) + 10·max(0, obs − q90). Rewards narrow intervals, penalises misses.</li>
        <li><b>Coverage</b>: share of quarter-hours inside [q10, q90]; a calibrated model gets 80%.</li>
        <li><b>Score</b>: geometric mean of the ${targets.length > 1 ? "four normalised values (RMSE and Winkler, both targets)" : "two normalised values (RMSE and Winkler)"}; rank sorts by it.</li></ul></details>
    </div>
    <div class="card"><div class="card-head"><div><h2>Configs in the charts below</h2><p class="sub">Same colour = same model family; dash = variant.</p></div></div><div class="chips" id="bt-sel"></div></div>
    <div class="card"><div class="card-head"><div><h2>Day viewer</h2><p class="sub">q50 per config, interval of the first selected config, truth in black. Times in UTC.</p></div>
      <div class="controls">${targetSeg("dv-t", targets, t0)}<input type="date" id="dv-d" value="${days[t0].days.at(-1)}" min="${days[t0].days[0]}" max="${days[t0].days.at(-1)}"></div></div>
      <p class="sub" id="dv-s"></p><div class="chart" id="dv-chart"></div></div>
    <div class="card"><div class="card-head"><div><h2>By month</h2></div><div class="controls">${targetSeg("mo-t", targets, t0)}${seg("mo-m", [["rmse", "RMSE"], ["winkler", "Winkler"]], "rmse")}</div></div><div class="chart" id="mo-chart"></div></div>
    <div class="card"><div class="card-head"><div><h2>Absolute error of q50 by hour and weekday</h2><p class="sub">${m.city} time.</p></div><div class="controls">${targetSeg("pr-t", targets, t0)}</div></div>
      <div class="grid2"><div class="chart" id="pr-h"></div><div class="chart" id="pr-w"></div></div></div>
    <div class="card"><div class="card-head"><div><h2>Win rate</h2><p class="sub">Cell (row A, column B): share of days scored for both where A has the lower score. Read across a row.</p></div>
      <div class="controls">${targetSeg("wr-t", targets, t0)}${seg("wr-m", [["rmse", "RMSE"], ["winkler", "Winkler"]], "rmse")}</div></div>
      <div class="tablewrap" id="wr"></div></div>`;

  const lbEl = root.querySelector("#bt-lb");
  function drawLb() {
    const rows = lb.filter(r => !types.size || types.has(type[r.config]));
    const cols = targets.flatMap(t => [[`${t} rmse rel`, `${LABEL[t]} RMSE`, f3], [`${t} winkler rel`, `${LABEL[t]} Winkler`, f3], [`${t} coverage`, `${LABEL[t]} cov.`, pct],
      ...(absolute ? [[`${t} rmse`, `${LABEL[t]} RMSE abs`, f1], [`${t} winkler`, `${LABEL[t]} Winkler abs`, f1], [`${t} days`, "days", String]] : [])]);
    lbEl.innerHTML = `<table><thead><tr><th>Rank</th><th>Config</th><th>Type</th><th class="n">Score</th>${cols.map(c => `<th class="n">${c[1]}</th>`).join("")}</tr></thead><tbody>${rows.map(r =>
      `<tr class="${r.config === meta.live_config ? "live" : ""} ${r.rank <= 3 ? `m${r.rank}` : ""}"><td class="n">${r.rank <= 3 ? `<span class="medal" title="${["Gold", "Silver", "Bronze"][r.rank - 1]}">${r.rank}</span>` : r.rank ?? "–"}</td><td class="mono">${esc(r.config)}${r.config === meta.live_config ? ' <span class="pill">live</span>' : ""}</td><td class="muted">${type[r.config]}</td><td class="n ${relClass(r.score)}"><b>${f3(r.score)}</b></td>${cols.map(([k, , f]) => `<td class="n ${k.endsWith("rel") ? relClass(r[k]) : ""}">${f(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  }
  drawLb();
  root.querySelector("#bt-types").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    types.has(b.dataset.v) ? types.delete(b.dataset.v) : types.add(b.dataset.v);
    b.setAttribute("aria-pressed", String(types.has(b.dataset.v)));
    drawLb();
  });
  root.querySelector("#bt-abs").addEventListener("change", e => { absolute = e.target.checked; drawLb(); });

  const selEl = root.querySelector("#bt-sel");
  const drawSel = () => (selEl.innerHTML = lb.map(r => { const st = configStyle(r.config); return `<button class="chip" data-v="${r.config}" aria-pressed="${sel.includes(r.config)}"><span class="sw" style="width:14px;border-color:${st.color};border-top-style:${st.dash.length ? "dashed" : "solid"}"></span> ${esc(r.config)}</button>`; }).join(""));
  drawSel();
  selEl.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    sel = sel.includes(b.dataset.v) ? sel.filter(c => c !== b.dataset.v) : [...sel, b.dataset.v];
    b.setAttribute("aria-pressed", String(sel.includes(b.dataset.v)));
    redrawSel();
  });
  const cfgSeries = c => { const st = configStyle(c); return line(c, st.color, { dash: st.dash, width: c === meta.live_config ? 2.5 : 1.75 }); };

  // By month
  let moT = t0, moM = "rmse";
  const months = t => [...new Set(days[t].days.map(d => d.slice(0, 7)))];
  const mo = mount(root.querySelector("#mo-chart"), (w, h) => {
    const ms = months(moT), x = ms.map((m, i) => i);
    const ys = sel.map(c => ms.map(m => mean(days[moT].days.map((d, i) => (d.startsWith(m) ? days[moT].configs[c]?.[moM][i] : null)))));
    return new uPlot({ width: w, height: h, legend: { show: false }, scales: { x: { time: false } },
      cursor: { points: { size: 6, width: 2, fill: css("--surface") } },
      series: [{}, ...sel.map(c => ({ ...cfgSeries(c), points: { show: true, size: 5 } }))],
      axes: axes(`${moM === "rmse" ? "RMSE" : "Winkler"} (${meta.units[moT]})`, (u, s) => s.map(i => (Number.isInteger(i) && ms[i] ? `${MON[+ms[i].slice(5) - 1]}\n${ms[i].slice(0, 4)}` : "")), { space: 40 }),
      plugins: [tooltip(seriesRows(i => { const m = ms[i]; return `${MON[+m.slice(5) - 1]} ${m.slice(0, 4)}`; }))],
    }, [x, ...ys], root.querySelector("#mo-chart"));
  }, 300);
  onSeg(root, "mo-t", v => { moT = v; rebuild(mo); });
  onSeg(root, "mo-m", v => { moM = v; rebuild(mo); });

  // Profiles
  let prT = t0;
  const prof1 = (el, key, labels) => mount(el, (w, h) => {
    const x = labels.map((_, i) => i);
    return new uPlot({ width: w, height: h, legend: { show: false }, scales: { x: { time: false } }, cursor: { points: { size: 6, width: 2, fill: css("--surface") } },
      series: [{}, ...sel.map(cfgSeries)],
      axes: axes(`MAE (${meta.units[prT]})`, (u, s) => s.map(i => labels[i] ?? ""), { space: 34 }),
      plugins: [tooltip(seriesRows(i => labels[i]))],
    }, [x, ...sel.map(c => prof[prT][c]?.[key] ?? x.map(() => null))], el);
  }, 260);
  const ph = prof1(root.querySelector("#pr-h"), "hour", Array.from({ length: 24 }, (_, h) => `${pad(h)}:00`));
  const pw = prof1(root.querySelector("#pr-w"), "weekday", WEEKDAYS);
  onSeg(root, "pr-t", v => { prT = v; rebuild(ph); rebuild(pw); });

  // Day viewer
  let dvT = t0;
  const target = await json(`target/${market}.json`);
  const off = (meta.start - target.start) / STEP;
  const x = Array.from({ length: meta.n }, (_, i) => meta.start + i * STEP);
  const dayIdx = new Map();
  x.forEach((s, i) => { const k = madrid(s).day; const r = dayIdx.get(k); r ? (r[1] = i) : dayIdx.set(k, [i, i]); });
  const dv = { el: root.querySelector("#dv-chart") };
  async function drawDay() {
    const key = root.querySelector("#dv-d").value, r = dayIdx.get(key);
    if (!r) return;
    const steps = await Promise.all(sel.map(c => json(`steps/${dvT}/${c}.json`)));
    const [i0, i1] = r, sl = a => a.slice(i0, i1 + 1);
    const xs = sl(x), obs = target[dvT].slice(off + i0, off + i1 + 1);
    const k = days[dvT].days.indexOf(key);
    root.querySelector("#dv-s").innerHTML = k < 0 ? "" : sel.map(c => `<span class="mono">${esc(c)}</span> RMSE ${f1(days[dvT].configs[c].rmse[k])}, Winkler ${f1(days[dvT].configs[c].winkler[k])}`).join(" · ");
    const first = steps[0];
    const build = (w, h) => new uPlot({
      width: w, height: h, tzDate: tzUTC, legend: { show: false }, scales: { x: { time: true } },
      cursor: { points: { size: 6, width: 2, fill: css("--surface") } },
      series: [{}, { stroke: "transparent", points: { show: false } }, { stroke: "transparent", points: { show: false } }, ...sel.map(cfgSeries), line("Observed", css("--obs"), { width: 2.25 })],
      bands: first ? [{ series: [2, 1], fill: css("--band") }] : [],
      axes: axes(`${LABEL[dvT]} (${meta.units[dvT]})`),
      plugins: [tooltip(seriesRows(fmtUTC))],
    }, [xs, first ? sl(first.q10) : xs.map(() => null), first ? sl(first.q90) : xs.map(() => null), ...steps.map(s => sl(s.q50)), obs], dv.el);
    if (dv.c) { dv.c.build = () => build(dv.el.clientWidth, chartHeight(dv.el.clientWidth)); rebuild(dv.c); } else dv.c = mount(dv.el, build, chartHeight);
  }
  onSeg(root, "dv-t", v => { dvT = v; drawDay(); });
  root.querySelector("#dv-d").addEventListener("change", drawDay);
  await drawDay();

  function redrawSel() { rebuild(mo); rebuild(ph); rebuild(pw); drawDay(); drawWr(); }

  // Win rate
  let wrT = t0, wrM = "rmse";
  function drawWr() {
    const order = lb.map(r => r.config).filter(c => sel.includes(c));
    const d = days[wrT].configs;
    const cell = (a, b) => { let w = 0, n = 0; const A = d[a][wrM], B = d[b][wrM]; for (let i = 0; i < A.length; i++) if (A[i] != null && B[i] != null) { n++; if (A[i] < B[i]) w++; } return n ? w / n : null; };
    root.querySelector("#wr").innerHTML = order.length < 2 ? `<p class="empty">Select at least two configs above.</p>` :
      `<table><thead><tr><th>A \\ B</th>${order.map(c => `<th class="n mono">${esc(c)}</th>`).join("")}</tr></thead><tbody>${order.map(a => `<tr><td class="mono">${esc(a)}</td>${order.map(b => {
        if (a === b) return `<td></td>`;
        const v = cell(a, b), g = v == null ? 0 : (v - 0.5) * 2;
        const bg = g >= 0 ? `color-mix(in srgb, var(--accent) ${Math.round(g * 70)}%, transparent)` : `color-mix(in srgb, var(--bad) ${Math.round(-g * 55)}%, transparent)`;
        return `<td class="n" style="background:${bg}">${pct(v)}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table>`;
  }
  drawWr();
  onSeg(root, "wr-t", v => { wrT = v; drawWr(); });
  onSeg(root, "wr-m", v => { wrM = v; drawWr(); });
}
function rebuild(c) { c.inst?.destroy(); c.inst = c.build(); }

// --- Target data -------------------------------------------------------------------------
async function renderTarget(root) {
  const [target, meta] = await Promise.all([json(`target/${market}.json`), json("meta.json")]);
  const mkt = mk(), shorts = shortsOf(mkt, meta), scatter = "price" in target;
  const n = target[shorts[0]].length, x = Array.from({ length: n }, (_, i) => target.start + i * STEP);
  const cal = x.map(madrid);
  let t = shorts[0];
  root.innerHTML = `
    <div class="head"><div><h1>Target data</h1><p class="lede">${mkt.about(fmtDay(cal[0].day), fmtDay(cal[n - 1].day))}</p></div>
    <div class="controls">${targetSeg("tg-t", shorts, t)}</div></div>
    <div class="kpis" id="tg-k"></div>
    <div class="card"><div class="card-head"><div><h2>Time series</h2><p class="sub">Every quarter-hour of the history. Times in UTC.</p></div><div class="controls">${seg("tg-r", [["14", "14 days"], ["90", "90 days"], ["all", "All"]], "90")}</div></div>
      <div class="chart" id="tg-ts"></div><div class="foot"><span></span><span class="hint">drag to zoom · double-click to reset</span></div></div>
    <div class="card"><div class="card-head"><div><h2>Typical patterns</h2><p class="sub">Mean (line) and q10–q90 (band) over the whole history, ${mkt.city} time.</p></div></div>
      <div class="grid3"><div class="chart" id="tg-p1"></div><div class="chart" id="tg-p2"></div><div class="chart" id="tg-p3"></div></div></div>
    <div class="grid2">
      <div class="card"><div class="card-head"><div><h2>Distribution</h2><p class="sub">Clipped to the 0.5–99.5% range for display.</p></div></div><div class="chart" id="tg-h"></div></div>
      ${scatter ? `<div class="card"><div class="card-head"><div><h2>Price vs energy</h2><p class="sub">A random sample of 5,000 quarter-hours.</p></div></div><canvas class="scatter" id="tg-sc" role="img" aria-label="Scatter of mFRR price against net mFRR energy"></canvas></div>` : ""}
    </div>
    <div class="card"><div class="card-head"><div><h2>Per day</h2><p class="sub" id="tg-ms"></p></div><div class="controls">${seg("tg-mk", [["miss", "Missing"], ["zero", "Exact zeros"]], "miss")}</div></div>
      <div style="overflow-x:auto"><div class="heat" id="tg-heat"></div></div></div>`;

  let range = "90", kind = "miss";
  const ts = mount(root.querySelector("#tg-ts"), (w, h) => new uPlot({
    width: w, height: h, tzDate: tzUTC, legend: { show: false }, scales: { x: { time: true } },
    cursor: { drag: { x: true, y: false }, points: { size: 6, width: 2, fill: css("--surface") }, bind: { dblclick: () => () => { setR(range); return null; } } },
    series: [{}, line(LABEL[t], css("--accent-ink"), { width: 1.25 })],
    axes: axes(`${LABEL[t]} (${meta.units[t]})`),
    plugins: [tooltip(seriesRows(fmtUTC))],
  }, [x, target[t]], root.querySelector("#tg-ts")), chartHeight);
  ts.keepX = true;
  const setR = r => { range = r; ts.inst.setScale("x", { min: r === "all" ? x[0] : x[n - 1] - +r * 86400, max: x[n - 1] }); };
  onSeg(root, "tg-r", setR);

  function quant(a, q) { if (!a.length) return null; const s = Float64Array.from(a).sort(); const p = (s.length - 1) * q, lo = Math.floor(p); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (p - lo); }
  function profile(keyFn, keys) {
    const g = new Map(keys.map(k => [k, []]));
    target[t].forEach((v, i) => { if (v != null) g.get(keyFn(i))?.push(v); });
    return { mean: keys.map(k => mean(g.get(k))), q10: keys.map(k => quant(g.get(k), 0.1)), q90: keys.map(k => quant(g.get(k), 0.9)) };
  }
  const months = [...new Set(cal.map(c => c.month))];
  const monthIdx = new Map(months.map((m, i) => [m, i]));
  const monthName = i => `${MON[+months[i].slice(5) - 1]} ${months[i].slice(0, 4)}`;
  const PROFILES = [
    ["tg-p1", "Quarter-hour of day", i => cal[i].hour * 4 + Math.floor((x[i] % 3600) / 900), Array.from({ length: 96 }, (_, i) => i), i => `${pad(i / 4)}:00`, i => `${pad(Math.floor(i / 4))}:${pad((i % 4) * 15)}`, 16],
    ["tg-p2", "Weekday", i => cal[i].wd, [0, 1, 2, 3, 4, 5, 6], i => WEEKDAYS[i], i => WEEKDAYS[i], 1],
    ["tg-p3", "Month", i => monthIdx.get(cal[i].month), months.map((_, i) => i), i => `${MON[+months[i].slice(5) - 1]}\n${months[i].slice(2, 4)}`, monthName, 3],
  ];
  const profCharts = PROFILES.map(([id, title, keyFn, keys, tick, ttl, step]) => mount(root.querySelector(`#${id}`), (w, h) => {
    const p = profile(keyFn, keys);
    return new uPlot({ width: w, height: h, title, legend: { show: false }, scales: { x: { time: false } }, cursor: { points: { size: 5, width: 2, fill: css("--surface") } },
      series: [{}, { stroke: css("--accent"), width: 1, points: { show: false } }, { stroke: css("--accent"), width: 1, points: { show: false } }, line("Mean", css("--accent-ink"), { width: 2 })],
      bands: [{ series: [2, 1], fill: css("--band") }],
      axes: axes("", (u, s) => s.map(i => (Number.isInteger(i) && i in keys ? tick(i) : "")), { space: 20, incrs: [step] }),
      plugins: [tooltip((u, i) => ttHTML(ttl(i), [[css("--accent-ink"), "Mean", f1(p.mean[i])], [css("--accent"), "q10–q90", `${f1(p.q10[i])} … ${f1(p.q90[i])}`]]))],
    }, [keys, p.q10, p.q90, p.mean], root.querySelector(`#${id}`));
  }, 230));

  function stats() {
    const v = target[t].filter(a => a != null), s = Float64Array.from(v).sort(), m = mean(v);
    const sd = Math.sqrt(mean(v.map(a => (a - m) ** 2)));
    root.querySelector("#tg-k").innerHTML = kpi("Mean", f1(m), meta.units[t]) + kpi("Std", f1(sd)) +
      kpi("Exact zeros", pct(v.filter(a => a === 0).length / v.length)) + kpi("Negative", pct(v.filter(a => a < 0).length / v.length)) +
      kpi("Min / max", `${minus(Math.round(s[0]))} / ${Math.round(s[s.length - 1])}`);
    return s;
  }
  const hist = mount(root.querySelector("#tg-h"), (w, h) => {
    const s = stats(), lo = quant(s, 0.005), hi = quant(s, 0.995), bins = 80, width = (hi - lo) / bins;
    const counts = new Array(bins).fill(0);
    for (const v of s) counts[Math.min(bins - 1, Math.max(0, Math.floor((Math.min(Math.max(v, lo), hi) - lo) / width)))]++;
    const centers = counts.map((_, i) => lo + (i + 0.5) * width);
    return new uPlot({ width: w, height: h, legend: { show: false }, scales: { x: { time: false } }, cursor: { points: { show: false } },
      series: [{}, { label: "Quarter-hours", stroke: css("--accent-ink"), fill: css("--band"), paths: uPlot.paths.bars({ size: [0.92, 100] }), points: { show: false } }],
      axes: axes("Quarter-hours", (u, sp) => sp.map(v => minus(Math.round(v))), { space: 50 }),
      plugins: [tooltip((u, i) => ttHTML(`${f1(centers[i] - width / 2)} … ${f1(centers[i] + width / 2)}`, [[css("--accent-ink"), "Quarter-hours", counts[i].toLocaleString("en-GB")]]))],
    }, [centers, counts], root.querySelector("#tg-h"));
  }, 280);

  // Scatter on a plain canvas (5,000 points; static). Spain only: price vs energy.
  const scEl = root.querySelector("#tg-sc");
  const pairs = []; if (scatter) for (let i = 0; i < n; i++) if (target.energy[i] != null && target.price[i] != null) pairs.push([target.energy[i], target.price[i]]);
  let seed = 1; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const sample = Array.from({ length: Math.min(5000, pairs.length) }, () => pairs[Math.floor(rnd() * pairs.length)]);
  const drawScatter = w => {
    const dpr = devicePixelRatio || 1, h = 360; scEl.width = w * dpr; scEl.height = h * dpr;
    const ctx = scEl.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const xs = sample.map(p => p[0]), ys = sample.map(p => p[1]);
    const qx = [quant(xs, 0.002), quant(xs, 0.998)], qy = [quant(ys, 0.002), quant(ys, 0.998)];
    const L = 58, B = 40, X = v => L + ((v - qx[0]) / (qx[1] - qx[0])) * (w - L - 12), Y = v => h - B - ((v - qy[0]) / (qy[1] - qy[0])) * (h - B - 10);
    ctx.font = `11px ${css("--mono")}`; ctx.fillStyle = css("--ink-soft"); ctx.strokeStyle = css("--grid");
    for (let k = 0; k <= 4; k++) {
      const vx = qx[0] + ((qx[1] - qx[0]) * k) / 4, vy = qy[0] + ((qy[1] - qy[0]) * k) / 4;
      ctx.beginPath(); ctx.moveTo(X(vx), 10); ctx.lineTo(X(vx), h - B); ctx.stroke(); ctx.fillText(minus(Math.round(vx)), X(vx) - 14, h - B + 16);
      ctx.beginPath(); ctx.moveTo(L, Y(vy)); ctx.lineTo(w - 12, Y(vy)); ctx.stroke(); ctx.fillText(minus(Math.round(vy)), 8, Y(vy) + 4);
    }
    ctx.font = `12px ${css("--sans")}`;
    ctx.fillText("Net mFRR energy (MWh)", L, h - 6);
    ctx.fillStyle = css("--accent-ink"); ctx.globalAlpha = 0.28;
    for (const [a, b] of sample) if (a >= qx[0] && a <= qx[1] && b >= qy[0] && b <= qy[1]) ctx.fillRect(X(a) - 1.2, Y(b) - 1.2, 2.4, 2.4);
    ctx.globalAlpha = 1; ctx.fillStyle = css("--ink-soft"); ctx.fillText("Price (EUR/MWh) ↑", L + 6, 22);
  };
  if (scatter) mount(scEl, w => { drawScatter(w); return { destroy() {}, setSize: ({ width }) => drawScatter(width) }; }, 360);

  function heat() {
    const per = new Map();
    target[t].forEach((v, i) => { const k = cal[i].day; per.set(k, (per.get(k) ?? 0) + (kind === "miss" ? v == null : v === 0)); });
    const max = Math.max(1, ...per.values()), total = [...per.values()].reduce((a, b) => a + b, 0);
    root.querySelector("#tg-ms").textContent = `${kind === "miss" ? "Missing" : "Exactly zero"} quarter-hours per day: ${total.toLocaleString("en-GB")} of ${n.toLocaleString("en-GB")} (${pct(total / n)}).${t === "energy" && kind === "miss" ? " Energy gaps were filled with 0 upstream: use Exact zeros to see them." : ""}`;
    let html = `<span class="lab"></span>${Array.from({ length: 31 }, (_, d) => `<span class="lab" style="text-align:center">${d % 5 === 0 ? d + 1 : ""}</span>`).join("")}`;
    for (const m of months) {
      html += `<span class="lab">${MON[+m.slice(5) - 1]} ${m.slice(2, 4)}</span>`;
      for (let d = 1; d <= 31; d++) {
        const k = `${m}-${pad(d)}`, v = per.get(k);
        html += v == null ? `<div></div>` : `<div title="${fmtDay(k)}: ${v}" style="background:${v ? `color-mix(in srgb, var(--warn) ${Math.round(15 + (85 * v) / max)}%, var(--grid))` : "var(--grid)"}"></div>`;
      }
    }
    root.querySelector("#tg-heat").innerHTML = html;
  }
  heat();
  setR(range);
  onSeg(root, "tg-mk", v => { kind = v; heat(); });
  onSeg(root, "tg-t", v => {
    t = v; heat();
    rebuild(ts); setR(range);
    profCharts.forEach(rebuild); rebuild(hist);
  });
}

// --- Input features ----------------------------------------------------------------------
async function renderInputs(root) {
  const cat = await json("catalog.json");
  const groups = new Set(cat.flatMap(r => (r.group ? r.group.split(", ") : [])));
  const cats = [...new Set(cat.map(r => r.category))].map(c => ({ c, n: cat.filter(r => r.category === c).length, used: cat.filter(r => r.category === c && r.used).length })).sort((a, b) => b.used - a.used || b.n - a.n);
  root.innerHTML = `
    <div class="head"><div><h1>Input features</h1><p class="lede">Every series we could feed a model. ESIOS serves <b>indicators</b> (time series), <b>offer indicators</b> (bid curves) and <b>archives</b> (bulk files). Calendar columns are computed. <b>Used</b> marks series in a feature group of <span class="mono">configs/groups.toml</span>; availability <b>future</b> = known for the target day before the gate, <b>past</b> = only known up to the forecast origin. Categories come from keyword rules on the English name, so a few rows may sit in the wrong bucket.</p></div></div>
    <div class="kpis">${kpi("Series", cat.length.toLocaleString("en-GB"))}${kpi("Used", String(cat.filter(r => r.used).length))}${kpi("Feature groups", String(groups.size))}</div>
    <div class="card"><div class="card-head"><div><h2>Catalog</h2><p class="sub" id="in-n"></p></div>
      <div class="controls"><select id="in-c" aria-label="Category"><option value="">All categories</option>${cats.map(c => `<option value="${esc(c.c)}">${esc(c.c)} (${c.used}/${c.n})</option>`).join("")}</select>
      <label class="tog"><input type="checkbox" id="in-u"> used only</label>
      <input type="search" id="in-q" placeholder="Search: mFRR, wind, 600…" aria-label="Search name, description or id"></div></div>
      <div class="tablewrap" style="max-height:640px" id="in-t"></div>
      <button class="chip" id="in-more" hidden>Show all</button></div>`;
  let limit = 200;
  const draw = () => {
    const c = root.querySelector("#in-c").value, u = root.querySelector("#in-u").checked, q = root.querySelector("#in-q").value.trim().toLowerCase();
    const rows = cat.filter(r => (!c || r.category === c) && (!u || r.used) && (!q || [r.name, r.name_es, r.description].some(s => (s ?? "").toLowerCase().includes(q)) || String(r.id) === q));
    root.querySelector("#in-n").textContent = `${rows.length.toLocaleString("en-GB")} of ${cat.length.toLocaleString("en-GB")} series`;
    root.querySelector("#in-more").hidden = rows.length <= limit;
    root.querySelector("#in-t").innerHTML = `<table><thead><tr><th>Used</th><th>Category</th><th>Source</th><th class="n">ID</th><th>Name</th><th>Group</th><th>Availability</th><th>Columns</th><th>Description</th></tr></thead><tbody>${rows.slice(0, limit).map(r =>
      `<tr><td>${r.used ? '<span class="pill">used</span>' : ""}</td><td class="muted">${esc(r.category)}${r.subcategory ? `<br><small>${esc(r.subcategory)}</small>` : ""}</td><td class="mono">${esc(r.source)}</td><td class="n">${esc(r.id)}</td><td>${esc(r.name)}${r.area ? `<br><small class="muted">${esc(r.area)}</small>` : ""}</td><td class="mono">${esc(r.group)}</td><td>${esc(r.availability)}</td><td class="mono">${esc(r.column)}</td><td class="desc">${esc(r.description)}</td></tr>`).join("")}</tbody></table>`;
  };
  ["in-c", "in-u"].forEach(id => root.querySelector(`#${id}`).addEventListener("change", draw));
  root.querySelector("#in-q").addEventListener("input", draw);
  root.querySelector("#in-more").addEventListener("click", () => { limit = Infinity; draw(); });
  draw();
}

// Outside Spain: the market's feature columns from configs/groups.toml (build.py writes them).
async function renderFeatures(root) {
  const m = mk(), rows = (await json("features.json"))[m.id];
  root.innerHTML = `
    <div class="head"><div><h1>Input features</h1><p class="lede">The columns the ${esc(m.name)} models see, one line each in <span class="mono">configs/groups.toml</span>. Availability <b>future</b> = known for the target day before the gate, <b>past</b> = only known up to the forecast origin.</p></div></div>
    <div class="card"><div class="card-head"><div><h2>Feature columns</h2><p class="sub">${rows.length} columns</p></div></div>
      <div class="tablewrap"><table><thead><tr><th>Group</th><th>Column</th><th>Source</th><th>Availability</th><th>Description</th></tr></thead><tbody>${rows.map(r =>
        `<tr><td class="mono">${esc(r.group)}</td><td class="mono">${esc(r.column)}</td><td class="mono">${esc(r.source)}</td><td>${esc(r.availability)}</td><td class="desc">${esc(r.description)}</td></tr>`).join("")}</tbody></table></div></div>`;
}

// --- Targets (Predico resources) --------------------------------------------------------
// live: wired (data, backtest, live run). city/gateLocal/source/missingTruth/about: tab texts.
const de50Hertz = what => (from, to) => `The truth series from <span class="mono">v2/target_50hertz.parquet</span>: ENTSO-E actual generation, ${what}, 50Hertz control area, MW (mean per quarter-hour), 15-min, ${from} – ${to}. As Predico does, points the curve omits repeat the previous value (curve type A03), each series before summing.`;
const DE = { program: "50Hertz", gate: "15:00 CET, D-1", live: true, city: "Berlin", gateLocal: "15:00 Berlin on D-1", source: "ENTSO-E", missingTruth: "" };
const MARKETS = [
  { id: "es-mfrr", name: "Spain mFRR", note: "energy & price", program: "EDP", gate: "17:00 CET, D-1", resources: ["spain_mfrr_energy", "spain_mfrr_price"], truth: "ESIOS (REE)", live: true, catalog: true,
    city: "Madrid", gateLocal: "17:00 Madrid on D-1", source: "ESIOS", missingTruth: "Quarter-hours without truth (no mFRR activation, so no price) are left out.",
    about: (from, to) => `The truth series from <span class="mono">v1/target.parquet</span>: mFRR activated <b>energy</b> (net, MWh per quarter-hour, upward positive) and mFRR <b>price</b> (EUR/MWh), Spain, 15-min, ${from} – ${to}. Missing energy is filled with 0 (no activation); missing price stays empty. No data during the Iberian blackout (28 Apr – 6 May 2025). Prices rise from Jul 2026 (monthly mean ~100 → ~157 EUR/MWh).` },
  { id: "be-solar", name: "Belgium solar", program: "Elia", gate: "10:00 CET, D-1 (plus hourly intraday)", resources: ["belgium_solar_opendata_elia"], truth: "Elia Open Data (ods032)" },
  { id: "be-wind", name: "Belgium offshore wind", program: "Elia", gate: "10:00 CET, D-1 (paused by Elia)", resources: ["offshore_wind_opendata_elia"], truth: "Elia Open Data" },
  { id: "de-solar", name: "Germany solar", note: "50Hertz", resources: ["germany_solar_50hertz_ca"], truth: "ENTSO-E A75 B16, 50Hertz control area", about: de50Hertz("solar (A75 B16)"), ...DE },
  { id: "de-wind", name: "Germany wind", note: "50Hertz", resources: ["germany_wind_50hertz_ca"], truth: "ENTSO-E A75 B18 + B19, 50Hertz control area", about: de50Hertz("wind offshore + onshore (A75 B18 + B19)"), ...DE },
];
let market = (() => { try { return MARKETS.find(m => m.id === localStorage.getItem("market"))?.id ?? "es-mfrr"; } catch { return "es-mfrr"; } })();
const TAB_NAMES = { live: "Live forecasts", backtest: "Backtest", target: "Target data", inputs: "Input features" };
function comingSoon(root, tab) {
  const m = MARKETS.find(x => x.id === market);
  root.innerHTML = `<div class="head"><div><h1>${esc(TAB_NAMES[tab])}: ${esc(m.name)}</h1></div></div>
    <div class="card soon"><span class="pill muted">coming soon</span>
      <p class="lede">${esc(m.name)} is a Predico target we plan to forecast next. Its data, backtest and live submissions are not wired up yet.</p>
      <dl><dt>Predico resource</dt><dd>${m.resources.map(esc).join(", ")}</dd><dt>Program</dt><dd>${esc(m.program)}</dd><dt>Gate</dt><dd>${esc(m.gate)}</dd><dt>Truth</dt><dd>${esc(m.truth)}</dd></dl>
      <p class="sub">Want to help? See <a href="https://github.com/arnaud-vs/open-predico/blob/main/PLAN.md#adding-a-target" target="_blank" rel="noopener">Adding a target</a> in the plan.</p></div>`;
}
function drawMarkets() {
  document.getElementById("markets").innerHTML = MARKETS.map(m => `<button class="chip" data-v="${m.id}" aria-pressed="${m.id === market}">${esc(m.name)}${m.live ? "" : " <small>soon</small>"}</button>`).join("");
}
document.getElementById("markets").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || b.dataset.v === market) return;
  market = b.dataset.v;
  try { localStorage.setItem("market", market); } catch {}
  drawMarkets(); show();
});

// --- Routing -----------------------------------------------------------------------------
const TABS = { live: renderLive, backtest: renderBacktest, target: renderTarget, inputs: root => (mk().catalog ? renderInputs : renderFeatures)(root) };
function show() {
  const tab = TABS[location.hash.slice(1)] ? location.hash.slice(1) : "live";
  document.querySelectorAll("section[data-tab]").forEach(s => (s.hidden = s.dataset.tab !== tab));
  document.querySelectorAll("nav a").forEach(a => (a.getAttribute("href") === `#${tab}` ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  const root = document.getElementById(`tab-${tab}`), key = `${market}:${tab}`;
  if (root.dataset.key === key) return;
  root.dataset.key = key;
  if (!mk().live) return comingSoon(root, tab);
  root.innerHTML = `<p class="empty">Loading…</p>`;
  TABS[tab](root).catch(e => { root.innerHTML = `<div class="card"><p class="empty">Could not load this tab: ${esc(e.message)}</p></div>`; console.error(e); });
}
drawMarkets();
addEventListener("hashchange", show);
(document.fonts?.ready ?? Promise.resolve()).then(show);
