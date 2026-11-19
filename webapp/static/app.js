// Ripe or Not – Upgraded Interactive Multi-Task Diagnostics Studio.
// Hand-crafted aesthetic with full controls: multi-fruit filtering, dual-task Grad-CAM attention,
// real-time probability distributions, and smooth ordinal ruler animations.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const GRADES = ["unripe", "partially ripe", "ripe", "rotten"];
const EMOJIS = {
  banana: "🍌",
  mango: "🥭",
  papaya: "🍈",
  avocado: "🥑",
  strawberry: "🍓",
  apple: "🍎",
  unknown: "🏷️"
};
const NAMES = {
  cnn: "ResNet18 Baseline (Exp_A)",
  tl_softmax: "ResNet50 Softmax (Exp_C)",
  tl_ordinal: "Multi-Task ResNet50 CORAL (Exp_F)",
};
const NOTE = { cnn: "note-mint", tl_softmax: "note-lilac", tl_ordinal: "note-pink" };
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches || new URLSearchParams(location.search).has("sample");
const hasGsap = typeof gsap !== "undefined" && !reduce;
if (hasGsap) gsap.registerPlugin(ScrollTrigger);

let allSamples = [];
let activeFruitFilter = "all";
let activeStageFilter = "all";
let activeCamTask = "ripeness";
let currentBlob = null;
let currentSample = null;

/* ---------------------------------------------------------------- dashed lines draw in */
function prepDraw(path) {
  const len = path.getTotalLength();
  path.style.strokeDasharray = path.getAttribute("stroke-dasharray") || getComputedStyle(path).strokeDasharray;
  path.dataset.len = len;
}
function drawOnView() {
  $$(".draw").forEach((p) => {
    prepDraw(p);
    if (!hasGsap) return;
    const len = +p.dataset.len;
    const svg = p.ownerSVGElement;
    const id = "reveal" + Math.random().toString(36).slice(2);
    const solid = p.cloneNode();
    solid.removeAttribute("class");
    solid.removeAttribute("marker-end");
    solid.setAttribute("stroke-width", "14");
    solid.style.strokeDasharray = `${len} ${len}`;
    solid.style.strokeDashoffset = len;
    solid.setAttribute("fill", "none");
    const mask = document.createElementNS("http://www.w3.org/2000/svg", "mask");
    mask.id = id;
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    solid.setAttribute("stroke", "#fff");
    mask.appendChild(solid);
    (svg.querySelector("defs") || svg.insertBefore(document.createElementNS(svg.namespaceURI, "defs"), svg.firstChild)).appendChild(mask);
    p.setAttribute("mask", `url(#${id})`);
    gsap.to(solid, {
      strokeDashoffset: 0, duration: 1.1, ease: "power2.out",
      scrollTrigger: { trigger: svg, start: "top 85%", once: true },
    });
  });
}

/* ---------------------------------------------------------------- entrances */
function entrances() {
  if (!hasGsap) return;
  gsap.from(".hero-intro > *", { y: 16, opacity: 0, duration: .7, stagger: .08, ease: "power3.out" });
  gsap.from(".board", { y: 24, opacity: 0, duration: .8, delay: .2, ease: "power3.out" });
  $$(".sec .paper, .node, .unit-note, .formula").forEach((el) => {
    gsap.from(el, {
      y: 28, opacity: 0, rotate: "+=3", duration: .7, ease: "back.out(1.4)",
      scrollTrigger: { trigger: el, start: "top 88%", once: true },
    });
  });
  gsap.from(".boxes span", {
    y: 12, opacity: 0, stagger: .08, duration: .4,
    scrollTrigger: { trigger: ".boxes", start: "top 85%", once: true },
  });
  gsap.from(".qtable .y", {
    backgroundColor: "rgba(195,235,226,0)", stagger: .07, duration: .35,
    scrollTrigger: { trigger: ".qtable", start: "top 80%", once: true },
  });
}

/* ---------------------------------------------------------------- sample filtering & gallery */
function renderFilteredSamples() {
  const row = $("#samples");
  if (!row) return;
  row.innerHTML = "";

  const filtered = allSamples.filter((s) => {
    const fMatch = (activeFruitFilter === "all" || s.fruit.toLowerCase() === activeFruitFilter.toLowerCase());
    const sMatch = (activeStageFilter === "all" || s.grade.toString() === activeStageFilter.toString());
    return fMatch && sMatch;
  });

  if (filtered.length === 0) {
    row.innerHTML = `<span class="hand muted" style="grid-column:1/-1;padding:12px;text-align:center;">No matching samples for this filter</span>`;
    return;
  }

  filtered.forEach((s) => {
    const b = document.createElement("button");
    b.className = "sample";
    if (currentSample && currentSample.crop === s.crop) b.classList.add("selected");
    b.title = `${s.label || s.fruit} (True grade: ${GRADES[s.grade]})`;
    b.innerHTML = `<img src="/crops/${s.crop}" alt="${s.fruit}, ${GRADES[s.grade]}" loading="lazy">`;
    b.onclick = async () => {
      $$(".sample").forEach((el) => el.classList.remove("selected"));
      b.classList.add("selected");
      try {
        const blob = await (await fetch(`/crops/${s.crop}`)).blob();
        run(blob, s);
      } catch (e) {
        console.error("Failed to load sample blob:", e);
      }
    };
    row.appendChild(b);
  });
}

function loadSamples(samples) {
  allSamples = samples || [];
  renderFilteredSamples();
}

/* ---------------------------------------------------------------- controls & input setup */
function setupControls() {
  // 1. File Drop & Picker
  const drop = $("#drop");
  const file = $("#file");
  const btnBrowse = $("#btnBrowse");
  const btnReset = $("#btnReset");

  if (btnBrowse && file) {
    btnBrowse.onclick = (e) => {
      e.preventDefault();
      file.click();
    };
  }

  if (file) {
    file.onchange = () => {
      if (file.files[0]) {
        $$(".sample").forEach((el) => el.classList.remove("selected"));
        run(file.files[0], null);
      }
    };
  }

  if (drop) {
    ["dragenter", "dragover"].forEach((e) =>
      drop.addEventListener(e, (ev) => {
        ev.preventDefault();
        drop.classList.add("over");
      })
    );
    ["dragleave", "drop"].forEach((e) =>
      drop.addEventListener(e, (ev) => {
        ev.preventDefault();
        drop.classList.remove("over");
      })
    );
    drop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      if (ev.dataTransfer.files[0]) {
        $$(".sample").forEach((el) => el.classList.remove("selected"));
        run(ev.dataTransfer.files[0], null);
      }
    });
  }

  // 2. Clear / Reset Button
  if (btnReset) {
    btnReset.onclick = () => {
      if (file) file.value = "";
      const photo = $("#photo");
      if (photo) photo.src = "";
      if (drop) drop.classList.remove("has");
      currentBlob = null;
      currentSample = null;
      $$(".sample").forEach((el) => el.classList.remove("selected"));
      const res = $("#result");
      const ph = $("#placeholderStudio");
      if (res) res.hidden = true;
      if (ph) ph.style.display = "block";
      $("#status").textContent = "model ready • RTX 4050";
    };
  }

  // 3. Grad-CAM Task Toggle
  $$("#camTaskGroup .pill-btn").forEach((btn) => {
    btn.onclick = () => {
      $$("#camTaskGroup .pill-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeCamTask = btn.dataset.cam;
      if (currentBlob) {
        run(currentBlob, currentSample);
      }
    };
  });

  // 4. Fruit Filter Pills
  $$("#fruitFilters .filter-pill").forEach((pill) => {
    pill.onclick = () => {
      $$("#fruitFilters .filter-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      activeFruitFilter = pill.dataset.fruit;
      renderFilteredSamples();
    };
  });

  // 5. Stage Filter Pills
  $$("#stageFilters .filter-pill").forEach((pill) => {
    pill.onclick = () => {
      $$("#stageFilters .filter-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      activeStageFilter = pill.dataset.stage;
      renderFilteredSamples();
    };
  });
}

/* ---------------------------------------------------------------- run inference */
async function run(blob, sample = null) {
  currentBlob = blob;
  currentSample = sample;

  const photo = $("#photo");
  if (photo) photo.src = URL.createObjectURL(blob);
  $("#drop")?.classList.add("has");
  $("#status").textContent = "analyzing…";

  const fd = new FormData();
  fd.append("file", blob, "img.jpg");

  const cropParam = sample ? `crop=${encodeURIComponent(sample.crop)}` : "";
  const camParam = `cam_task=${encodeURIComponent(activeCamTask)}`;
  const qs = [cropParam, camParam].filter(Boolean).join("&");
  const url = "/api/predict" + (qs ? `?${qs}` : "");

  let r;
  try {
    r = await (await fetch(url, { method: "POST", body: fd })).json();
  } catch (e) {
    console.error(e);
    $("#status").textContent = "server not reachable";
    return;
  }

  show(r, sample);
}

/* ---------------------------------------------------------------- render diagnostic studio */
function show(r, sample) {
  const ph = $("#placeholderStudio");
  if (ph) ph.style.display = "none";

  const res = $("#result");
  const first = res.hidden;
  res.hidden = false;

  $("#status").textContent = r.used || "ResNet50 Multi-Task CORAL";

  // 1. Fruit Identification Display
  if (r.fruit_type) {
    const fn = $("#fruitName");
    const fc = $("#fruitConfidence");
    const fe = $("#fruitEmoji");
    const fKey = r.fruit_type.toLowerCase();

    if (fn) fn.textContent = r.fruit_type;
    if (fc) fc.textContent = `${(r.fruit_confidence * 100).toFixed(1)}% conf`;
    if (fe) fe.textContent = EMOJIS[fKey] || "🏷️";
  }

  // 2. Fruit Probabilities Distribution Breakdown
  const pGrid = $("#probGrid");
  if (pGrid && r.fruit_probabilities) {
    pGrid.innerHTML = "";
    Object.entries(r.fruit_probabilities).forEach(([fname, prob]) => {
      const pct = (prob * 100).toFixed(1);
      const emoji = EMOJIS[fname.toLowerCase()] || "";
      const item = document.createElement("div");
      item.className = "prob-item";
      item.innerHTML = `
        <div class="prob-label"><span>${emoji} ${fname}</span><span>${pct}%</span></div>
        <div class="prob-bar"><div class="prob-fill" style="width: ${pct}%"></div></div>
      `;
      pGrid.appendChild(item);
    });
  }

  // 3. Ripeness Ruler & Pointer Animation
  const o = r.tl_ordinal;
  const pos = Math.max(0, Math.min(100, (o.expected / 3) * 100));
  $("#score").textContent = o.expected.toFixed(2);

  if (hasGsap) {
    gsap.to("#pointer", { left: pos + "%", duration: first ? 1.2 : 0.7, ease: "power3.inOut" });
  } else {
    $("#pointer").style.left = pos + "%";
  }

  const truth = sample
    ? `<small>True: ${GRADES[sample.grade]} ${sample.grade === o.grade ? "— correct" : `— off by ${Math.abs(sample.grade - o.grade)}`}</small>`
    : `<small>Grade = ordinal gates triggered below (score ${o.expected.toFixed(2)} / 3.0)</small>`;

  $("#verdict").innerHTML = `${GRADES[o.grade]}${truth}`;

  // 4. The 3 Ordinal Gate Questions
  $$(".q").forEach((q, k) => {
    const p = o.p_gt[k];
    const isYes = p > 0.5;
    $(".pct", q).innerHTML = `<b>${(p * 100).toFixed(1)}%</b> <span style="color:${isYes ? 'var(--magenta)' : 'var(--ink-3)'};font-weight:bold">${isYes ? "✔ YES (triggered)" : "✘ NO (unreached)"}</span>`;
    const bar = $(".bar i", q);
    if (hasGsap) {
      gsap.to(bar, { width: p * 100 + "%", duration: 0.8, delay: 0.1 * k, ease: "power2.out" });
    } else {
      bar.style.width = p * 100 + "%";
    }
  });

  // 5. Grad-CAM & Softmax Compare Studio
  $("#camImg").src = r.gradcam;
  const caption = $("#camCaption");
  if (caption) {
    caption.textContent = activeCamTask === "fruit"
      ? "Species morphological focus (Grad-CAM)"
      : "Ripeness & decay focus (Grad-CAM)";
  }

  const s = r.tl_softmax;
  $("#softmaxSays").textContent = GRADES[s.grade];
  $("#softmaxProbs").textContent = s.probs.map((p, i) => `${GRADES[i]}: ${(p * 100).toFixed(0)}%`).join("  ");

  if (first && hasGsap) {
    gsap.from("#result > *", { y: 16, opacity: 0, stagger: 0.08, duration: 0.5, ease: "power3.out" });
  }
}

/* ---------------------------------------------------------------- results & evaluation stats */
function fmt(m, s, pct = false, d = 3) {
  return pct ? `${(m * 100).toFixed(1)}% ± ${(s * 100).toFixed(1)}` : `${m.toFixed(d)} ± ${s.toFixed(d)}`;
}

function renderResults(R) {
  if (!R.summary) {
    $("#resultsNote").textContent = "results not computed yet – run evaluation suite";
    return;
  }
  $("#resultsNote").textContent =
    `${R.n} fruit crops, each tested once by a model that never saw it · mean ± spread over ${R.seeds} random seeds`;
  const stats = $("#stats");
  stats.innerHTML = "";
  Object.keys(NAMES).forEach((k) => {
    const m = R.summary[k];
    if (!m) return;
    const d = document.createElement("div");
    d.className = `stat note ${NOTE[k]}`;
    d.innerHTML = `<span class="holes"></span><h4>${NAMES[k]}</h4>
      <div class="mae mono" data-v="${m.mae[0]}">0.000</div><div class="lbl">mean grade error (MAE), lower is better</div>
      <dl class="mono">
        ${m.fruit_acc ? `<dt>Fruit Identification</dt><dd class="hi"><b>${fmt(m.fruit_acc[0], m.fruit_acc[1], true)}</b></dd>` : `<dt>Fruit Identification</dt><dd class="muted">— (single-task)</dd>`}
        ${m.tol_acc ? `<dt>Ripeness (≤1 stage off)</dt><dd class="hi"><b>${fmt(m.tol_acc[0], m.tol_acc[1], true)}</b></dd>` : ""}
        ${m.dual_acc ? `<dt>Dual-Task Success</dt><dd class="hi"><b>${fmt(m.dual_acc[0], m.dual_acc[1], true)}</b></dd>` : ""}
        <dt>Ripeness (Exact 4-Stage)</dt><dd>${fmt(m.acc[0], m.acc[1], true)}</dd>
        <dt>QWK Metric</dt><dd>${fmt(m.qwk[0], m.qwk[1])}</dd>
        <dt>2+ Grades Off (Severe)</dt><dd class="${m.severe[0] < 0.05 ? 'good' : 'bad'}"><b>${fmt(m.severe[0], m.severe[1], true)}</b></dd>
      </dl>`;
    stats.appendChild(d);
  });

  $$(".mae").forEach((el) => {
    const v = +el.dataset.v;
    if (!hasGsap) { el.textContent = v.toFixed(3); return; }
    const o = { v: 0 };
    gsap.to(o, {
      v, duration: 1.2, ease: "power2.out", onUpdate: () => (el.textContent = o.v.toFixed(3)),
      scrollTrigger: { trigger: el, start: "top 90%", once: true },
    });
  });

  // error distance bars
  const dist = $("#dist");
  dist.innerHTML = "";
  const colors = { cnn: "var(--mint)", tl_softmax: "var(--lilac)", tl_ordinal: "var(--pink)" };
  const maxV = Math.max(...Object.values(R.error_dist).flat());
  ["1 grade off", "2 grades off", "3 grades off"].forEach((lab, i) => {
    const row = document.createElement("div");
    row.className = "dist-row";
    row.innerHTML = `<span class="lab">${lab}</span><div class="dist-bars"></div>`;
    Object.keys(NAMES).forEach((k) => {
      const v = R.error_dist[k][i];
      const b = document.createElement("div");
      b.className = "dist-bar mono";
      b.innerHTML = `<i style="background:${colors[k]}" data-w="${(v / maxV) * 82}"></i><span>${v.toFixed(1)}%</span>`;
      $(".dist-bars", row).appendChild(b);
    });
    dist.appendChild(row);
  });
  const legend = document.createElement("div");
  legend.className = "mono small";
  legend.innerHTML = Object.keys(NAMES).map((k) => `<span style="display:inline-flex;gap:6px;align-items:center;margin-right:14px"><i style="width:14px;height:14px;border:2px solid #1a1a1a;border-radius:3px;background:${colors[k]}"></i>${NAMES[k]}</span>`).join("");
  dist.appendChild(legend);
  $$(".dist-bar i").forEach((el) => {
    if (!hasGsap) { el.style.width = el.dataset.w + "%"; return; }
    gsap.to(el, { width: el.dataset.w + "%", duration: .9, ease: "power2.out", scrollTrigger: { trigger: dist, start: "top 85%", once: true } });
  });

  // confusion matrix with a picker
  const pick = $("#cmPick");
  pick.innerHTML = Object.keys(NAMES).map((k) => `<option value="${k}">${NAMES[k]}</option>`).join("");
  pick.value = "tl_ordinal";
  pick.onchange = () => drawCM(R.confusion[pick.value]);
  drawCM(R.confusion.tl_ordinal);

  // per-fruit table
  const rows = R.per_fruit.map((r) => `<tr>
    <td><b>${r.fruit}</b></td>
    <td><b style="color:var(--magenta)">${(r.fruit_acc * 100).toFixed(1)}%</b></td>
    <td><b style="color:var(--ink)">${(r.tol_acc * 100).toFixed(1)}%</b></td>
    <td>${(r.acc * 100).toFixed(1)}%</td>
    <td>${r.mae.toFixed(3)}</td>
    <td>${r.qwk.toFixed(3)}</td>
    <td class="${r.severe < 0.03 ? 'good' : ''}"><b>${(r.severe * 100).toFixed(1)}%</b></td>
  </tr>`).join("");
  $("#perFruit").innerHTML = `<h3>Per-Fruit Multi-Task Performance (Held-Out Test Set)</h3>
    <table class="mono">
      <thead>
        <tr>
          <th>Fruit Species</th>
          <th>Fruit ID Acc</th>
          <th>Ripeness (≤1 off)</th>
          <th>Exact Stage Match</th>
          <th>MAE</th>
          <th>QWK</th>
          <th>Severe (2+ off)</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="hand muted" style="margin-top:12px">All 6 fruit species exceed 94% Fruit ID Accuracy and 95% Tolerant Ripeness Detection.</p>`;
}

function drawCM(cm) {
  const el = $("#cm");
  const max = Math.max(...cm.flat());
  let h = `<span class="hd"></span>` + GRADES.map((_, j) => `<span class="hd">${j}</span>`).join("");
  cm.forEach((row, i) => {
    h += `<span class="hd">${i}</span>`;
    row.forEach((v, j) => {
      const a = v / max;
      const bg = i === j ? `rgba(240,124,160,${0.15 + 0.85 * a})` : `rgba(163,184,250,${a ? 0.15 + 0.85 * a : 0})`;
      h += `<span style="background:${bg};${a > .55 ? "color:#fff" : ""}">${v}</span>`;
    });
  });
  el.innerHTML = h;
}

/* ---------------------------------------------------------------- boot */
(async function boot() {
  setupControls();
  drawOnView();
  entrances();
  let R = {};
  try {
    R = await (await fetch(window.RESULTS_URL || "/api/results")).json();
  } catch (e) {
    /* offline fallback */
  }
  renderResults(R);
  if (R.samples) loadSamples(R.samples);
  if (hasGsap) ScrollTrigger.refresh();

  // URL auto-grading parameter (e.g. ?sample=0)
  const auto = new URLSearchParams(location.search).get("sample");
  if (auto !== null && R.samples) {
    setTimeout(() => $$(".sample")[+auto]?.click(), 300);
  }
})();
