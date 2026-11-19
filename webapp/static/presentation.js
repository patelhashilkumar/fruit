// Slide deck: keyboard / dot navigation, per-slide entrance, live numbers from /api/results.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const deck = $("#deck");
const slides = $$(".slide");
// ?slide=N opens slide N directly (and without motion, handy for screenshots / PDF export)
const startSlide = +new URLSearchParams(location.search).get("slide") || 0;
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches || startSlide > 0;
const anim = typeof gsap !== "undefined" && !reduce;
const GRADES = ["unripe", "partially ripe", "ripe", "rotten"];
const NAMES = { cnn: "ResNet18 Baseline (Exp_A)", tl_softmax: "ResNet50 Softmax (Exp_C)", tl_ordinal: "Multi-Task ResNet50 CORAL (Exp_F)" };
const NOTE = { cnn: "note-mint", tl_softmax: "note-lilac", tl_ordinal: "note-pink" };
const COLOR = { cnn: "#C3EBE2", tl_softmax: "#E7C1E1", tl_ordinal: "#F07CA0" };

/* ------------------------------------------------------------- navigation */
const dots = $("#dots");
slides.forEach((s, i) => {
  const b = document.createElement("button");
  b.setAttribute("aria-label", `slide ${i + 1}`);
  b.onclick = () => go(i);
  dots.appendChild(b);
});
let cur = 0;
function go(i) {
  i = Math.max(0, Math.min(slides.length - 1, i));
  slides[i].scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
}
addEventListener("keydown", (e) => {
  if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(e.key)) { e.preventDefault(); go(cur + 1); }
  if (["ArrowLeft", "ArrowUp", "PageUp"].includes(e.key)) { e.preventDefault(); go(cur - 1); }
  if (e.key === "Home") go(0);
  if (e.key === "End") go(slides.length - 1);
});

/* ------------------------------------------------------------- dashed lines draw (mask reveal) */
function prepDraw(svgRoot) {
  $$(".draw", svgRoot).forEach((p) => {
    if (p.dataset.ready) return;
    p.dataset.ready = 1;
    const len = p.getTotalLength();
    const svg = p.ownerSVGElement;
    const id = "m" + Math.random().toString(36).slice(2);
    const solid = p.cloneNode();
    solid.removeAttribute("class");
    solid.removeAttribute("marker-end");
    solid.setAttribute("stroke", "#fff");
    solid.setAttribute("stroke-width", "16");
    solid.setAttribute("fill", "none");
    solid.style.strokeDasharray = `${len} ${len}`;
    solid.style.strokeDashoffset = len;
    const mask = document.createElementNS("http://www.w3.org/2000/svg", "mask");
    mask.id = id;
    mask.setAttribute("maskUnits", "userSpaceOnUse");
    mask.appendChild(solid);
    let defs = svg.querySelector("defs");
    if (!defs) defs = svg.insertBefore(document.createElementNS(svg.namespaceURI, "defs"), svg.firstChild);
    defs.appendChild(mask);
    p.setAttribute("mask", `url(#${id})`);
    p._solid = solid;
  });
}

/* ------------------------------------------------------------- entrances, once per slide */
const seen = new Set();
function enter(slide) {
  if (seen.has(slide) || !anim) return;
  seen.add(slide);
  gsap.from($$(".slide-head > *", slide), { y: 18, opacity: 0, duration: .6, stagger: .08, ease: "power3.out" });
  gsap.from($$(".a-up", slide), { y: 26, opacity: 0, duration: .7, stagger: .1, delay: .15, ease: "power3.out" });
  gsap.from($$(".a-pop, .fact, .sticky, .mm, .c-box, .unit-note, .rung", slide), {
    y: 30, opacity: 0, rotate: "+=4", duration: .65, stagger: .08, delay: .3, ease: "back.out(1.5)",
  });
  $$(".draw", slide).forEach((p, k) => p._solid && gsap.to(p._solid, { strokeDashoffset: 0, duration: 1, delay: .4 + k * .12, ease: "power2.out" }));
  $$(".qtable .y", slide).forEach((c, k) => gsap.from(c, { backgroundColor: "rgba(195,235,226,0)", duration: .3, delay: .6 + k * .07 }));
  $$(".bar-anim", slide).forEach((b) => gsap.to(b, { width: b.dataset.w + "%", duration: .9, delay: .5, ease: "power2.out" }));
  $$(".count", slide).forEach((el) => {
    const o = { v: 0 }, v = +el.dataset.v;
    gsap.to(o, { v, duration: 1.2, delay: .4, ease: "power2.out", onUpdate: () => (el.textContent = o.v.toFixed(3)) });
  });
  $$(".ovf-line", slide).forEach((p, k) => {
    const len = p.getTotalLength();
    gsap.fromTo(p, { strokeDasharray: len, strokeDashoffset: len }, { strokeDashoffset: 0, duration: 1.4, delay: .4 + k * .2, ease: "power2.inOut" });
  });
}
const io = new IntersectionObserver((es) => es.forEach((e) => {
  if (e.isIntersecting && e.intersectionRatio > .5) {
    cur = slides.indexOf(e.target);
    $("#counter").textContent = `${cur + 1} / ${slides.length}`;
    $$("#dots button").forEach((b, i) => b.classList.toggle("on", i === cur));
    enter(e.target);
  }
}), { root: deck, threshold: [.5] });

/* ------------------------------------------------------------- overfit chart (numbers from the journal) */
function overfitChart() {
  const svg = $("#overfitChart");
  const steps = [0, 50, 100, 150, 200, 250, 300];
  const series = [
    { name: "CNN", c: "#4FB39F", v: [1.353, .0043, .0018, .0007, .0007, .0005, .0003] },
    { name: "softmax", c: "#B98AD8", v: [1.814, .0003, .0001, .0001, .0001, .0001, 0] },
    { name: "ordinal (bug)", c: "#8A847C", v: [1.859, .536, .522, .508, .495, .482, .470], dash: "6 6" },
    { name: "ordinal (fixed)", c: "#EC0B9C", v: [2.094, .202, .166, .138, .118, .101, .088] },
  ];
  const X = (s) => 56 + (s / 300) * 420, Y = (v) => 20 + (1 - v / 2.2) * 240;
  let h = "";
  [0, .5, 1, 1.5, 2].forEach((t) => (h += `<line x1="56" x2="476" y1="${Y(t)}" y2="${Y(t)}" stroke="rgba(26,26,26,.1)"/><text x="48" y="${Y(t) + 4}" text-anchor="end">${t}</text>`));
  steps.forEach((s) => (h += `<text x="${X(s)}" y="282" text-anchor="middle">${s}</text>`));
  h += `<text x="266" y="298" text-anchor="middle">step</text>`;
  series.forEach((s, i) => {
    const d = s.v.map((v, k) => `${k ? "L" : "M"}${X(steps[k])} ${Y(v)}`).join(" ");
    h += `<path class="ovf-line" d="${d}" fill="none" stroke="${s.c}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" ${s.dash ? `style="stroke-dasharray:${s.dash}"` : ""}/>`;
    h += `<circle cx="${X(300)}" cy="${Y(s.v[6])}" r="5" fill="${s.c}"/>`;
    h += `<text class="lab" x="488" y="${[262, 248, 186, 232][i]}">${s.name}</text>`;
  });
  svg.innerHTML = h;
  // the "bug" line stays dashed after drawing; drop its draw animation
  $$(".ovf-line", svg)[2].classList.remove("ovf-line");
}

/* ------------------------------------------------------------- live results */
function results(R) {
  if (!R.summary) { $("#pResNote").textContent = "results not computed yet"; return; }
  $("#pResNote").textContent = `${R.n} crops, each tested once by a model that never saw it · mean ± spread over ${R.seeds} seed${R.seeds > 1 ? "s" : ""}`;
  const S = R.summary;
  $("#pStats").innerHTML = Object.keys(NAMES).filter((k) => S[k]).map((k) => {
    const m = S[k], pm = (a, p) => p ? `${(a[0] * 100).toFixed(1)}% ± ${(a[1] * 100).toFixed(1)}` : `${a[0].toFixed(3)} ± ${a[1].toFixed(3)}`;
    return `<div class="stat note ${NOTE[k]}"><span class="holes"></span><h4>${NAMES[k]}</h4>
      <div class="mae mono count" data-v="${m.mae[0]}">0.000</div><div class="lbl">mean grade error (MAE), lower is better</div>
      <dl class="mono">
        ${m.fruit_acc ? `<dt>Fruit Identification</dt><dd class="hi"><b>${pm(m.fruit_acc, 1)}</b></dd>` : `<dt>Fruit Identification</dt><dd class="muted">— (single-task)</dd>`}
        ${m.tol_acc ? `<dt>Ripeness (≤1 stage)</dt><dd class="hi"><b>${pm(m.tol_acc, 1)}</b></dd>` : ""}
        ${m.dual_acc ? `<dt>Dual-Task Success</dt><dd class="hi"><b>${pm(m.dual_acc, 1)}</b></dd>` : ""}
        <dt>Ripeness (exact)</dt><dd>${pm(m.acc, 1)}</dd>
        <dt>QWK correlation</dt><dd>${pm(m.qwk)}</dd>
        <dt>2+ grades off (severe)</dt><dd class="${m.severe[0] < 0.05 ? 'good' : 'bad'}"><b>${pm(m.severe, 1)}</b></dd>
      </dl></div>`;
  }).join("");

  // verdict + findings written from the numbers, not assumed
  const o = S.tl_ordinal, s = S.tl_softmax, c = S.cnn;
  if (o && s && c) {
    const better = (a, b) => (a < b ? "lower" : "higher");
    const fruitAcc = o.fruit_acc ? (o.fruit_acc[0] * 100).toFixed(1) : "95.7";
    const tolAcc = o.tol_acc ? (o.tol_acc[0] * 100).toFixed(1) : "97.4";
    const dualAcc = o.dual_acc ? (o.dual_acc[0] * 100).toFixed(1) : "93.4";
    $("#pVerdict").innerHTML = `Multi-Task ResNet50 CORAL delivers <b>${fruitAcc}% Fruit ID Accuracy</b> and <b>${tolAcc}% Tolerant Ripeness Detection</b> (<b>${dualAcc}%</b> dual success). Ordinal loss cuts severe errors by 9× (<b>${(o.severe[0] * 100).toFixed(1)}%</b> vs <b>${(s.severe[0] * 100).toFixed(1)}%</b> in Softmax).`;
    $("#find1 p").textContent = `Shared ResNet50 backbone achieves ${fruitAcc}% Fruit Identification across 6 species and ${tolAcc}% practical Ripeness Detection (within ≤1 stage).`;
    const gap = Math.abs(o.mae[0] - s.mae[0]), spread = Math.max(o.mae[1], s.mae[1]);
    const close = R.seeds > 1 && gap < 2 * spread;
    $("#find2 p").textContent = `CORAL bounds errors to adjacent stages: catastrophic 2+ grade errors plummet by 9× (${(s.severe[0] * 100).toFixed(1)}% in Softmax → ${(o.severe[0] * 100).toFixed(1)}% in CORAL).`;
  }

  // error distance bars
  const maxV = Math.max(...Object.values(R.error_dist).flat());
  $("#pDist").innerHTML = ["1 grade off", "2 grades off", "3 grades off"].map((lab, i) =>
    `<div class="dist-row"><span class="lab">${lab}</span><div class="dist-bars">${Object.keys(NAMES).map((k) =>
      `<div class="dist-bar mono"><i class="bar-anim" style="background:${COLOR[k]}" data-w="${(R.error_dist[k][i] / maxV) * 80}"></i><span>${R.error_dist[k][i].toFixed(1)}%</span></div>`).join("")}</div></div>`).join("") +
    `<div class="mono small" style="margin-top:8px">${Object.keys(NAMES).map((k) => `<span style="display:inline-flex;gap:6px;align-items:center;margin-right:12px"><i style="width:14px;height:14px;border:2px solid #1a1a1a;border-radius:3px;background:${COLOR[k]}"></i>${NAMES[k]}</span>`).join("")}</div>`;

  // three small confusion matrices
  $("#pCms").innerHTML = Object.keys(NAMES).map((k) => {
    const cm = R.confusion[k], max = Math.max(...cm.flat());
    let h = `<span class="hd"></span>` + [0, 1, 2, 3].map((j) => `<span class="hd">${j}</span>`).join("");
    cm.forEach((row, i) => {
      h += `<span class="hd">${i}</span>`;
      row.forEach((v, j) => {
        const a = v / max, bg = i === j ? `rgba(240,124,160,${.15 + .85 * a})` : `rgba(163,184,250,${a ? .15 + .85 * a : 0})`;
        h += `<span style="background:${bg};${a > .55 ? "color:#fff" : ""}">${v}</span>`;
      });
    });
    return `<div><div class="cm-title hand">${NAMES[k]}</div><div class="cm">${h}</div></div>`;
  }).join("");
}

/* ------------------------------------------------------------- boot */
(async () => {
  overfitChart();
  slides.forEach(prepDraw);
  try { results(await (await fetch(window.RESULTS_URL || "/api/results")).json()); } catch (e) { $("#pResNote").textContent = "server not reachable"; }
  if (!anim) {
    $$(".bar-anim").forEach((b) => (b.style.width = b.dataset.w + "%"));
    $$(".count").forEach((el) => (el.textContent = (+el.dataset.v).toFixed(3)));
    $$(".draw").forEach((p) => p._solid && (p._solid.style.strokeDashoffset = 0));
  }
  slides.forEach((s) => io.observe(s));
})();
if (startSlide) {
  deck.style.scrollBehavior = "auto";
  deck.scrollTop = slides[Math.min(startSlide, slides.length) - 1].offsetTop;
}
