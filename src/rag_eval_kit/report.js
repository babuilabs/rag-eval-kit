(() => {
const $ = (s, el = document) => el.querySelector(s);
const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const D = JSON.parse($('#explorer-data').textContent);
const gates = JSON.parse($('#gate-data').textContent);
const ORDER = ['fact', 'paraphrase', 'numeric', 'multi_hop', 'unanswerable'];
const Qs = [...D.questions].sort((a, b) => ORDER.indexOf(a.type) - ORDER.indexOf(b.type) || a.id.localeCompare(b.id));
const byId = Object.fromEntries(D.questions.map(q => [q.id, q]));

// How one version handled one question.
function state(q, a) {
  if (q.type === 'unanswerable') return a.x ? 'right' : 'wrong';
  if (a.x) return 'idk';
  return a.s >= 1 ? 'right' : a.s > 0 ? 'partial' : 'wrong';
}
// Cost of showing an answer: same definition as the gate (partial counts half; refusals are never shown).
const loss = (q, a) => a.x ? 0 : q.type === 'unanswerable' ? 1 : a.s >= 1 ? 0 : a.s > 0 ? 0.5 : 1;

function makeDots(host) {
  return Qs.map(q => {
    const d = el('button', 'dot'); d.type = 'button'; d.title = q.id + ' · ' + q.q;
    d.setAttribute('aria-label', q.id); d.onclick = () => openQuestion(q.id); host.append(d); return d;
  });
}
function paint(dots, classes) {
  dots.forEach((d, i) => {
    const c = 'dot ' + classes[i];
    if (d.className.replace(' flip', '') === c) return;
    const go = () => { d.className = c; if (!calm) { d.classList.add('flip'); setTimeout(() => d.classList.remove('flip'), 520); } };
    calm ? go() : setTimeout(go, i * 6);
  });
}
function count(node, to) {
  const from = +node.textContent || 0;
  node.dataset.to = to;
  if (calm || from === to || document.hidden) { node.textContent = to; return; }
  const t0 = performance.now();
  const tick = t => {
    if (+node.dataset.to !== to) return;  // a newer count took over
    const k = Math.min((t - t0) / 600, 1); node.textContent = Math.round(from + (to - from) * k);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  // Animation frames pause in background tabs; make sure the final number always lands.
  setTimeout(() => { if (+node.dataset.to === to) node.textContent = to; }, 700);
}

// ---- Hero: step through the versions
const heroDots = makeDots($('#dots')), steps = $('#steps');
let cur = 0, timer = null;
D.versions.forEach((v, i) => {
  const b = el('button', 'step', v.id.split('_')[0]); b.type = 'button'; b.title = v.label;
  b.onclick = () => { stop(); show(i); }; steps.append(b);
});
const play = el('button', 'play', '▶ Replay'); play.type = 'button'; play.onclick = () => autoplay(); steps.append(play);
function show(i) {
  cur = i; const v = D.versions[i], ans = D.answers[v.id];
  const cls = Qs.map(q => state(q, ans[q.id]));
  paint(heroDots, cls);
  $('#ver').textContent = v.label; $('#change').textContent = D.steps[v.id] || '';
  count($('#t-right'), cls.filter(c => c === 'right').length);
  count($('#t-wrong'), cls.filter(c => c === 'wrong').length);
  count($('#t-idk'), cls.filter(c => c === 'idk').length);
  steps.querySelectorAll('.step').forEach((b, j) => b.setAttribute('aria-pressed', j === i));
}
function stop() { clearInterval(timer); timer = null; }
function autoplay() {
  stop(); show(0); let i = 0;
  timer = setInterval(() => { i += 1; if (i >= D.versions.length) return stop(); show(i); }, 1900);
}
if (calm) show(D.versions.length - 1); else { show(0); setTimeout(autoplay, 700); }

// ---- Gate: same dots, some held back for a person
const gDots = makeDots($('#gdots')), pills = $('#gate-src'), slider = $('#alpha');
let src = gates[0].source;
gates.forEach(g => { const b = el('button', 'pill', g.label); b.type = 'button'; b.dataset.src = g.source;
  b.onclick = () => { src = g.source; drawGate(); }; pills.append(b); });
function drawGate() {
  const g = gates.find(x => x.source === src), key = String(Math.round(+slider.value * 1000) / 1000);
  const r = g.grid[key], ans = D.answers[src];
  pills.querySelectorAll('.pill').forEach(b => b.setAttribute('aria-pressed', b.dataset.src === src));
  $('#alpha-out').textContent = pct(+slider.value, 1);
  // Show as many held-back answers as the average held-out split, picking the ones held back most often.
  const held = Math.max(g.no_gate.answered - r.answered, 0);
  const freq = Object.fromEntries(g.ids.map((id, i) => [id, r.held_freq[i]]));
  const shownIds = Qs.filter(q => !ans[q.id].x).map(q => q.id).sort((a, b) => freq[b] - freq[a]);
  const heldIds = new Set(shownIds.slice(0, Math.round(held * Qs.length)));
  paint(gDots, Qs.map(q => {
    const a = ans[q.id];
    if (a.x) return 'idk';
    if (heldIds.has(q.id)) return 'held';
    return loss(q, a) === 0 ? 'right' : 'wrong';
  }));
  $('#g-wrong').textContent = pct(r.wrong, 1); $('#g-answered').textContent = pct(r.answered); $('#g-held').textContent = pct(held);
  const strict = r.answered < 0.005 ? ' This target is stricter than 52 calibration questions can certify, so everything is held back.' : '';
  $('#g-base').textContent = 'Without the checker: ' + pct(g.no_gate.wrong, 1) + ' wrong, ' + pct(g.no_gate.answered) +
    ' answered. Figures are averages over 1000 held-out splits.' + strict;
}
slider.oninput = drawGate; drawGate();

// ---- Explorer (inside the "Go deeper" layer)
const L = $('#left'), R = $('#right'), T = $('#type'), F = $('#show'), S = $('#search'), list = $('#qlist');
D.versions.forEach(v => { L.append(new Option(v.label, v.id)); R.append(new Option(v.label, v.id)); });
L.value = D.versions[0].id; R.value = D.versions[D.versions.length - 1].id;
ORDER.forEach(t => T.append(new Option(t.replace('_', '-'), t)));
function verdict(q, a) {
  if (q.type === 'unanswerable') return a.x ? ['good', 'said “don’t know”'] : ['bad', 'made up an answer'];
  if (a.x) return ['bad', 'refused'];
  return a.s >= 1 ? ['good', 'correct'] : a.s > 0 ? ['mid', 'partial'] : ['bad', 'wrong'];
}
const clean = t => (t || '').replace(/\*\*/g, '');
let limit = 25;
function filtered() {
  const l = D.answers[L.value], r = D.answers[R.value], term = S.value.trim().toLowerCase();
  return Qs.filter(q => {
    if (T.value && q.type !== T.value) return false;
    if (term && !(q.q + ' ' + q.ref + ' ' + q.id).toLowerCase().includes(term)) return false;
    const a = l[q.id].s, b = r[q.id].s;
    switch (F.value) {
      case 'fixed': return a < 1 && b >= 1;
      case 'broke': return a >= 1 && b < 1;
      case 'wrong': return b < 1;
      case 'flag': return q.weak;
    }
    return true;
  });
}
function side(label, q, a) {
  const d = el('div'), v = verdict(q, a);
  const h = el('h4', null, label + ' '); h.append(el('span', 'b ' + v[0], v[1])); d.append(h, el('p', null, clean(a.a)));
  const rec = a.rec == null ? '' : ' · evidence retrieved: ' + (a.rec >= 1 ? 'all' : a.rec > 0 ? 'some' : 'none');
  d.append(el('div', 'why', 'Grader: ' + clean(a.why) + rec)); return d;
}
function item(q, l, r) {
  const det = el('details', 'qitem'), sum = el('summary'), left = el('div', 'qtext');
  det.dataset.id = q.id;
  left.append(el('span', 'qid', q.id), el('span', 'tag', q.type.replace('_', '-')), document.createTextNode(q.q));
  if (q.weak) left.append(el('span', 'flag', '⚑ flagged in audit'));
  const badges = el('div', 'badges');
  [l[q.id], r[q.id]].forEach(a => { const v = verdict(q, a); badges.append(el('span', 'b ' + v[0], v[1])); });
  sum.append(left, badges); det.append(sum);
  det.addEventListener('toggle', () => {
    if (!det.open || det.dataset.done) return; det.dataset.done = 1;
    const body = el('div', 'qbody'), pair = el('div', 'pair');
    pair.append(side(L.selectedOptions[0].text, q, l[q.id]), side(R.selectedOptions[0].text, q, r[q.id]));
    body.append(pair, el('p', 'ref', 'Reference: ' + (q.ref || 'Not in the handbook. The right reply is “I don’t know”.')));
    det.append(body);
  });
  return det;
}
function render() {
  const qs = filtered(), l = D.answers[L.value], r = D.answers[R.value];
  $('#count').textContent = 'Showing ' + Math.min(limit, qs.length) + ' of ' + qs.length + ' matching questions.';
  list.replaceChildren(...qs.slice(0, limit).map(q => item(q, l, r)));
  $('#more').hidden = qs.length <= limit;
}
[L, R, T, F].forEach(x => x.onchange = () => { limit = 25; render(); });
S.oninput = () => { limit = 25; render(); };
$('#more').onclick = () => { limit += 25; render(); };
render();

// Clicking a dot opens that question in the explorer, comparing the baseline with the version on screen.
function openQuestion(id) {
  stop();
  const ex = $('#explorer'); ex.open = true;
  L.value = D.versions[0].id; R.value = D.versions[cur === 0 ? D.versions.length - 1 : cur].id;
  T.value = ''; F.value = ''; S.value = id; limit = 25; render();
  const det = list.querySelector('[data-id="' + id + '"]');
  if (det) { det.open = true; det.classList.add('hit'); det.scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'center' }); }
}
})();
