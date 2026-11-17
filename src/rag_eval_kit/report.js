(() => {
const $ = (s, el = document) => el.querySelector(s);
const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
const D = JSON.parse($('#explorer-data').textContent);
const gate = JSON.parse($('#gate-data').textContent);
const ORDER = ['fact', 'paraphrase', 'numeric', 'multi_hop', 'unanswerable'];
const Qs = [...D.questions].sort((a, b) => ORDER.indexOf(a.type) - ORDER.indexOf(b.type) || a.id.localeCompare(b.id));
const FIRST = D.versions[0].id, LAST = D.versions[D.versions.length - 1].id;

// How one version handled one question.
function state(q, a) {
  if (q.type === 'unanswerable') return a.x ? 'right' : 'wrong';
  if (a.x) return 'idk';
  return a.s >= 1 ? 'right' : a.s > 0 ? 'partial' : 'wrong';
}
// Run once when an element scrolls into view (immediately if observers are unavailable).
function onView(node, fn, margin = '0px 0px -15% 0px') {
  if (!('IntersectionObserver' in window)) return fn();
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: margin });
  io.observe(node);
}
document.querySelectorAll('.reveal').forEach(n => calm ? n.classList.add('in') : onView(n, () => n.classList.add('in')));
onView($('#journey'), () => $('#journey').classList.remove('pre'));
onView($('.phone'), () => $('.hero').classList.add('play'), '0px 0px -25% 0px');

function count(node, to) {
  const from = +node.textContent || 0;
  node.dataset.to = to;
  if (calm || from === to || document.hidden) { node.textContent = to; return; }
  const t0 = performance.now();
  const tick = t => {
    if (+node.dataset.to !== to) return;  // a newer count took over
    const k = Math.min((t - t0) / 700, 1); node.textContent = Math.round(from + (to - from) * k);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  // Animation frames pause in background tabs; make sure the final number always lands.
  setTimeout(() => { if (+node.dataset.to === to) node.textContent = to; }, 800);
}

// ---- Before / after dots
const dots = Qs.map(q => {
  const d = el('button', 'dot'); d.type = 'button'; d.title = q.q;
  d.setAttribute('aria-label', q.id); d.onclick = () => openQuestion(q.id); $('#dots').append(d); return d;
});
let cur = FIRST, timer = null;
function show(v) {
  cur = v; const ans = D.answers[v], cls = Qs.map(q => state(q, ans[q.id]));
  dots.forEach((d, i) => {
    const c = 'dot ' + cls[i];
    if (d.className.replace(' flip', '') === c) return;
    const go = () => { d.className = c; if (!calm) { d.classList.add('flip'); setTimeout(() => d.classList.remove('flip'), 520); } };
    calm ? go() : setTimeout(go, i * 7);
  });
  count($('#t-right'), cls.filter(c => c === 'right').length);
  count($('#t-wrong'), cls.filter(c => c === 'wrong').length);
  count($('#t-idk'), cls.filter(c => c === 'idk').length);
  $('#show-before').setAttribute('aria-pressed', v === FIRST);
  $('#show-after').setAttribute('aria-pressed', v === LAST);
}
$('#show-before').onclick = () => { clearTimeout(timer); show(FIRST); };
$('#show-after').onclick = () => { clearTimeout(timer); show(LAST); };
show(FIRST);
// The first time the grid is on screen, play the change once.
onView($('#dots'), () => { timer = setTimeout(() => show(LAST), calm ? 0 : 1200); }, '0px 0px -20% 0px');

// ---- Safety net on the final version
const slider = $('#alpha'), bar = $('#bar100').children;
function drawGate() {
  const r = gate.grid[String(Math.round(+slider.value * 1000) / 1000)], base = gate.no_gate;
  const parts = { right: r.right, wrong: Math.max(r.answered - r.right, 0), held: Math.max(base.answered - r.answered, 0), idk: 1 - base.answered };
  $('#alpha-out').textContent = pct(+slider.value, 1);
  ['right', 'wrong', 'held', 'idk'].forEach((k, i) => { bar[i].style.setProperty('--w', pct(parts[k], 2)); $('#o-' + k).textContent = Math.round(parts[k] * 100); });
  $('#g-base').textContent = 'Without the checker, ' + Math.round(base.wrong * 1000) / 10 + ' in 100 answers shown are wrong. ' +
    'Figures are averages over ' + gate.splits + ' tests, each on questions the checker was not tuned on; a partly right answer counts as half an error.';
}
slider.oninput = drawGate; drawGate();

// ---- Explorer (inside the "Go deeper" layer)
const L = $('#left'), R = $('#right'), T = $('#type'), F = $('#show'), S = $('#search'), list = $('#qlist');
D.versions.forEach(v => { L.append(new Option(v.label, v.id)); R.append(new Option(v.label, v.id)); });
L.value = FIRST; R.value = LAST;
ORDER.forEach(t => T.append(new Option(D.kinds[t] || t, t)));
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
  left.append(el('span', 'qid', q.id), el('span', 'tag', D.kinds[q.type] || q.type), document.createTextNode(q.q));
  if (q.weak) left.append(el('span', 'flag', '⚑ flagged in audit'));
  const badges = el('div', 'badges');
  [l[q.id], r[q.id]].forEach(a => { const v = verdict(q, a); badges.append(el('span', 'b ' + v[0], v[1])); });
  sum.append(left, badges); det.append(sum);
  det.addEventListener('toggle', () => {
    if (!det.open || det.dataset.done) return; det.dataset.done = 1;
    const body = el('div', 'qbody'), pair = el('div', 'pair');
    pair.append(side(L.selectedOptions[0].text, q, l[q.id]), side(R.selectedOptions[0].text, q, r[q.id]));
    body.append(pair, el('p', 'ref', 'Handbook: ' + (q.ref || 'Not in the handbook. The right reply is “I don’t know”.')));
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

// Clicking a dot opens that question in the explorer, before vs after.
function openQuestion(id) {
  clearTimeout(timer);
  const ex = $('#explorer'); ex.open = true;
  L.value = FIRST; R.value = LAST; T.value = ''; F.value = ''; S.value = id; limit = 25; render();
  const det = list.querySelector('[data-id="' + id + '"]');
  if (det) { det.open = true; det.classList.add('hit'); det.scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'center' }); }
}
})();
