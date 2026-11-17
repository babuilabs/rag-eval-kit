(() => {
const $ = (s, el = document) => el.querySelector(s);
const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// Safe gate slider
const gates = JSON.parse($('#gate-data').textContent);
let src = gates[0].source;
const pills = $('#gate-src'), slider = $('#alpha');
gates.forEach(g => { const b = el('button', 'pill', g.label); b.type = 'button'; b.dataset.src = g.source;
  b.onclick = () => { src = g.source; draw(); }; pills.append(b); });
function draw() {
  const g = gates.find(x => x.source === src);
  const r = g.grid[String(Math.round(+slider.value * 1000) / 1000)];
  pills.querySelectorAll('.pill').forEach(b => b.setAttribute('aria-pressed', b.dataset.src === src));
  $('#alpha-out').textContent = pct(+slider.value, 1);
  const idk = 1 - g.no_gate.answered, held = Math.max(g.no_gate.answered - r.answered, 0), notRight = r.answered - r.right;
  [['#s-right', r.right], ['#s-wrong', notRight], ['#s-held', held], ['#s-idk', idk]].forEach(([s, v]) => $(s).style.width = pct(Math.max(v, 0), 2));
  $('#g-wrong').textContent = pct(r.wrong, 1); $('#g-p90').textContent = pct(r.wrong_p90, 1);
  $('#g-answered').textContent = pct(r.answered); $('#g-held').textContent = pct(held);
  const strict = r.answered < 0.005 ? ' This target is stricter than 52 calibration questions can certify (about 1 in 53), so the gate holds back every answer.' : '';
  $('#g-base').textContent = 'Without the gate: ' + pct(g.no_gate.wrong, 1) + ' wrong shown, ' + pct(g.no_gate.answered) + ' answered.' + strict;
}
slider.oninput = draw; draw();

// Question explorer
const D = JSON.parse($('#explorer-data').textContent);
const L = $('#left'), R = $('#right'), T = $('#type'), F = $('#show'), Q = $('#search'), list = $('#qlist');
D.versions.forEach(v => { L.append(new Option(v.label, v.id)); R.append(new Option(v.label, v.id)); });
L.value = D.versions[0].id; R.value = D.versions[D.versions.length - 1].id;
[...new Set(D.questions.map(q => q.type))].forEach(t => T.append(new Option(t.replace('_', '-'), t)));
function verdict(q, a) {
  if (q.type === 'unanswerable') return a.x ? ['good', 'said “don’t know”'] : ['bad', 'made up an answer'];
  if (a.x) return ['bad', 'refused'];
  return a.s >= 1 ? ['good', 'correct'] : a.s > 0 ? ['mid', 'partial'] : ['bad', 'wrong'];
}
const clean = t => (t || '').replace(/\*\*/g, '');
let shown = 25;
function filtered() {
  const l = D.answers[L.value], r = D.answers[R.value], term = Q.value.trim().toLowerCase();
  return D.questions.filter(q => {
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
  d.append(el('div', 'why', 'Judge: ' + clean(a.why) + rec)); return d;
}
function render() {
  const qs = filtered(), l = D.answers[L.value], r = D.answers[R.value];
  $('#count').textContent = 'Showing ' + Math.min(shown, qs.length) + ' of ' + qs.length + ' matching questions (104 in total).';
  list.replaceChildren();
  qs.slice(0, shown).forEach(q => {
    const det = el('details', 'qitem'), sum = el('summary'), left = el('div', 'qtext');
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
    list.append(det);
  });
  $('#more').hidden = qs.length <= shown;
}
[L, R, T, F].forEach(x => x.onchange = () => { shown = 25; render(); });
Q.oninput = () => { shown = 25; render(); };
$('#more').onclick = () => { shown += 25; render(); };
render();
})();
