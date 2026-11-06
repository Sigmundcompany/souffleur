/* Souffleur : montage. Tout le calcul se fait sur l'appareil, rien n'est envoyé. */
(() => {
const SR = 48000;
const pause = () => new Promise(r => setTimeout(r));

/* ---------- Outils de signal ---------- */
function mono(buf) {
  const n = buf.length, out = new Float32Array(n), c = buf.numberOfChannels;
  for (let k = 0; k < c; k++) {
    const d = buf.getChannelData(k);
    for (let i = 0; i < n; i++) out[i] += d[i] / c;
  }
  return out;
}
function makeFFT(N) {
  const cos = new Float64Array(N / 2), sin = new Float64Array(N / 2), rev = new Uint32Array(N);
  for (let i = 0; i < N / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / N); sin[i] = Math.sin(2 * Math.PI * i / N); }
  const bits = Math.log2(N);
  for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b); rev[i] = r; }
  return (re, im, inverse) => {
    for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= N; size *= 2) {
      const half = size / 2, step = N / size;
      for (let s = 0; s < N; s += size) {
        for (let k = 0, t = 0; k < half; k++, t += step) {
          const wr = cos[t], wi = sign * sin[t], a = s + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        }
      }
    }
  };
}
function frameDb(x, sr, hop = .01, win = .02) {
  const h = Math.round(sr * hop), w = Math.round(sr * win), n = Math.max(0, Math.floor((x.length - w) / h));
  const out = new Float32Array(n);
  for (let f = 0; f < n; f++) {
    let e = 0; const o = f * h;
    for (let i = 0; i < w; i += 2) e += x[o + i] * x[o + i];
    out[f] = 10 * Math.log10(e / (w / 2) + 1e-10);
  }
  return out;
}
function speechThreshold(db) {
  const s = Float32Array.from(db).sort();
  const floor = s[Math.floor(s.length * .1)] ?? -100, speech = s[Math.floor(s.length * .95)] ?? -20;
  return { floor, speech, thr: Math.max(floor + (speech - floor) * .3, speech - 35) };
}

/* ---------- Détection des blancs ---------- */
function detectSilences(x, sr, { minPause = .5, keep = .15 } = {}) {
  const hop = .01, db = frameDb(x, sr, hop), { thr } = speechThreshold(db);
  const quiet = Array.from(db, v => v < thr);
  for (let i = 0; i < quiet.length;) { /* un petit bruit isolé ne casse pas un blanc */
    if (quiet[i]) { i++; continue; }
    let j = i; while (j < quiet.length && !quiet[j]) j++;
    if (j - i < 6 && i > 0 && j < quiet.length) for (let k = i; k < j; k++) quiet[k] = true;
    i = j;
  }
  const dur = x.length / sr, cuts = [];
  for (let i = 0; i < quiet.length;) {
    if (!quiet[i]) { i++; continue; }
    let j = i; while (j < quiet.length && quiet[j]) j++;
    const a = i * hop, b = j >= quiet.length ? dur : j * hop;
    const start = i === 0 ? 0 : a + keep, end = j >= quiet.length ? dur : b - keep;
    if ((b - a) >= minPause && end - start > .05) cuts.push({ start, end, kind: 'silence', on: true });
    i = j;
  }
  return cuts;
}

/* ---------- Détection des « euh » : voyelle tenue, voix stable, suivie ou précédée d'une pause ---------- */
function detectFillers(x, sr) {
  const step = Math.max(1, Math.round(sr / 16000)), fs = sr / step, n = Math.floor(x.length / step);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < step; k++) s += x[i * step + k]; y[i] = s / step; }
  const N = 512, hop = Math.round(fs * .01), frames = Math.max(0, Math.floor((n - N) / hop));
  const fft = makeFFT(N), re = new Float64Array(N), im = new Float64Array(N);
  const win = Float64Array.from({ length: N }, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / N));
  const B = 16, edges = Array.from({ length: B + 1 }, (_, b) => Math.round(150 * Math.pow(4000 / 150, b / B) / fs * N));
  const db = new Float32Array(frames), f0 = new Float32Array(frames), bands = new Array(frames), cent = new Float32Array(frames);
  const lagMin = Math.floor(fs / 400), lagMax = Math.ceil(fs / 70);
  for (let f = 0; f < frames; f++) {
    const o = f * hop; let e = 0;
    for (let i = 0; i < N; i++) { const v = y[o + i]; e += v * v; re[i] = v * win[i]; im[i] = 0; }
    db[f] = 10 * Math.log10(e / N + 1e-10);
    if (db[f] < -60) continue;
    let best = 0, lag = 0;
    for (let l = lagMin; l <= lagMax; l += 1) {
      let c = 0, a = 0, b = 0;
      for (let i = 0; i + l < N; i += 2) { const p = y[o + i], q = y[o + i + l]; c += p * q; a += p * p; b += q * q; }
      const r = c / Math.sqrt(a * b + 1e-12);
      if (r > best) { best = r; lag = l; }
    }
    if (best > .7) f0[f] = fs / lag;
    fft(re, im);
    const v = new Float32Array(B); let sum = 0, wsum = 0;
    for (let b = 0; b < B; b++) {
      let s = 0; for (let k = edges[b]; k < edges[b + 1] || k === edges[b]; k++) s += re[k] * re[k] + im[k] * im[k];
      v[b] = Math.log(s + 1e-9);
      const fc = 150 * Math.pow(4000 / 150, (b + .5) / B); sum += s; wsum += s * fc;
    }
    bands[f] = v; cent[f] = wsum / (sum + 1e-12);
    if (f % 1500 === 0) { /* laisse respirer l'interface via l'appelant */ }
  }
  const { thr } = speechThreshold(db);
  const stable = f => {
    if (!f0[f] || !f0[f - 1] || db[f] < thr + 6) return false;
    if (Math.abs(f0[f] / f0[f - 1] - 1) > .05) return false;
    let d = 0; for (let b = 0; b < B; b++) d += Math.abs(bands[f][b] - bands[f - 1][b]);
    return d / B < .55;
  };
  const out = [], hopS = hop / fs;
  for (let f = 1; f < frames;) {
    if (!stable(f)) { f++; continue; }
    let g = f, miss = 0, last = f;
    while (g < frames - 1 && miss < 3) { g++; if (stable(g)) { last = g; miss = 0; } else miss++; }
    const len = (last - f) * hopS;
    let c = 0; for (let k = f; k <= last; k++) c += cent[k]; c /= (last - f + 1);
    if (len >= .24 && len <= 1.6 && c > 350 && c < 1500) {
      let a = f, b = last;
      while (a > 0 && db[a - 1] > thr && f - a < 12) a--;
      while (b < frames - 1 && db[b + 1] > thr && b - last < 25) b++;
      const quietAt = (from, dir) => { let q = 0; for (let k = 1; k <= 14; k++) { const i = from + dir * k; if (i < 0 || i >= frames || db[i] < thr) q++; } return q >= 6; };
      if (quietAt(b, 1) || quietAt(a, -1)) out.push({ start: Math.max(0, a * hopS - .02), end: b * hopS + .04, kind: 'filler', on: true });
    }
    f = g + 1;
  }
  return out;
}

/* ---------- Réduction de bruit : soustraction spectrale ---------- */
async function denoise(x, sr, amount, progress) {
  const N = 1024, H = 256, fft = makeFFT(N), F = Math.floor((x.length - N) / H);
  if (F < 8) return x;
  const win = Float64Array.from({ length: N }, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / N));
  const e = new Float32Array(F);
  for (let f = 0; f < F; f++) { let s = 0; for (let i = 0; i < N; i += 8) { const v = x[f * H + i]; s += v * v; } e[f] = s; }
  const order = Array.from(e.keys()).sort((a, b) => e[a] - e[b]).slice(0, Math.max(8, Math.floor(F * .12)));
  const pick = order.filter((_, i) => i % Math.ceil(order.length / 300) === 0);
  const noise = new Float64Array(N / 2 + 1), re = new Float64Array(N), im = new Float64Array(N);
  for (const f of pick) {
    for (let i = 0; i < N; i++) { re[i] = x[f * H + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= N / 2; k++) noise[k] += Math.hypot(re[k], im[k]) / pick.length;
  }
  const over = 1.4 + amount * 1.6, floor = Math.pow(10, -(5 + amount * 19) / 20);
  const out = new Float32Array(x.length), prev = new Float64Array(N / 2 + 1).fill(1), g = new Float64Array(N / 2 + 1);
  for (let f = 0; f < F; f++) {
    const o = f * H;
    for (let i = 0; i < N; i++) { re[i] = x[o + i] * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k <= N / 2; k++) {
      const m = Math.hypot(re[k], im[k]) + 1e-12;
      let v = Math.max(floor, 1 - over * noise[k] / m);
      v = v > prev[k] ? v : prev[k] * .65 + v * .35;
      g[k] = v;
    }
    prev.set(g);
    for (let k = 0; k <= N / 2; k++) {
      const v = (g[Math.max(0, k - 1)] + 2 * g[k] + g[Math.min(N / 2, k + 1)]) / 4;
      re[k] *= v; im[k] *= v;
      if (k && k < N / 2) { re[N - k] = re[k]; im[N - k] = -im[k]; }
    }
    fft(re, im, true);
    for (let i = 0; i < N; i++) out[o + i] += re[i] / N * win[i] / 1.5;
    if (f % 3000 === 0) { progress?.(f / F); await pause(); }
  }
  return out;
}

/* ---------- Voix claire : filtres et compression ---------- */
async function voiceChain(x, sr) {
  const ctx = new OfflineAudioContext(1, x.length, sr);
  const b = ctx.createBuffer(1, x.length, sr); b.copyToChannel(x, 0);
  const src = ctx.createBufferSource(); src.buffer = b;
  const f = (type, freq, gain, Q) => { const n = ctx.createBiquadFilter(); n.type = type; n.frequency.value = freq; n.gain.value = gain; if (Q) n.Q.value = Q; return n; };
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -24; comp.ratio.value = 3; comp.knee.value = 12; comp.attack.value = .008; comp.release.value = .18;
  const chain = [f('highpass', 85, 0, .7), f('peaking', 280, -2, 1), f('peaking', 3200, 2.5, .9), f('highshelf', 9000, 2, 0), comp];
  let last = src; for (const n of chain) { last.connect(n); last = n; }
  last.connect(ctx.destination); src.start();
  return (await ctx.startRendering()).getChannelData(0);
}

/* ---------- Volume : norme BS.1770, cible réseaux sociaux ---------- */
function biquad(x, c) {
  const y = new Float32Array(x.length); let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = c[0] * x[i] + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = v; y[i] = v;
  }
  return y;
}
function kWeight(x, sr) {
  let w = 2 * Math.PI * 1500 / sr, cs = Math.cos(w), al = Math.sin(w) / (2 / Math.SQRT2 * 1), A = Math.pow(10, 4 / 40), r = 2 * Math.sqrt(A) * al;
  al = Math.sin(w) * Math.SQRT1_2; r = 2 * Math.sqrt(A) * al;
  let a0 = (A + 1) - (A - 1) * cs + r;
  const shelf = [A * ((A + 1) + (A - 1) * cs + r) / a0, -2 * A * ((A - 1) + (A + 1) * cs) / a0, A * ((A + 1) + (A - 1) * cs - r) / a0, 2 * ((A - 1) - (A + 1) * cs) / a0, ((A + 1) - (A - 1) * cs - r) / a0];
  w = 2 * Math.PI * 38 / sr; cs = Math.cos(w); al = Math.sin(w); a0 = 1 + al;
  const hp = [(1 + cs) / 2 / a0, -(1 + cs) / a0, (1 + cs) / 2 / a0, -2 * cs / a0, (1 - al) / a0];
  return biquad(biquad(x, shelf), hp);
}
function loudness(x, sr) {
  const k = kWeight(x, sr), b = Math.round(sr * .4), h = Math.round(sr * .1), z = [];
  for (let o = 0; o + b <= k.length; o += h) { let s = 0; for (let i = 0; i < b; i += 2) s += k[o + i] * k[o + i]; z.push(s / (b / 2)); }
  const L = v => -.691 + 10 * Math.log10(v + 1e-12), mean = a => a.reduce((p, c) => p + c, 0) / (a.length || 1);
  const g1 = z.filter(v => L(v) > -70);
  if (!g1.length) return -70;
  const rel = L(mean(g1)) - 10;
  return L(mean(g1.filter(v => L(v) > rel)));
}
function limit(x, sr, ceiling = .89) {
  const g = new Float32Array(x.length).fill(1), att = 1 / (sr * .004), rel = 1 / (sr * .12);
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > ceiling) g[i] = ceiling / a; }
  for (let i = x.length - 2; i >= 0; i--) g[i] = Math.min(g[i], g[i + 1] + att);
  for (let i = 1; i < x.length; i++) g[i] = Math.min(g[i], g[i - 1] + rel);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * g[i];
  return out;
}
function joinRanges(x, sr, ranges, fade = .008) {
  const total = ranges.reduce((s, r) => s + Math.round(r.end * sr) - Math.round(r.start * sr), 0);
  const out = new Float32Array(total), f = Math.round(sr * fade); let o = 0;
  for (const r of ranges) {
    const a = Math.round(r.start * sr), n = Math.round(r.end * sr) - a;
    out.set(x.subarray(a, a + n), o);
    for (let i = 0; i < Math.min(f, n / 2); i++) { const v = i / f; out[o + i] *= v; out[o + n - 1 - i] *= v; }
    o += n;
  }
  return out;
}

/* ---------- État du montage ---------- */
const $ = id => document.getElementById(id);
const O = Object.assign({ pause: .5, cutSilence: true, cutFiller: true, noise: true, noiseAmt: 60, voice: true, loud: true }, store.get('souffleur.montage', {}));
const saveO = () => store.set('souffleur.montage', O);
let blob = null, name = 'video', raw = null, dur = 0, stage1 = null, final = null;
let cuts = [], manual = [], fillers = [], markStart = null, zoom = 1;
let ctx = null, rawBuf = null, doneBuf = null, srcNode = null, ab = 'after', playing = false, watcher = 0, procRun = 0, outBlob = null, cancelled = false;

function kept() {
  const on = cuts.filter(c => c.on).map(c => [c.start, c.end]).sort((a, b) => a[0] - b[0]);
  const out = []; let t = 0;
  for (const [a, b] of on) { if (a > t + .04) out.push({ start: t, end: a }); t = Math.max(t, b); }
  if (dur > t + .04) out.push({ start: t, end: dur });
  return out;
}
function rebuildCuts() {
  const auto = [];
  if (O.cutSilence) auto.push(...detectSilences(raw, SR, { minPause: O.pause }));
  if (O.cutFiller) auto.push(...fillers.map(c => ({ ...c })));
  cuts = [...auto, ...manual];
  refresh();
}
function refresh() {
  const k = kept(), out = k.reduce((s, r) => s + r.end - r.start, 0);
  const n = kind => cuts.filter(c => c.on && c.kind === kind).length;
  const bits = [`${fmtTime(dur)} → <b>${fmtTime(out)}</b>`];
  bits.push(`${n('silence')} blanc${n('silence') > 1 ? 's' : ''}`, `${n('filler')} « euh »`);
  if (n('manual')) bits.push(`${n('manual')} coupe${n('manual') > 1 ? 's' : ''} à la main`);
  $('cutStats').innerHTML = bits.join(' · ');
  draw();
  scheduleLoud();
}

/* ---------- Son traité ---------- */
function status(msg, pct) {
  const s = $('cutBusy');
  s.hidden = !msg;
  if (msg) { $('cutBusyMsg').textContent = msg; $('cutBar').style.width = (pct ?? 0) + '%'; $('cutBar').parentElement.hidden = pct == null; }
}
async function process1() {
  const run = ++procRun;
  $('cutProc').textContent = 'Traitement du son en cours…';
  let x = raw;
  if (O.noise) x = await denoise(x, SR, O.noiseAmt / 100);
  if (run !== procRun) return;
  if (O.voice) x = await voiceChain(x, SR);
  if (run !== procRun) return;
  stage1 = x;
  applyLoud();
}
function applyLoud() {
  let x = stage1;
  if (O.loud) {
    const k = kept(); const l = loudness(k.length ? joinRanges(x, SR, k, 0) : x, SR);
    const g = Math.pow(10, Math.max(-20, Math.min(24, -14 - l)) / 20);
    const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[i] * g;
    x = limit(y, SR);
  }
  final = x;
  doneBuf = ctx.createBuffer(1, x.length, SR); doneBuf.copyToChannel(x, 0);
  $('cutProc').textContent = '';
  if (playing && ab === 'after') startAudio($('cutVideo').currentTime);
}
let loudT = 0;
function scheduleLoud() { clearTimeout(loudT); loudT = setTimeout(() => { if (stage1) applyLoud(); }, 400); }
let procT = 0;
function scheduleProc() { clearTimeout(procT); procT = setTimeout(process1, 250); }

/* ---------- Lecture ---------- */
function stopAudio() { if (srcNode) { try { srcNode.stop(); } catch {} srcNode.disconnect(); srcNode = null; } }
function startAudio(t) {
  stopAudio();
  const b = ab === 'after' && doneBuf ? doneBuf : rawBuf;
  if (!b) return;
  srcNode = ctx.createBufferSource(); srcNode.buffer = b; srcNode.connect(ctx.destination); srcNode.start(0, t);
}
function nextKept(t) {
  const k = kept();
  for (const r of k) { if (t < r.end - .02) return t >= r.start ? t : r.start; }
  return null;
}
function setPlay(on) {
  const v = $('cutVideo');
  playing = on;
  $('cutPlayIcon').setAttribute('href', on ? '#i-pause' : '#i-play');
  $('cutPlay').setAttribute('aria-label', on ? 'Mettre en pause' : 'Lire');
  clearInterval(watcher);
  if (!on) { v.pause(); stopAudio(); return; }
  ctx.resume();
  let t = nextKept(v.currentTime);
  if (t === null) t = nextKept(0) ?? 0;
  v.currentTime = t; v.play().catch(() => {}); startAudio(t);
  watcher = setInterval(() => {
    const now = v.currentTime, n = nextKept(now);
    if (n === null) { setPlay(false); return; }
    if (n > now + .03) { v.currentTime = n; startAudio(n); }
    playhead();
  }, 30);
}
function playhead() {
  const t = $('cutVideo').currentTime, w = $('wave').clientWidth;
  $('cutHead').style.transform = `translateX(${(t / dur * w).toFixed(1)}px)`;
  $('cutTime').textContent = clockTime(t);
}

/* ---------- Timeline ---------- */
let peaks = null;
function draw() {
  const cv = $('wave'), box = $('cutScroll');
  if (!raw || !box.clientWidth) return;
  const w = Math.min(14000, box.clientWidth * zoom), h = cv.parentElement.clientHeight, dpr = Math.min(2, devicePixelRatio || 1);
  cv.style.width = w + 'px'; cv.parentElement.style.width = w + 'px';
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const g = cv.getContext('2d'), css = getComputedStyle(document.documentElement);
  const col = n => css.getPropertyValue(n).trim();
  g.scale(dpr, dpr);
  const cols = Math.floor(w), step = raw.length / cols, mid = h / 2;
  if (!peaks || peaks.length !== cols) {
    peaks = new Float32Array(cols);
    for (let i = 0; i < cols; i++) { let m = 0; const a = Math.floor(i * step), b = Math.floor((i + 1) * step); for (let k = a; k < b; k += 16) { const v = Math.abs(raw[k]); if (v > m) m = v; } peaks[i] = m; }
    const top = peaks.reduce((a, b) => Math.max(a, b), .01);
    for (let i = 0; i < cols; i++) peaks[i] = Math.pow(peaks[i] / top, .7);
  }
  const X = t => t / dur * w;
  g.fillStyle = col('--paper');
  for (let i = 0; i < cols; i++) { const v = Math.max(1, peaks[i] * (mid - 4)); g.fillRect(i, mid - v, 1, v * 2); }
  for (const c of cuts) {
    const x = X(c.start), cw = Math.max(2, X(c.end) - x);
    if (c.on) {
      g.fillStyle = col('--panel'); g.globalAlpha = .82; g.fillRect(x, 0, cw, h); g.globalAlpha = 1;
      g.fillStyle = c.kind === 'filler' ? col('--cue') : c.kind === 'manual' ? col('--rec') : col('--muted');
      g.fillRect(x, 0, cw, 4); g.fillRect(x, h - 4, cw, 4);
    } else {
      g.strokeStyle = col('--muted'); g.setLineDash([3, 3]); g.strokeRect(x + .5, .5, cw - 1, h - 1); g.setLineDash([]);
    }
  }
  if (markStart !== null) { g.fillStyle = col('--rec'); g.fillRect(X(markStart), 0, 2, h); }
  playhead();
}
function onTimeline(e) {
  const r = $('wave').getBoundingClientRect(), t = Math.max(0, Math.min(dur, (e.clientX - r.left) / r.width * dur));
  const hit = cuts.find(c => t >= c.start - .02 && t <= c.end + .02);
  if (hit) { hit.on = !hit.on; refresh(); }
  const was = playing; if (was) setPlay(false);
  $('cutVideo').currentTime = t; playhead();
  if (was) setPlay(true);
}
function markCut() {
  const t = $('cutVideo').currentTime, b = $('cutMark');
  if (markStart === null) { markStart = t; b.textContent = 'Couper jusqu\'ici'; b.classList.add('danger'); draw(); return; }
  const s = Math.min(markStart, t), e = Math.max(markStart, t);
  markStart = null; b.textContent = 'Début de coupe'; b.classList.remove('danger');
  if (e - s > .05) { const c = { start: s, end: e, kind: 'manual', on: true }; manual.push(c); cuts.push(c); }
  refresh();
}

/* ---------- Export ---------- */
let libReady = null;
function loadLib() {
  return libReady ||= new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = 'vendor/mediabunny.min.js'; s.onload = () => res(window.Mediabunny); s.onerror = () => { libReady = null; rej(new Error('lib')); };
    document.head.append(s);
  });
}
async function exportVideo() {
  setPlay(false); cancelled = false;
  if (!final) { $('cutProc').textContent = 'Le son est encore en traitement, réessaie dans un instant.'; return; }
  if (!window.VideoEncoder || !window.AudioEncoder) return fail("Ce navigateur ne sait pas exporter la vidéo. Utilise Chrome, Edge, ou Safari à jour.");
  status('Préparation de l\'export…', 0);
  let input, output;
  try {
    const M = await loadLib();
    input = new M.Input({ source: new M.BlobSource(blob), formats: M.ALL_FORMATS });
    const track = await input.getPrimaryVideoTrack();
    if (!track) throw new Error('video');
    const size = { width: track.displayWidth, height: track.displayHeight };
    const vCodec = await M.getFirstEncodableVideoCodec(['avc', 'vp9', 'vp8'], size);
    const aCodec = await M.getFirstEncodableAudioCodec(['aac', 'opus'], { numberOfChannels: 1, sampleRate: SR });
    if (!vCodec || !aCodec) throw new Error('codec');
    const mp4 = vCodec === 'avc';
    output = new M.Output({ format: mp4 ? new M.Mp4OutputFormat({ fastStart: 'in-memory' }) : new M.WebMOutputFormat(), target: new M.BufferTarget() });
    const vs = new M.VideoSampleSource({ codec: vCodec, quality: M.QUALITY_VERY_HIGH, sizeChangeBehavior: 'contain' });
    const as = new M.AudioBufferSource({ codec: aCodec, quality: M.QUALITY_VERY_HIGH });
    output.addVideoTrack(vs, { rotation: track.rotation }); output.addAudioTrack(as);
    await output.start();

    const ranges = kept(), total = ranges.reduce((s, r) => s + r.end - r.start, 0);
    const audio = joinRanges(final, SR, ranges);
    const pushAudio = async () => {
      const chunk = SR * 2;
      for (let o = 0; o < audio.length && !cancelled; o += chunk) {
        const part = audio.subarray(o, Math.min(audio.length, o + chunk));
        const b = new AudioBuffer({ length: part.length, numberOfChannels: 1, sampleRate: SR });
        b.copyToChannel(part, 0);
        await as.add(b);
      }
      as.close();
    };
    const pushVideo = async () => {
      const sink = new M.VideoSampleSink(track);
      let base = 0, last = -1;
      for (const r of ranges) {
        for await (const s of sink.samples(r.start, r.end)) {
          if (cancelled) { s.close(); break; }
          const t = Math.max(0, s.timestamp - r.start) + base;
          if (s.timestamp < r.end && t > last + .0005) {
            s.setTimestamp(t); last = t;
            await vs.add(s);
            status('Export de la vidéo…', Math.min(99, t / total * 100));
          }
          s.close();
        }
        base += r.end - r.start;
      }
      vs.close();
    };
    await Promise.all([pushAudio(), pushVideo()]);
    if (cancelled) { await output.cancel(); status(''); return; }
    await output.finalize();
    outBlob = new Blob([output.target.buffer], { type: mp4 ? 'video/mp4' : 'video/webm' });
    showDone(mp4 ? 'mp4' : 'webm', total);
  } catch (err) {
    console.error(err);
    try { await output?.cancel(); } catch {}
    fail(err.message === 'codec' ? "Ce navigateur ne sait pas créer la vidéo finale. Utilise Chrome, Edge, ou Safari à jour."
      : err.message === 'lib' ? "L'outil d'export n'a pas pu se charger. Vérifie ta connexion puis réessaie."
      : "L'export a échoué. Réessaie, ou raccourcis la vidéo si elle est très longue.");
  } finally { input?.dispose?.(); }
}
function fail(msg) { status(''); $('cutProc').textContent = msg; }
let outName = '';
function showDone(ext, total) {
  status('');
  outName = `${name}-monte.${ext}`;
  const v = $('doneVideo'); v.src = URL.createObjectURL(outBlob);
  $('doneInfo').textContent = `${fmtTime(total)} · ${(outBlob.size / 1048576).toFixed(1).replace('.', ',')} Mo`;
  const f = new File([outBlob], outName, { type: outBlob.type });
  $('doneShare').hidden = !(navigator.canShare && navigator.canShare({ files: [f] }));
  $('cutDone').hidden = false;
}

/* ---------- Ouverture ---------- */
async function open(file, fileName) {
  blob = file; name = (fileName || 'video').replace(/\.[^.]+$/, '');
  $('desk').hidden = true; $('stage').hidden = true; $('cut').hidden = false; $('cutDone').hidden = true;
  $('cutProc').textContent = ''; markStart = null; manual = []; zoom = 1; peaks = null; final = null; stage1 = null; doneBuf = null;
  ctx ||= new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  const v = $('cutVideo'); v.src = URL.createObjectURL(blob);
  status('Analyse du son…');
  await pause();
  try {
    const off = new OfflineAudioContext(1, 1, SR);
    const buf = await off.decodeAudioData(await blob.arrayBuffer());
    raw = mono(buf); dur = raw.length / SR;
    rawBuf = ctx.createBuffer(1, raw.length, SR); rawBuf.copyToChannel(raw, 0);
  } catch (err) {
    console.error(err); status('');
    $('cutProc').textContent = "Impossible de lire le son de cette vidéo. Essaie un fichier MP4.";
    return;
  }
  await pause();
  status('Recherche des blancs et des « euh »…');
  await pause();
  fillers = detectFillers(raw, SR);
  rebuildCuts();
  status('');
  process1();
}
function close() {
  setPlay(false); procRun++;
  const v = $('cutVideo'); URL.revokeObjectURL(v.src); v.removeAttribute('src'); v.load();
  $('cut').hidden = true; $('desk').hidden = false;
  raw = final = stage1 = rawBuf = doneBuf = null; blob = outBlob = null;
}

/* ---------- Commandes ---------- */
function bind() {
  const check = (id, key, after) => { const e = $(id); e.checked = O[key]; e.onchange = () => { O[key] = e.checked; saveO(); after(); }; };
  check('oSilence', 'cutSilence', rebuildCuts);
  check('oFiller', 'cutFiller', rebuildCuts);
  check('oNoise', 'noise', scheduleProc);
  check('oVoice', 'voice', scheduleProc);
  check('oLoud', 'loud', () => { if (stage1) applyLoud(); });
  const p = $('oPause'), po = $('oPauseOut'), show = () => { po.textContent = O.pause.toFixed(1).replace('.', ',') + ' s'; };
  p.value = O.pause; show();
  p.oninput = () => { O.pause = +p.value; saveO(); show(); rebuildCuts(); };
  const n = $('oNoiseAmt'), no = $('oNoiseOut'), showN = () => { no.textContent = O.noiseAmt + ' %'; };
  n.value = O.noiseAmt; showN();
  n.oninput = () => { O.noiseAmt = +n.value; saveO(); showN(); if (O.noise) scheduleProc(); };
  $('cutPlay').onclick = () => setPlay(!playing);
  $('cutClose').onclick = close;
  $('cutMark').onclick = markCut;
  $('cutExport').onclick = exportVideo;
  $('cutCancel').onclick = () => { cancelled = true; };
  $('cutZoomIn').onclick = () => { zoom = Math.min(16, zoom * 2); peaks = null; draw(); };
  $('cutZoomOut').onclick = () => { zoom = Math.max(1, zoom / 2); peaks = null; draw(); };
  $('wave').onclick = onTimeline;
  for (const b of document.querySelectorAll('[data-ab]')) b.onclick = () => {
    ab = b.dataset.ab;
    document.querySelectorAll('[data-ab]').forEach(x => x.setAttribute('aria-pressed', x === b));
    if (playing) startAudio($('cutVideo').currentTime);
  };
  $('doneBack').onclick = () => { URL.revokeObjectURL($('doneVideo').src); $('doneVideo').removeAttribute('src'); $('cutDone').hidden = true; };
  $('doneSave').onclick = () => {
    const a = document.createElement('a'); a.href = $('doneVideo').src; a.download = outName;
    document.body.append(a); a.click(); a.remove();
  };
  $('doneShare').onclick = async () => { try { await navigator.share({ files: [new File([outBlob], outName, { type: outBlob.type })] }); } catch {} };
  addEventListener('resize', () => { if (!$('cut').hidden) { peaks = null; draw(); } });
  addEventListener('beforeunload', e => { if (blob) e.preventDefault(); });
}
bind();
window.Montage = {
  open,
  engine: { detectSilences, detectFillers, denoise, voiceChain, loudness, limit, joinRanges },
  state: () => ({ dur, cuts, kept: kept(), final: !!final, proc: $('cutProc').textContent, done: !$('cutDone').hidden, size: outBlob?.size, type: outBlob?.type, busy: !$('cutBusy').hidden, msg: $('cutBusyMsg').textContent }),
  result: () => outBlob,
};
})();
