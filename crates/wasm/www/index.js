// ─── WASM bootstrap ──────────────────────────────────────────────────────────
// Die gesamte Spiel-Logik (Zuggenerierung, Suche, Bewertung) liegt in Rust und
// wird per wasm-bindgen exportiert. `engine` hält den einzigen globalen Spiel-
// zustand; alles UI-seitige (Highlights, Animation, Chat) lebt in dieser Datei.
import init, { WasmEngine } from './pkg/chaturaji_wasm.js';
await init();
const engine = new WasmEngine();

// ─── Konstanten ───────────────────────────────────────────────────────────────
const BOARD_SIZE = 660;
const CELL = BOARD_SIZE / 8;
const MOVE_ANIM_MS = 180; // Animationsdauer in ms (0 = deaktiviert)

const COLOR_HEX = {
  Red: '#e74c3c', Blue: '#3498db', Yellow: '#f1c40f', Green: '#2ecc71',
};
const BOARD_LIGHT = '#f0d9b5';
const BOARD_DARK  = '#b58863';

// ─── SVG Bild-Cache ───────────────────────────────────────────────────────────
// Dateien unter: www/pieces/{farbe}-{figur}.svg (z.B. red-king.svg)
const imgCache = {};

function getPieceImage(color, kind) {
  const key = color + '-' + kind;
  if (key in imgCache) return imgCache[key];
  imgCache[key] = null;
  const img = new Image();
  img.onload  = () => { imgCache[key] = img; draw(); };
  img.onerror = () => { imgCache[key] = null; };
  img.src = 'pieces/' + color.toLowerCase() + '-' + kind.toLowerCase() + '.svg';
  return null;
}

// ─── DOM-Referenzen ───────────────────────────────────────────────────────────
const canvas     = document.getElementById('board');
const ctx        = canvas.getContext('2d');
const overlay    = document.getElementById('overlay-msg');
const turnEl     = document.getElementById('turn-indicator');
const engineInfo = document.getElementById('engine-info');
const moveList   = document.getElementById('movelist');
const depthInput = document.getElementById('depth');
const depthVal   = document.getElementById('depth-val');
const netStatus  = document.getElementById('net-status');
const netBars    = document.getElementById('net-eval-bars');
const unloadBtn  = document.getElementById('btn-unload-net');

depthInput.addEventListener('input', () => { depthVal.textContent = depthInput.value; });

// ─── Zustand ──────────────────────────────────────────────────────────────────
// `state` ist eine Momentaufnahme aus der Engine (Felder, Punkte, aktive
// Spieler, Spielende). Wird nach jedem Zug per `engine.get_state()` neu geholt
// statt inkrementell gepflegt – die Rust-Seite bleibt die Single Source of Truth.
let selected   = null;
let legalMoves = [];
let state      = null;
let animReq    = null; // laufender requestAnimationFrame-Handle

// ─── Zeichnen ─────────────────────────────────────────────────────────────────
function draw() {
  state = engine.get_state();
  ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);
  drawSquares();
  drawHighlights();
  drawPieces();
  updateScores();
  updateStatus();
  updateNetEval();
}

// Engine indiziert Felder als rank*8 + file mit a1=0, h8=63. Canvas-Y wächst
// nach unten, daher der Flip (7 - rank): rank 0 (Reihe 1) liegt am unteren
// Rand. Die UI rendert immer aus Sicht von Rot unten — keine Brettdrehung.
function sq2xy(sq) {
  return [sq % 8 * CELL, (7 - Math.floor(sq / 8)) * CELL];
}

function xy2sq(x, y) {
  const f = Math.floor(x / CELL);
  const r = 7 - Math.floor(y / CELL);
  return (f >= 0 && f < 8 && r >= 0 && r < 8) ? r * 8 + f : null;
}

function drawSquares() {
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      ctx.fillStyle = (f + r) % 2 === 0 ? BOARD_LIGHT : BOARD_DARK;
      ctx.fillRect(f * CELL, (7 - r) * CELL, CELL, CELL);
    }
  }
  ctx.fillStyle = '#777';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let f = 0; f < 8; f++)
    ctx.fillText(String.fromCharCode(97 + f), f * CELL + CELL / 2, BOARD_SIZE - 3);
  ctx.textAlign = 'left';
  for (let r = 0; r < 8; r++)
    ctx.fillText(r + 1, 3, (7 - r) * CELL + 13);
}

function drawHighlights() {
  if (selected === null) return;
  const [sx, sy] = sq2xy(selected);
  ctx.fillStyle = 'rgba(20,85,30,0.75)';
  ctx.fillRect(sx, sy, CELL, CELL);
  for (const mv of legalMoves) {
    const [tx, ty] = sq2xy(mv.to);
    if (mv.captures) {
      ctx.strokeStyle = 'rgba(20,85,30,0.7)';
      ctx.lineWidth = 4;
      ctx.strokeRect(tx + 2, ty + 2, CELL - 4, CELL - 4);
    } else {
      ctx.fillStyle = 'rgba(20,85,30,0.35)';
      ctx.beginPath();
      ctx.arc(tx + CELL / 2, ty + CELL / 2, CELL / 6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// `isActive` = ist die Farbe noch im Spiel? Ausgeschiedene Spieler bleiben
// als gedimmte Steine sichtbar (sie können noch geschlagen werden und zählen
// weiter Punkte für den Schlagenden).
function drawOnePiece(piece, x, y, isActive) {
  const img = getPieceImage(piece.color, piece.kind);
  if (!img) return; // SVG noch nicht geladen → in dem Frame nichts zeichnen,
                    // onload löst danach automatisch ein draw() aus.
  ctx.globalAlpha = isActive ? 1.0 : 0.3;
  const pad = CELL * 0.06;
  ctx.drawImage(img, x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
  ctx.globalAlpha = 1;
}

// `skipSq` lässt ein Feld ungezeichnet — wird während einer Animation für
// das Startfeld der wandernden Figur verwendet, damit sie nicht doppelt
// (statisch + animiert) erscheint.
function drawPieces(skipSq = -1) {
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let sq = 0; sq < 64; sq++) {
    if (sq === skipSq) continue;
    const piece = state.squares[sq];
    if (!piece) continue;
    const colorIdx = ['Red','Blue','Yellow','Green'].indexOf(piece.color);
    drawOnePiece(piece, ...sq2xy(sq), state.active[colorIdx]);
  }
}

// engine-Feldstring (z.B. "b6") → Feldindex 0-63
function engineSqIdx(s) {
  return (parseInt(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97);
}

// Figur von fromSq nach toSq animieren; state muss bereits der Zustand
// NACH dem Zug sein (damit geschlagene Figuren korrekt verschwunden sind).
function animateMove(fromSq, toSq, movingPiece, onDone) {
  if (animReq !== null) { cancelAnimationFrame(animReq); animReq = null; }
  if (!movingPiece || MOVE_ANIM_MS <= 0) { onDone(); return; }

  const [fx, fy] = sq2xy(fromSq);
  const [tx, ty] = sq2xy(toSq);
  const colorIdx = ['Red','Blue','Yellow','Green'].indexOf(movingPiece.color);
  const t0 = performance.now();

  function frame(now) {
    const p = Math.min((now - t0) / MOVE_ANIM_MS, 1);
    const e = p < 0.5 ? 2 * p * p : -1 + (4 - 2 * p) * p; // ease-in-out

    ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);
    drawSquares();
    drawHighlights();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    drawPieces(toSq); // alle Figuren außer der animierten
    drawOnePiece(movingPiece, fx + (tx - fx) * e, fy + (ty - fy) * e, state.active[colorIdx]);

    if (p < 1) {
      animReq = requestAnimationFrame(frame);
    } else {
      animReq = null;
      onDone();
    }
  }
  animReq = requestAnimationFrame(frame);
}

// ─── Score & Status ───────────────────────────────────────────────────────────
function updateScores() {
  ['red','blue','yellow','green'].forEach((c, i) => {
    document.getElementById('score-' + c).textContent = state.scores[i];
  });
}

function updateStatus() {
  if (state.is_over) {
    const w = state.winner || 'Niemand';
    overlay.classList.add('hidden');
    turnEl.innerHTML = 'Spiel beendet &ndash; Sieger: <strong>' + w + '</strong>';
  } else {
    overlay.classList.add('hidden');
    const active = ['Red','Blue','Yellow','Green'].filter((_, i) => state.active[i]);
    turnEl.innerHTML = 'Zug: <strong style="color:' + COLOR_HEX[state.to_move] + '">'
      + state.to_move + '</strong> &nbsp;|&nbsp; Aktiv: ' + active.join(', ');
  }
}

// ─── Netz-Bewertungsbalken ────────────────────────────────────────────────────
// Liefert `null`, wenn kein neuronales Netz geladen ist — dann arbeitet die
// Engine rein mit Alpha-Beta und es gibt nichts anzuzeigen.
function updateNetEval() {
  const vals = engine.evaluate_position();
  if (!vals) { netBars.classList.add('hidden'); return; }
  netBars.classList.remove('hidden');
  ['red','blue','yellow','green'].forEach((c, i) => {
    const v = vals[i];
    document.getElementById('val-' + c).textContent = v.toFixed(3);
    // tanh-Ausgabe [-1,1] → Balkenbreite [0,100]%
    document.getElementById('bar-' + c).style.width = ((v + 1) / 2 * 100).toFixed(1) + '%';
  });
}

// ─── Klick-Interaktion ────────────────────────────────────────────────────────
// Erster Klick wählt eine eigene Figur aus, zweiter Klick zieht (oder hebt
// die Auswahl wieder auf). Während einer Top-Zug-Vorschau frisst der erste
// Klick nur die Vorschau, ohne ein neues Feld zu selektieren.
canvas.addEventListener('click', (e) => {
  if (state && state.is_over) return;
  if (previewMv !== null) { clearPreview(); draw(); return; }
  // Skalierung Canvas-Pixel ↔ CSS-Pixel: das Brett kann per CSS verkleinert
  // sein, intern wird aber immer mit BOARD_SIZE gerechnet.
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (BOARD_SIZE / rect.width);
  const y = (e.clientY - rect.top)  * (BOARD_SIZE / rect.height);
  const sq = xy2sq(x, y);
  if (sq === null) return;

  // Zug ausführen wenn Zielfeld einer legalen Bewegung entspricht
  if (selected !== null) {
    const mv = legalMoves.find(m => m.to === sq);
    if (mv) {
      const fromSq = mv.from;
      const movingPiece = state.squares[fromSq];
      if (engine.apply_move(mv.notation)) {
        appendMoveLog(mv);
        state = engine.get_state();
        selected = null; legalMoves = [];
        animateMove(fromSq, mv.to, movingPiece, () => draw());
        return;
      }
    }
    if (sq === selected) { selected = null; legalMoves = []; draw(); return; }
  }

  // Figur auswählen
  const piece = state.squares[sq];
  if (piece && piece.color === state.to_move) {
    selected = sq;
    legalMoves = engine.legal_moves_from(sq);
  } else {
    selected = null; legalMoves = [];
  }
  draw();
});

// H\u00E4ngt einen Halbzug an die Zugliste an. Vier Halbz\u00FCge bilden eine "Runde"
// (Rot, Blau, Gelb, Gr\u00FCn) und werden in einem einzigen <li> zusammengefasst.
// F\u00FCr den ersten Zug einer Runde wird ein neues <li> angelegt, danach wird
// der gesamte Zeileninhalt re-gerendert (einfacher als einzelne Zellen zu
// patchen, kostet kaum etwas bei nur 4 Spalten).
function appendMoveLog(mv) {
  const text = mv.notation
    + (mv.captures ? ' \u00D7' : '')
    + (mv.promoted ? ' =B' : ''); // chaturaji: Bauer wird beim Promotionsfeld zum Boat (=B)
  const entry = { text, color: PLY_COLORS[plyCount % 4] };
  moveLogEntries.push(entry);
  if (plyCount % 4 === 0) {
    const li = document.createElement('li');
    li.innerHTML = roundHtml([entry]);
    moveList.appendChild(li);
  } else {
    const li = moveList.lastElementChild;
    li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount + 1));
  }
  plyCount++;
  moveList.scrollTop = moveList.scrollHeight;
}

// ─── Buttons ──────────────────────────────────────────────────────────────────
document.getElementById('btn-engine-move').addEventListener('click', () => {
  if (state && state.is_over) return;
  clearPreview();
  const depth = parseInt(depthInput.value);
  engineInfo.textContent = 'Suche bei Tiefe ' + depth + '\u2026';
  // setTimeout(…, 10) gibt dem Browser einen Tick, um den "Suche…"-Text
  // anzuzeigen, bevor die synchrone (und potenziell sekundenlange) WASM-
  // Suche den Main-Thread blockiert.
  setTimeout(() => {
    const t0  = performance.now();
    const res = engine.best_move(depth);
    const ms  = (performance.now() - t0).toFixed(0);
    let fromSq = -1, toSq = -1, movingPiece = null;
    if (res.best_move) {
      fromSq = engineSqIdx(res.best_move.slice(0, 2));
      toSq   = engineSqIdx(res.best_move.slice(2, 4));
      movingPiece = state?.squares[fromSq] ?? null;
      engine.apply_move(res.best_move);
      appendMoveLog({ notation: res.best_move, captures: false, promoted: false });
      state = engine.get_state();
    }
    const netNote = res.used_network ? ' \uD83E\uDDE0 Netz aktiv' : '';
    engineInfo.textContent = 'Tiefe ' + res.depth
      + ' \u00B7 ' + (res.nodes || 0).toLocaleString() + ' Knoten'
      + ' \u00B7 ' + ms + 'ms' + netNote;
    selected = null; legalMoves = [];
    animateMove(fromSq, toSq, movingPiece, () => draw());
  }, 10);
});

document.getElementById('btn-undo').addEventListener('click', () => {
  clearPreview();
  if (engine.undo()) {
    plyCount--;
    moveLogEntries.pop();
    if (plyCount % 4 === 0) {
      if (moveList.lastChild) moveList.removeChild(moveList.lastChild);
    } else {
      const li = moveList.lastElementChild;
      li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount));
    }
    selected = null; legalMoves = [];
    draw();
  }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  previewMv = null; previewRow = null;
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  selected = null; legalMoves = [];
  engineInfo.textContent = '';
  replayMoves = []; replayIdx = 0; replayEvals = [];
  allForfeits = []; appliedForfeitStack.length = 0; forfeitsAppliedAtIdx.length = 0;
  playerClocks = [null, null, null, null];
  prevPlayerClocks.length = 0;
  refreshAllScoreTimes();
  document.getElementById('replay-controls').classList.add('hidden');
  document.getElementById('top-moves-panel').classList.add('hidden');
  gameChat = []; gameChatUidToColor = {}; gameChatIdx = 0; maxChatFullMoveNr = 0;
  document.getElementById('chat-messages').innerHTML = '<p class="chat-empty">Kein Spiel geladen.</p>';
  document.getElementById('game-info').innerHTML = '';
  const gameNrEl = document.getElementById('gameNr');
  if (gameNrEl) gameNrEl.textContent = '';
  const defaults = ['Rot','Blau','Gelb','Grün'];
  ['red','blue','yellow','green'].forEach((c, i) => {
    const el = document.querySelector('.score.' + c + ' .score-name');
    if (el) el.textContent = defaults[i];
  });
  draw();
});

// ─── Netz laden / entladen ────────────────────────────────────────────────────
document.getElementById('net-file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const err = engine.load_network_json(ev.target.result);
    if (err) {
      netStatus.textContent = 'Fehler: ' + err;
      netStatus.className = 'err';
    } else {
      const info = engine.network_info();
      netStatus.textContent = '\u2705 ' + file.name
        + ' | Schritte: ' + info.steps.toLocaleString()
        + ' | ' + info.params.toLocaleString() + ' Parameter';
      netStatus.className = 'ok';
      unloadBtn.disabled = false;
      draw();
    }
  };
  reader.readAsText(file);
  // Input zurücksetzen damit dieselbe Datei erneut ladbar ist
  e.target.value = '';
});

unloadBtn.addEventListener('click', () => {
  engine.unload_network();
  netStatus.textContent = 'Kein Netz geladen \u2013 Engine nutzt Alpha-Beta';
  netStatus.className = '';
  unloadBtn.disabled = true;
  netBars.classList.add('hidden');
});

// ─── PGN ─────────────────────────────────────────────────────────────────────
document.getElementById('btn-load-pgn').addEventListener('click', () => {
  const pgn = document.getElementById('pgn-input').value.trim();
  if (!pgn) return;
  previewMv = null; previewRow = null;
  const err = engine.load_pgn(pgn);
  if (err) { alert('PGN-Fehler: ' + err); return; }
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  selected = null; legalMoves = [];
  draw();
});

document.getElementById('btn-export-pgn').addEventListener('click', () => {
  document.getElementById('pgn-output').textContent = engine.export_pgn();
});

// ─── Replay-Zustand ───────────────────────────────────────────────────────────
let replayMoves = [];
let replayIdx   = 0;
let replayEvals = [];   // vorberechnete Analysedaten (game_analysis/*.json)

// Forfeits/Resignations aus dem Chat (Engine erkennt nur Königsschläge).
// `appliedForfeitStack` = bereits angewandte Forfeits in chronologischer Reihen-
// folge; `forfeitsAppliedAtIdx[i]` = wie viele Forfeits direkt vor dem Zug an
// replayIdx=i angewandt wurden (für sauberes Undo).
let allForfeits          = [];
let appliedForfeitStack  = [];
let forfeitsAppliedAtIdx = [];

// Verbleibende Bedenkzeit pro Spieler in ms (null = noch kein Zug gespielt).
// `prevPlayerClocks[idx]` speichert den vorherigen Wert für sauberes Undo:
//   { pIdx: 0-3, prev: alter ms-Wert oder null }
let playerClocks     = [null, null, null, null];
let prevPlayerClocks = [];

const COLOR_NAMES = ['red', 'blue', 'yellow', 'green'];

function formatClock(ms) {
  if (ms == null) return '';
  return (Math.floor(ms / 100) / 10).toFixed(1); // truncate auf 1 Nachkommastelle
}

function updateScoreTime(pIdx) {
  const el = document.querySelector('.score-time[data-color="' + COLOR_NAMES[pIdx] + '"]');
  if (el) el.textContent = formatClock(playerClocks[pIdx]);
}

function refreshAllScoreTimes() {
  for (let i = 0; i < 4; i++) updateScoreTime(i);
}

// ─── Zugprotokoll-Zustand ─────────────────────────────────────────────────────
let plyCount = 0;
const moveLogEntries = []; // {text, color, score?}
const PLY_COLORS = ['red', 'blue', 'yellow', 'green'];

function roundHtml(entries) {
  const cells = Array.from({ length: 4 }, (_, i) => entries[i] ?? null);
  const moves = cells.map(e =>
    e ? `<span class="mv-cell mv-${e.color}">${e.text}</span>`
      : `<span class="mv-cell"></span>`
  ).join('');
  const scores = cells.map(e =>
    (e && e.score != null)
      ? `<span class="score-cell mv-${e.color}">${e.score}</span>`
      : `<span class="score-cell"></span>`
  ).join('');
  return moves + scores;
}

// chess.com nutzt für Chaturaji ein 14×14-Brett (a–n × 1–14): die äußeren
// 3 Reihen pro Seite sind Startbereiche der vier Spieler, das eigentliche
// Spielfeld ist das innere 8×8 (d4–k11 in cc-Notation = a1–h8 in Engine-
// Notation). Daher Offset 3 in beide Richtungen. Felder außerhalb des
// inneren Bretts liefern null — solche Züge kann die Engine nicht abbilden.
function ccSqToEngine(sq) {
  const file = sq.charCodeAt(0) - 97;        // a=0 … n=13
  const rank = parseInt(sq.slice(1), 10);    // 1-14
  const ef = file - 3;
  const er = rank - 3;
  if (ef < 0 || ef > 7 || er < 1 || er > 8) return null;
  return String.fromCharCode(97 + ef) + er;
}

function ccMoveToEngine(cc) {
  if (cc === '--') return '--';
  const s = cc.replace(/^[KBNR]/, '').replace(/[+#]$/, '');
  const sepIdx = s.search(/[-x]/);
  if (sepIdx === -1) return null;
  const from  = ccSqToEngine(s.slice(0, sepIdx));
  const toRaw = s.slice(sepIdx + 1).replace(/^[KBNR]/, '');
  const to    = ccSqToEngine(toRaw);
  return (from && to) ? from + to : null;
}

function ccMoveToDisplay(cc) {
  if (cc === '--') return '--';
  const piece   = /^[KBNR]/.test(cc) ? cc[0] : '';
  const hasX    = cc.includes('x');
  const suffix  = cc.endsWith('#') ? '#' : cc.endsWith('+') ? '+' : '';
  const s       = cc.replace(/^[KBNR]/, '').replace(/[+#]$/, '');
  const sepIdx  = s.search(/[-x]/);
  if (sepIdx === -1) return null;
  const from   = ccSqToEngine(s.slice(0, sepIdx));
  const toRaw  = s.slice(sepIdx + 1).replace(/^[KBNR]/, '');
  const to     = ccSqToEngine(toRaw);
  if (!from || !to) return null;
  return piece + from + (hasX ? 'x' : '-') + to + suffix;
}

// Parst chess.com-PGN4. Format-Eigenheiten:
//  - Header-Zeilen `[Tag "..."]` werden vorab entfernt.
//  - Rundenzähler stehen als `1.`, `2.` … (eine Runde = 4 Halbzüge).
//  - `--` markiert einen Spieler, der aussetzen muss (eliminiert / no-move).
//  - Optionaler Kommentar `{date=...}` direkt nach dem Zug enthält den
//    Server-Zeitstempel des Halbzugs — wird für die Chat-Synchronisation
//    benötigt (Nachrichten erscheinen, sobald ihr `time` ≤ Zug-`time`).
function parsePgn4Moves(pgn4) {
  const movesText = pgn4.replace(/^\[.*\]\s*$/gm, '').trim();
  const pattern = /(\d+)\.+|(--|[KBNR]?[a-n]\d{1,2}[-x][KBNR]?[a-n]\d{1,2}[+#]?)\s*(?:\{([^}]*)\})?/g;
  const moves = [];
  let currentFullMoveNr = 1;
  let m;
  while ((m = pattern.exec(movesText)) !== null) {
    if (m[1] !== undefined) { currentFullMoveNr = parseInt(m[1]); continue; }
    const cc = m[2];
    const comment = m[3] || '';
    const dateMatch  = comment.match(/date=(\S+)/);
    const clockMatch = comment.match(/clock=(\d+)/);
    const time  = dateMatch  ? new Date(dateMatch[1]).getTime() : null;
    const clock = clockMatch ? parseInt(clockMatch[1], 10)      : null; // ms verbleibend nach diesem Zug
    if (cc === '--') {
      moves.push({ display: '--', engine: '--', time, clock, fullMoveNr: currentFullMoveNr });
    } else {
      const eng = ccMoveToEngine(cc);
      const dsp = ccMoveToDisplay(cc);
      if (eng && dsp) moves.push({ display: dsp, engine: eng, time, clock, fullMoveNr: currentFullMoveNr });
    }
  }
  return moves;
}

function updateReplayPos() {
  document.getElementById('replay-pos').textContent = replayIdx + ' / ' + replayMoves.length;
}

// ─── Top-Züge ─────────────────────────────────────────────────────────────────
function engineMoveToDisplay(eng) {
  if (!eng || eng.length < 4) return eng;
  return eng.slice(0, 2) + '-' + eng.slice(2, 4);
}

// Vorschau: aktuell auf dem Brett gespielter Top-Zug (oder null)
let previewMv  = null;
let previewRow = null;

// Nimmt einen aktiven Vorschauzug zurück. Caller ist verantwortlich für draw()
// oder eine Folgeanimation.
function clearPreview() {
  if (previewMv === null) return;
  engine.undo();
  state = engine.get_state();
  selected = null; legalMoves = [];
  if (previewRow) previewRow.classList.remove('active');
  previewMv = null; previewRow = null;
}

function applyPreview(t, row) {
  const wasActive = previewMv === t.mv;
  clearPreview();
  if (wasActive) { draw(); return; }

  const fromSq = engineSqIdx(t.mv.slice(0, 2));
  const toSq   = engineSqIdx(t.mv.slice(2, 4));
  const movingPiece = state?.squares[fromSq] ?? null;
  // Vorberechnete Top-Züge enthalten keinen Promotionssuffix; falls der reine
  // Zug abgelehnt wird, ist es vermutlich eine Bauernumwandlung — Retry mit
  // 'p' (Promotion zu Boat, einzige Promotion in Chaturaji).
  if (engine.apply_move(t.mv) || engine.apply_move(t.mv + 'p')) {
    previewMv  = t.mv;
    previewRow = row;
    row.classList.add('active');
    state = engine.get_state();
    animateMove(fromSq, toSq, movingPiece, () => draw());
  }
}

function renderTopMoves() {
  const panel = document.getElementById('top-moves-panel');
  const list  = document.getElementById('top-moves-list');
  if (!panel || !list) return;

  // Vorberechnete Daten für die aktuelle Position (replayIdx zeigt auf nächsten Zug)
  const evalEntry = replayEvals[replayIdx] ?? null;
  const tops = evalEntry?.top ?? null;
  if (!tops || tops.length === 0) { panel.classList.add('hidden'); return; }

  const st = engine.get_state();
  const COLOR_CLASS = ['red', 'blue', 'yellow', 'green'];
  const playerIdx   = ['Red', 'Blue', 'Yellow', 'Green'].indexOf(st.to_move);
  const colorClass  = COLOR_CLASS[playerIdx] ?? '';
  const barColor    = COLOR_HEX[st.to_move] ?? '#f0c040';

  list.innerHTML = '';
  previewMv = null; previewRow = null;
  for (const t of tops) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'top-move-row';
    row.innerHTML =
      `<span class="top-move-notation ${colorClass}">${engineMoveToDisplay(t.mv)}</span>` +
      `<div class="top-move-bar-wrap"><div class="top-move-bar" style="width:${t.pct}%;background:${barColor}"></div></div>` +
      `<span class="top-move-pct">${t.pct}%</span>`;
    row.addEventListener('click', () => applyPreview(t, row));
    list.appendChild(row);
  }
  panel.classList.remove('hidden');
}

// ─── Chat progressiv rendern ──────────────────────────────────────────────────
// Chat-Nachrichten erscheinen synchron zum Replay: jede Nachricht hat einen
// Server-Zeitstempel `time`, und sobald ein nachgespielter Zug einen späteren
// `time` hat, werden alle dazwischen liegenden Nachrichten angefügt. So
// entsteht beim Vor-/Zurückblättern dieselbe zeitliche Reihenfolge wie live.
let gameChat = [];
let gameChatUidToColor = {};
let gameChatIdx = 0; // Index der nächsten noch nicht angezeigten Nachricht
let maxChatFullMoveNr = 0; // letzte Runde, in der überhaupt gechattet wurde

function buildChatMsgEl(msg) {
  const div = document.createElement('div');
  div.className = 'chat-msg';
  if (msg.type === 'info') {
    div.classList.add('info');
    div.textContent = msg.message;
  } else {
    const color = gameChatUidToColor[msg.playerId];
    if (color) div.classList.add(color.toLowerCase());
    div.classList.add('player');
    const userSpan = document.createElement('span');
    userSpan.className = 'chat-user';
    userSpan.textContent = (msg.username || '?') + ':';
    div.appendChild(userSpan);
    div.appendChild(document.createTextNode(' ' + msg.message));
  }
  return div;
}

// Zeigt alle Nachrichten mit time <= upToTime an (inkrementell).
function renderChatUpTo(upToTime) {
  const container = document.getElementById('chat-messages');
  let added = false;
  while (gameChatIdx < gameChat.length) {
    const msg = gameChat[gameChatIdx];
    if (msg.time != null && msg.time > upToTime) break;
    container.appendChild(buildChatMsgEl(msg));
    gameChatIdx++;
    added = true;
  }
  if (added) container.scrollTop = container.scrollHeight;
}

// Aus Chat-Nachrichten Forfeits/Resignations extrahieren — Eliminierungen, die
// nicht durch Königsschlag passieren und die die Engine sonst nicht erkennt.
// Königsschlag-Mate ("X checkmated!") ignorieren wir, das macht die Engine selbst.
function extractForfeits(chat) {
  const out = [];
  for (const m of chat) {
    if (typeof m?.message !== 'string' || m.fullMoveNr == null) continue;
    const match = m.message.match(/^(Red|Blue|Yellow|Green)\s+(forfeits|resigned)/i);
    if (match) {
      const c = match[1].toLowerCase();
      out.push({ fmn: m.fullMoveNr, color: c[0].toUpperCase() + c.slice(1) });
    }
  }
  out.sort((a, b) => a.fmn - b.fmn);
  return out;
}

function initChatView(chat, uidToColor) {
  maxChatFullMoveNr = Math.max(0, ...chat.map(m => m.fullMoveNr || 0));
  gameChat = chat.filter(m => m.type !== 'info' || m.fullMoveNr).sort((a, b) => (a.time ?? 0) - (b.time ?? 0));
  gameChatUidToColor = uidToColor;
  gameChatIdx = 0;
  document.getElementById('chat-messages').innerHTML = '';
}

function resetChatToTime(upToTime) {
  gameChatIdx = 0;
  document.getElementById('chat-messages').innerHTML = '';
  if (upToTime != null) renderChatUpTo(upToTime);
}

// ─── Spiel-JSON laden ─────────────────────────────────────────────────────────
function loadGame(gameData) {
  const pgn4  = gameData.pgn4  || '';
  const chat  = gameData.chat  || [];
  const colors = ['Red','Blue','Yellow','Green'];

  const uidToColor = {};
  const players    = [];
  for (let i = 1; i <= 4; i++) {
    const uid      = gameData['uid'      + i];
    const username = gameData['username' + i];
    if (uid)      uidToColor[uid] = colors[i - 1];
    if (username) players.push({ color: colors[i - 1].toLowerCase(), username });
  }

  const gameNrEl = document.getElementById('gameNr');
  if (gameNrEl) gameNrEl.textContent = gameData.gameNr ? '#' + gameData.gameNr : '';

  const gameInfo = document.getElementById('game-info');
  gameInfo.innerHTML =
    'Spiel #' + (gameData.gameNr || '?') +
    ' &nbsp;&middot;&nbsp; ' + (gameData.result || '') +
    '<div class="game-players">' +
    players.map(p => '<span class="player-badge ' + p.color + '">' + p.username + '</span>').join('') +
    '</div>';

  initChatView(chat, uidToColor);

  const scoreColors = ['red','blue','yellow','green'];
  for (let i = 1; i <= 4; i++) {
    const username = gameData['username' + i];
    const rating   = gameData['rating'   + i];
    const el = document.querySelector('.score.' + scoreColors[i-1] + ' .score-name');
    if (el && username) {
      el.textContent = rating ? `${username} (${rating})` : username;
    }
  }

  replayMoves = parsePgn4Moves(pgn4);
  replayIdx   = 0;
  allForfeits = extractForfeits(chat);
  appliedForfeitStack.length = 0;
  forfeitsAppliedAtIdx.length = 0;
  playerClocks = [null, null, null, null];
  prevPlayerClocks.length = 0;
  refreshAllScoreTimes();

  previewMv = null; previewRow = null;
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  selected = null; legalMoves = [];

  document.getElementById('replay-controls').classList.remove('hidden');
  updateReplayPos();
  draw();
}

document.getElementById('game-json-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (ev) => {
    try {
      const data = JSON.parse(ev.target.result);
      loadGame(data);
      // Analyse direkt aus der Datei (game_analysis/) übernehmen, falls vorhanden …
      if (Array.isArray(data.evals)) {
        replayEvals = data.evals;
        renderTopMoves();
      } else {
        // … sonst zusätzlich vom Server holen (z.B. wenn nur game_data/ geladen wurde)
        try {
          const r = await fetch('/game_analysis/' + file.name);
          if (r.ok) {
            const ana = await r.json();
            replayEvals = ana.evals || [];
            renderTopMoves();
          }
        } catch { /* kein Server oder keine Analyse – kein Problem */ }
      }
    } catch (err) { alert('Fehler beim Laden: ' + err.message); }
  };
  reader.readAsText(file);
  e.target.value = '';
});

document.getElementById('analysis-json-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const data = JSON.parse(ev.target.result);
      replayEvals = data.evals || [];
      renderTopMoves();
    } catch (err) { alert('Fehler beim Laden der Analyse: ' + err.message); }
  };
  reader.readAsText(file);
  e.target.value = '';
});

// ─── Replay-Navigation ────────────────────────────────────────────────────────
function replayApplyNext() {
  // Laufende Animation sofort beenden
  if (animReq !== null) { cancelAnimationFrame(animReq); animReq = null; draw(); }
  clearPreview();

  if (replayIdx >= replayMoves.length) return;
  const mv = replayMoves[replayIdx];

  // Vor diesem Zug noch ausstehende Forfeits/Resignations anwenden, damit die
  // Engine `to_move` korrekt weiterschiebt (sonst hängt sie auf einem
  // ausgestiegenen Spieler fest und alle Folgezüge laufen ins Leere).
  let forfeitsAppliedHere = 0;
  for (const f of allForfeits) {
    if (f.fmn <= (mv.fullMoveNr ?? Infinity) && !appliedForfeitStack.includes(f)) {
      if (engine.forfeit_color(f.color)) {
        appliedForfeitStack.push(f);
        forfeitsAppliedHere++;
      }
    }
  }
  forfeitsAppliedAtIdx[replayIdx] = forfeitsAppliedHere;

  // Synthetische '--' für ausgeschiedene Spieler einfügen, damit die
  // Rundengruppierung (4 Züge pro Zeile) auch nach Eliminierungen stimmt.
  // Beispiel: Blau ist eliminiert → Engine erwartet als nächstes Gelb. Damit
  // die Gelb-Notation in der Gelb-Spalte landet, schiebt die Schleife einen
  // '--'-Eintrag in der Blau-Spalte vor. `safety < 3` verhindert eine
  // Endlosschleife, falls der Engine-Zustand inkonsistent zur Zugliste wird.
  const stBefore = engine.get_state();
  const playerIdx = PLY_COLORS.indexOf(stBefore.to_move.toLowerCase());
  let safety = 0;
  while (plyCount % 4 !== playerIdx && safety++ < 3) {
    const skip = { text: '--', color: PLY_COLORS[plyCount % 4] };
    moveLogEntries.push(skip);
    if (plyCount % 4 === 0) {
      const li = document.createElement('li');
      li.innerHTML = roundHtml([skip]);
      moveList.appendChild(li);
    } else {
      moveList.lastElementChild.innerHTML =
        roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount + 1));
    }
    plyCount++;
  }

  let fromSq = -1, toSq = -1, movingPiece = null;
  if (mv.engine !== '--') {
    fromSq = engineSqIdx(mv.engine.slice(0, 2));
    toSq   = engineSqIdx(mv.engine.slice(2, 4));
    movingPiece = state?.squares[fromSq] ?? null;
    // Promotionsfallback (siehe applyPreview): chess.com-PGN4 markiert
    // Bauernumwandlungen nicht explizit, daher ggf. mit 'p' nachreichen.
    engine.apply_move(mv.engine) || engine.apply_move(mv.engine + 'p');
  }

  // Bedenkzeit-Anzeige des ziehenden Spielers aktualisieren (nur wenn der
  // PGN-Kommentar einen clock-Wert mitliefert). Vorherigen Wert für Undo merken.
  if (mv.clock != null && movingPiece) {
    const pIdx = COLOR_NAMES.indexOf(movingPiece.color.toLowerCase());
    if (pIdx >= 0) {
      prevPlayerClocks[replayIdx] = { pIdx, prev: playerClocks[pIdx] };
      playerClocks[pIdx] = mv.clock;
      updateScoreTime(pIdx);
    }
  }

  const score = replayEvals[replayIdx]?.score ?? null;
  const entry = { text: mv.display, color: PLY_COLORS[plyCount % 4], score };
  moveLogEntries.push(entry);
  if (plyCount % 4 === 0) {
    const li = document.createElement('li');
    li.innerHTML = roundHtml([entry]);
    moveList.appendChild(li);
  } else {
    const li = moveList.lastElementChild;
    li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount + 1));
  }
  plyCount++;
  moveList.scrollTop = moveList.scrollHeight;
  replayIdx++;
  updateReplayPos();
  selected = null; legalMoves = [];
  // Sobald die letzte Runde mit Chat erreicht ist, alle restlichen Nachrichten
  // freigeben — sonst würden Post-Game-Kommentare (deren `time` nach dem
  // letzten Zug liegt) bis zum Spielende stumm bleiben.
  const chatUpTo = (maxChatFullMoveNr > 0 && mv.fullMoveNr >= maxChatFullMoveNr) ? Infinity : mv.time;
  if (chatUpTo != null) renderChatUpTo(chatUpTo);

  // Zustand nach dem Zug holen und Figur animiert bewegen
  state = engine.get_state();
  animateMove(fromSq, toSq, movingPiece, () => draw());
  renderTopMoves();
}

function replayUndoPrev() {
  clearPreview();
  if (replayIdx <= 0) return;
  replayIdx--;
  if (replayMoves[replayIdx].engine !== '--') {
    engine.undo();
  }
  // Forfeits zurücknehmen, die unmittelbar vor diesem Zug angewandt wurden.
  // Reihenfolge im Engine-History-Stack: …, forfeit_n, …, forfeit_1, move →
  // erst der move (oben) wurde gerade gepoppt, jetzt rückwärts die Forfeits.
  const fc = forfeitsAppliedAtIdx[replayIdx] || 0;
  for (let i = 0; i < fc; i++) {
    engine.undo();
    appliedForfeitStack.pop();
  }
  forfeitsAppliedAtIdx[replayIdx] = 0;

  // Bedenkzeit zurückspulen
  const pc = prevPlayerClocks[replayIdx];
  if (pc) {
    playerClocks[pc.pIdx] = pc.prev;
    updateScoreTime(pc.pIdx);
    delete prevPlayerClocks[replayIdx];
  }

  // Letzten echten Zug entfernen
  plyCount--;
  moveLogEntries.pop();
  if (plyCount % 4 === 0) {
    if (moveList.lastChild) moveList.removeChild(moveList.lastChild);
  } else {
    const li = moveList.lastElementChild;
    li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount));
  }

  // Vorausgehende synthetische '--'-Einträge ebenfalls entfernen — sie
  // wurden in replayApplyNext() hinzugefügt und haben kein Gegenstück in
  // der Engine, dürfen also nicht "übrig bleiben".
  while (moveLogEntries.length > 0 && moveLogEntries[moveLogEntries.length - 1].text === '--') {
    plyCount--;
    moveLogEntries.pop();
    if (plyCount % 4 === 0) {
      if (moveList.lastChild) moveList.removeChild(moveList.lastChild);
    } else {
      const li = moveList.lastElementChild;
      li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount));
    }
  }

  // Chat auf den Zeitpunkt des letzten noch angezeigten Zugs zurücksetzen
  let lastTime = null;
  for (let i = replayIdx - 1; i >= 0; i--) {
    if (replayMoves[i].time != null) { lastTime = replayMoves[i].time; break; }
  }
  resetChatToTime(lastTime);

  updateReplayPos();
  selected = null; legalMoves = [];
  draw();
  renderTopMoves();
}

document.getElementById('btn-replay-start').addEventListener('click', () => {
  previewMv = null; previewRow = null;
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  replayIdx = 0;
  appliedForfeitStack.length = 0;
  forfeitsAppliedAtIdx.length = 0;
  playerClocks = [null, null, null, null];
  prevPlayerClocks.length = 0;
  refreshAllScoreTimes();
  selected = null; legalMoves = [];
  gameChatIdx = 0;
  document.getElementById('chat-messages').innerHTML = '';
  updateReplayPos();
  draw();
});

document.getElementById('btn-replay-prev').addEventListener('click', replayUndoPrev);
document.getElementById('btn-replay-next').addEventListener('click', replayApplyNext);

document.getElementById('btn-replay-end').addEventListener('click', () => {
  while (replayIdx < replayMoves.length) replayApplyNext();
});

// Pfeiltasten als Replay-Navigation – aber nur wenn der Fokus nicht in
// einem Eingabefeld liegt (sonst kollidiert es mit normalem Cursor-Movement
// im PGN-Textarea).
document.addEventListener('keydown', (e) => {
  if (replayMoves.length === 0) return;
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') { e.preventDefault(); replayApplyNext(); }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); replayUndoPrev();  }
});

// ─── Spiel-Filter (games_index.json) ─────────────────────────────────────────
let gamesIndex = []; // [{file, gameNr, date, result, players, ratings}, ...]

function renderGameFilterResults() {
  const input    = document.getElementById('game-filter');
  const colorSel = document.getElementById('game-filter-color');
  const placeSel = document.getElementById('game-filter-placement');
  const list     = document.getElementById('game-filter-results');
  const count    = document.getElementById('game-filter-count');
  if (!input || !list) return;
  const tokens = input.value.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  // "" → keine Auswahl. Empty values treated as "no filter" via NaN.
  const colorIdx  = colorSel && colorSel.value !== '' ? parseInt(colorSel.value, 10) : NaN;
  const placement = placeSel && placeSel.value !== '' ? parseInt(placeSel.value, 10) : NaN;

  // Drei UND-verknüpfte Filter:
  //   1. Jeder Suchtoken muss als Substring in irgendeinem Spielernamen vorkommen.
  //   2. Falls Farbe gesetzt UND mind. ein Token getippt → der erste Token muss
  //      genau zum Spieler in diesem Farb-Slot passen.
  //   3. Falls Platzierung gesetzt → der „erste Spieler" muss diesen Platz haben.
  //      Erster Spieler = Spieler im Farb-Slot (falls gesetzt) ODER erster
  //      Slot, der zum ersten Token passt.
  const matches = gamesIndex.filter(entry => {
    const names = entry.players.map(p => (p || '').toLowerCase());

    // 1. Namen-AND
    if (!tokens.every(tok => names.some(n => n.includes(tok)))) return false;

    // 2. Farbe verankert den ersten Token an einen bestimmten Slot.
    if (!isNaN(colorIdx) && tokens.length > 0) {
      if (!names[colorIdx].includes(tokens[0])) return false;
    }

    // 3. Platzierung des „ersten Spielers".
    if (!isNaN(placement)) {
      const slot = !isNaN(colorIdx)         ? colorIdx
                 : tokens.length > 0        ? names.findIndex(n => n.includes(tokens[0]))
                 : -1;
      if (slot < 0) return false; // ohne Anker keine sinnvolle Anwendung
      const placements = entry.placements || [0,0,0,0];
      if (placements[slot] !== placement) return false;
    }

    return true;
  });

  count.textContent = `${matches.length} / ${gamesIndex.length}`;
  list.innerHTML = '';
  for (const e of matches) {
    const li = document.createElement('li');
    li.className = 'game-row';
    const dateStr = e.date ? new Date(e.date).toISOString().slice(0, 10) : '';
    const placeIcon = ['', '🥇', '🥈', '🥉', '4'];
    const placements = e.placements || [0,0,0,0];
    const playersHtml = e.players.map((p, i) => {
      const c  = ['red','blue','yellow','green'][i];
      const rt = e.ratings[i] != null ? ` (${e.ratings[i]})` : '';
      const pl = placements[i] >= 1 && placements[i] <= 4 ? `${placeIcon[placements[i]]} ` : '';
      return `<span class="player-badge ${c}">${pl}${p}${rt}</span>`;
    }).join('');
    li.innerHTML =
      `<span class="gr-date">${dateStr}</span>` +
      `<span class="gr-players">${playersHtml}</span>` +
      `<span class="gr-nr">#${e.gameNr ?? '?'}</span>`;
    li.addEventListener('click', () => loadGameFromIndex(e.file));
    list.appendChild(li);
  }
}

async function loadGameFromIndex(filename) {
  try {
    const r = await fetch('/game_analysis/' + filename);
    if (!r.ok) { alert('Datei nicht erreichbar: ' + filename); return; }
    const data = await r.json();
    loadGame(data);
    if (Array.isArray(data.evals)) {
      replayEvals = data.evals;
      renderTopMoves();
    }
  } catch (err) {
    alert('Fehler beim Laden: ' + err.message);
  }
}

(async () => {
  try {
    const r = await fetch('games_index.json');
    if (!r.ok) return; // Index optional – ohne Server kein Filter
    gamesIndex = await r.json();
    renderGameFilterResults();
    document.getElementById('game-filter')
      .addEventListener('input', renderGameFilterResults);
    const colorSel = document.getElementById('game-filter-color');
    const placeSel = document.getElementById('game-filter-placement');
    if (colorSel) colorSel.addEventListener('change', renderGameFilterResults);
    if (placeSel) placeSel.addEventListener('change', renderGameFilterResults);
  } catch { /* still ohne Server lauffähig */ }
})();

// URL-Parameter ?game=<stem> → entsprechendes Analyse-File auto-laden.
// Akzeptiert mit oder ohne ".json"-Suffix.
(async () => {
  const param = new URLSearchParams(window.location.search).get('game');
  if (!param) return;
  const filename = param.endsWith('.json') ? param : param + '.json';
  await loadGameFromIndex(filename);
})();

// ─── Start ────────────────────────────────────────────────────────────────────
draw();
