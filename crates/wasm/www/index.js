// ─── WASM bootstrap ──────────────────────────────────────────────────────────
import init, { WasmEngine } from './pkg/chaturaji_wasm.js';
await init();
const engine = new WasmEngine();

// ─── Konstanten ───────────────────────────────────────────────────────────────
const BOARD_SIZE = 560;
const CELL = BOARD_SIZE / 8;

const COLOR_HEX = {
  Red: '#e74c3c', Blue: '#3498db', Yellow: '#f1c40f', Green: '#2ecc71',
};
const BOARD_LIGHT = '#f0d9b5';
const BOARD_DARK  = '#b58863';
const GLYPH = { King:'\u265A', Bishop:'\u265D', Knight:'\u265E', Boat:'\u26F5', Pawn:'\u265F' };

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
let selected   = null;
let legalMoves = [];
let state      = null;

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

function drawPieces() {
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let sq = 0; sq < 64; sq++) {
    const piece = state.squares[sq];
    if (!piece) continue;
    const colorIdx = ['Red','Blue','Yellow','Green'].indexOf(piece.color);
    const active = state.active[colorIdx];
    const [x, y] = sq2xy(sq);
    ctx.globalAlpha = active ? 1.0 : 0.3;

    const img = getPieceImage(piece.color, piece.kind);
    if (img) {
      const pad = CELL * 0.06;
      ctx.drawImage(img, x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
    } else {
      // Fallback: farbiger Kreis + Glyph
      ctx.fillStyle = COLOR_HEX[piece.color];
      ctx.beginPath();
      ctx.arc(x + CELL / 2, y + CELL / 2, CELL * 0.38, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#111';
      ctx.font = Math.round(CELL * 0.6) + 'px serif';
      ctx.fillText(GLYPH[piece.kind] || '?', x + CELL / 2, y + CELL / 2 + 1);
    }
    ctx.globalAlpha = 1;
  }
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
    overlay.textContent = '\uD83C\uDFC6 ' + w + ' gewinnt!';
    overlay.classList.remove('hidden');
    turnEl.innerHTML = 'Spiel beendet &ndash; Sieger: <strong>' + w + '</strong>';
  } else {
    overlay.classList.add('hidden');
    const active = ['Red','Blue','Yellow','Green'].filter((_, i) => state.active[i]);
    turnEl.innerHTML = 'Zug: <strong style="color:' + COLOR_HEX[state.to_move] + '">'
      + state.to_move + '</strong> &nbsp;|&nbsp; Aktiv: ' + active.join(', ');
  }
}

// ─── Netz-Bewertungsbalken ────────────────────────────────────────────────────
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
canvas.addEventListener('click', (e) => {
  if (state && state.is_over) return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (BOARD_SIZE / rect.width);
  const y = (e.clientY - rect.top)  * (BOARD_SIZE / rect.height);
  const sq = xy2sq(x, y);
  if (sq === null) return;

  // Zug ausführen wenn Zielfeld einer legalen Bewegung entspricht
  if (selected !== null) {
    const mv = legalMoves.find(m => m.to === sq);
    if (mv && engine.apply_move(mv.notation)) {
      appendMoveLog(mv);
      selected = null; legalMoves = [];
      draw(); return;
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

function appendMoveLog(mv) {
  const li = document.createElement('li');
  li.textContent = mv.notation
    + (mv.captures ? ' \u00D7' : '')
    + (mv.promoted ? ' =B' : '');
  moveList.appendChild(li);
  moveList.scrollTop = moveList.scrollHeight;
}

// ─── Buttons ──────────────────────────────────────────────────────────────────
document.getElementById('btn-engine-move').addEventListener('click', () => {
  if (state && state.is_over) return;
  const depth = parseInt(depthInput.value);
  engineInfo.textContent = 'Suche bei Tiefe ' + depth + '\u2026';
  setTimeout(() => {
    const t0  = performance.now();
    const res = engine.best_move(depth);
    const ms  = (performance.now() - t0).toFixed(0);
    if (res.best_move) {
      engine.apply_move(res.best_move);
      appendMoveLog({ notation: res.best_move, captures: false, promoted: false });
    }
    const netNote = res.used_network ? ' \uD83E\uDDE0 Netz aktiv' : '';
    engineInfo.textContent = 'Tiefe ' + res.depth
      + ' \u00B7 ' + (res.nodes || 0).toLocaleString() + ' Knoten'
      + ' \u00B7 ' + ms + 'ms' + netNote;
    selected = null; legalMoves = [];
    draw();
  }, 10);
});

document.getElementById('btn-undo').addEventListener('click', () => {
  if (engine.undo()) {
    if (moveList.lastChild) moveList.removeChild(moveList.lastChild);
    selected = null; legalMoves = [];
    draw();
  }
});

document.getElementById('btn-reset').addEventListener('click', () => {
  engine.reset();
  moveList.innerHTML = '';
  selected = null; legalMoves = [];
  engineInfo.textContent = '';
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
  const err = engine.load_pgn(pgn);
  if (err) { alert('PGN-Fehler: ' + err); return; }
  moveList.innerHTML = '';
  selected = null; legalMoves = [];
  draw();
});

document.getElementById('btn-export-pgn').addEventListener('click', () => {
  document.getElementById('pgn-output').textContent = engine.export_pgn();
});

// ─── Start ────────────────────────────────────────────────────────────────────
draw();
