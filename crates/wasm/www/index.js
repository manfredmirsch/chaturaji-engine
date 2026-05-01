// ─── WASM bootstrap ──────────────────────────────────────────────────────────
import init, { WasmEngine } from './pkg/chaturaji_wasm.js';
await init();
const engine = new WasmEngine();

// ─── Konstanten ───────────────────────────────────────────────────────────────
const BOARD_SIZE = 560;
const CELL = BOARD_SIZE / 8;
const MOVE_ANIM_MS = 180; // Animationsdauer in ms (0 = deaktiviert)

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

function drawOnePiece(piece, x, y, isActive) {
  ctx.globalAlpha = isActive ? 1.0 : 0.3;
  const img = getPieceImage(piece.color, piece.kind);
  if (img) {
    const pad = CELL * 0.06;
    ctx.drawImage(img, x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
  } else {
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
  if (previewMv !== null) { clearPreview(); draw(); return; }
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

function appendMoveLog(mv) {
  const text = mv.notation
    + (mv.captures ? ' \u00D7' : '')
    + (mv.promoted ? ' =B' : '');
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
  document.getElementById('replay-controls').classList.add('hidden');
  document.getElementById('top-moves-panel').classList.add('hidden');
  gameChat = []; gameChatUidToColor = {}; gameChatIdx = 0; maxChatFullMoveNr = 0;
  document.getElementById('chat-messages').innerHTML = '<p class="chat-empty">Kein Spiel geladen.</p>';
  document.getElementById('game-info').innerHTML = '';
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

// chess.com 14×14 koordinate → Engine a1-h8 (offset = 3)
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
    const dateMatch = comment.match(/date=(\S+)/);
    const time = dateMatch ? new Date(dateMatch[1]).getTime() : null;
    if (cc === '--') {
      moves.push({ display: '--', engine: '--', time, fullMoveNr: currentFullMoveNr });
    } else {
      const eng = ccMoveToEngine(cc);
      const dsp = ccMoveToDisplay(cc);
      if (eng && dsp) moves.push({ display: dsp, engine: eng, time, fullMoveNr: currentFullMoveNr });
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

  list.innerHTML = '';
  previewMv = null; previewRow = null;
  for (const t of tops) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'top-move-row';
    row.innerHTML =
      `<span class="top-move-notation ${colorClass}">${engineMoveToDisplay(t.mv)}</span>` +
      `<div class="top-move-bar-wrap"><div class="top-move-bar" style="width:${t.pct}%"></div></div>` +
      `<span class="top-move-pct">${t.pct}%</span>`;
    row.addEventListener('click', () => applyPreview(t, row));
    list.appendChild(row);
  }
  panel.classList.remove('hidden');
}

// ─── Chat progressiv rendern ──────────────────────────────────────────────────
let gameChat = [];
let gameChatUidToColor = {};
let gameChatIdx = 0; // Index der nächsten noch nicht angezeigten Nachricht
let maxChatFullMoveNr = 0;

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
    const el = document.querySelector('.score.' + scoreColors[i-1] + ' .score-name');
    if (el && username) el.textContent = username;
  }

  replayMoves = parsePgn4Moves(pgn4);
  replayIdx   = 0;

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

  // Synthetische '--' für ausgeschiedene Spieler einfügen, damit die
  // Rundengruppierung (4 Züge pro Zeile) auch nach Eliminierungen stimmt.
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
    engine.apply_move(mv.engine) || engine.apply_move(mv.engine + 'p');
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

  // Letzten echten Zug entfernen
  plyCount--;
  moveLogEntries.pop();
  if (plyCount % 4 === 0) {
    if (moveList.lastChild) moveList.removeChild(moveList.lastChild);
  } else {
    const li = moveList.lastElementChild;
    li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount));
  }

  // Vorausgehende synthetische '--'-Einträge ebenfalls entfernen
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

document.addEventListener('keydown', (e) => {
  if (replayMoves.length === 0) return;
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') { e.preventDefault(); replayApplyNext(); }
  if (e.key === 'ArrowLeft')  { e.preventDefault(); replayUndoPrev();  }
});

// ─── Start ────────────────────────────────────────────────────────────────────
draw();
