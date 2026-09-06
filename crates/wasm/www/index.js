// ─── WASM bootstrap ──────────────────────────────────────────────────────────
// Die gesamte Spiel-Logik (Zuggenerierung, Suche, Bewertung) liegt in Rust und
// wird per wasm-bindgen exportiert. `engine` hält den einzigen globalen Spiel-
// zustand; alles UI-seitige (Highlights, Animation, Chat) lebt in dieser Datei.
import init, { WasmEngine } from './pkg/chaturaji_wasm.js?v=2605111501';
import { initI18n, setLang, t, getLang, onLanguageChange, applyTranslations, SUPPORTED_LANGS } from './i18n.js';

await init({ module_or_path: './pkg/chaturaji_wasm_bg.wasm?v=2605111501' });
const engine = new WasmEngine();

// Sprache (Default: en) frühestmöglich anwenden — vor dem ersten draw().
await initI18n('en');

// ─── Konstanten ───────────────────────────────────────────────────────────────
const BOARD_SIZE = 660;
const MARGIN_COORDS = 8;

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

// ─── Aufzeichnen (Recording) ──────────────────────────────────────────────────
// `recording` = läuft gerade eine Aufnahme? `recordedMoves` = die neu gespielten
// Halbzüge (Klick oder Engine) als {engine, time}. `recordPrefix` = bei Modus 2
// die übernommenen Züge des geladenen Spiels bis zum gewählten Zug. `lastLoadedGame`
// = zuletzt via loadGame() geöffnete Rohdaten (für Metadaten-Vorbelegung).
// `lastLoadedFile` = Dateiname des zuletzt aus dem Index geladenen Spiels; dient
// als Basis für den Dateinamen einer Aufzeichnung (alter Name + fortlaufende Nr.).
// Früh deklariert, weil updateRecordControls() bereits beim Modul-Load läuft.
let recording      = false;
let recordedMoves  = [];
let recordPrefix   = [];
let lastLoadedGame = null;
let lastLoadedFile = null;

// Sperrt während laufender Engine-Suche/Animation alle weiteren Aktionen,
// damit Notation und Engine-Zustand nicht durcheinanderkommen.
let isBusy = false;
// Pro Aktion vergebene Epoche – setTimeout-Bodies prüfen, ob sie noch
// "ihre" Epoche sehen, sonst brechen sie ab (z.B. nach Reset).
let actionEpoch = 0;

function setBusy(b) {
  isBusy = b;
  const ids = ['btn-engine-move', 'btn-undo', 'btn-reset', 'btn-play-prev', 'btn-play-next'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.disabled = b;
  }
  // Nach Freigabe Prev/Next-Status aus dem Redo-Stack rekonstruieren.
  if (!b && typeof updatePlayReplayButtons === 'function') updatePlayReplayButtons();
}

// Engine-Suchergebnis-Visualisierung: Top-N-Züge die in der aktuellen Stellung
// untersucht wurden, werden als farbige Pfeile auf dem Brett gezeichnet bis der
// nächste Zug fällt. Format: [{mv, pct, rank}] sortiert nach Stärke.
let engineCandidates = null;

// Farben/Linienstärke pro Rang (0 = Bestmove). pct=Anteil von 1.0 für Alpha.
const CANDIDATE_STYLE = [
  { stroke: '#d22', width: 8, alpha: 0.85 }, // 1. Rot, dick
  { stroke: '#e80', width: 6, alpha: 0.65 }, // 2. Orange
  { stroke: '#dc3', width: 5, alpha: 0.55 }, // 3. Gelb
];

// ─── Zeichnen ─────────────────────────────────────────────────────────────────
function draw() {
  state = engine.get_state();
  ctx.clearRect(0, 0, BOARD_SIZE, BOARD_SIZE);
  drawSquares();
  drawHighlights();
  drawPieces();
  drawEngineCandidates();
  updateScores();
  updateStatus();
  updateNetEval();
}

// Pfeil von Feld `fromSq` nach `toSq` mit Stilangabe aus CANDIDATE_STYLE.
function drawArrow(fromSq, toSq, style) {
  const [fx, fy] = sq2xy(fromSq);
  const [tx, ty] = sq2xy(toSq);
  const x1 = fx + CELL / 2, y1 = fy + CELL / 2;
  const x2 = tx + CELL / 2, y2 = ty + CELL / 2;
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len;
  // Schaft endet kurz vor dem Zielfeld, Pfeilkopf füllt den Rest
  const headLen = Math.min(CELL * 0.45, len * 0.45);
  const shaftEndX = x2 - ux * headLen * 0.6;
  const shaftEndY = y2 - uy * headLen * 0.6;

  ctx.save();
  ctx.globalAlpha = style.alpha;
  ctx.strokeStyle = style.stroke;
  ctx.fillStyle   = style.stroke;
  ctx.lineWidth   = style.width;
  ctx.lineCap     = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(shaftEndX, shaftEndY);
  ctx.stroke();
  // Pfeilspitze
  const px = -uy, py = ux;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - ux * headLen + px * headLen * 0.45,
             y2 - uy * headLen + py * headLen * 0.45);
  ctx.lineTo(x2 - ux * headLen - px * headLen * 0.45,
             y2 - uy * headLen - py * headLen * 0.45);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawEngineCandidates() {
  if (!engineCandidates || engineCandidates.length === 0) return;
  // Schwächste zuerst, damit der Bestmove obenauf liegt
  for (let i = engineCandidates.length - 1; i >= 0; i--) {
    const c = engineCandidates[i];
    const fromSq = engineSqIdx(c.mv.slice(0, 2));
    const toSq   = engineSqIdx(c.mv.slice(2, 4));
    drawArrow(fromSq, toSq, CANDIDATE_STYLE[i] ?? CANDIDATE_STYLE[2]);
  }
}

function clearEngineCandidates() {
  if (engineCandidates) {
    engineCandidates = null;
    const panel = document.getElementById('engine-candidates-panel');
    if (panel) panel.classList.add('hidden');
  }
}

// Engine indiziert Felder als rank*8 + file mit a1=0, h8=63. Canvas-Y wächst
// nach unten. Per Doppelklick auf einen Score-Namen kann der Spieler nach
// "unten links" gedreht werden — `viewRot` hält 0..3 viertel Drehungen im
// Uhrzeigersinn (0 = Rot unten, Default).
let viewRot = (() => {
  const v = parseInt(localStorage.getItem('viewRot') || '0', 10);
  return (v >= 0 && v <= 3) ? v : 0;
})();

// engine (file, rank) -> canvas (cf, cr) je nach Rotation
function frToCanvas(f, r) {
  switch (viewRot) {
    case 0: return [f,     7 - r];
    case 1: return [7 - r, 7 - f];
    case 2: return [7 - f, r];
    case 3: return [r,     f];
  }
}

function canvasToFR(cf, cr) {
  switch (viewRot) {
    case 0: return [cf,     7 - cr];
    case 1: return [7 - cr, 7 - cf];
    case 2: return [7 - cf, cr];
    case 3: return [cr,     cf];
  }
}

function sq2xy(sq) {
  const [cf, cr] = frToCanvas(sq % 8, Math.floor(sq / 8));
  return [cf * CELL, cr * CELL];
}

function xy2sq(x, y) {
  const cf = Math.floor(x / CELL);
  const cr = Math.floor(y / CELL);
  if (cf < 0 || cf >= 8 || cr < 0 || cr >= 8) return null;
  const [f, r] = canvasToFR(cf, cr);
  return r * 8 + f;
}

function drawSquares() {
  // Felder
  for (let cr = 0; cr < 8; cr++) {
    for (let cf = 0; cf < 8; cf++) {
      const [f, r] = canvasToFR(cf, cr);
      ctx.fillStyle = (f + r) % 2 === 0 ? BOARD_LIGHT : BOARD_DARK;
      ctx.fillRect(cf * CELL, cr * CELL, CELL, CELL);
    }
  }
  ctx.font = '12px sans-serif';

  const horizLabel = (cf, cr) => {
    const [f, r] = canvasToFR(cf, cr);
    return (viewRot === 0 || viewRot === 2) ? String.fromCharCode(97 + f) : String(r + 1);
  };
  const vertLabel = (cf, cr) => {
    const [f, r] = canvasToFR(cf, cr);
    return (viewRot === 0 || viewRot === 2) ? String(r + 1) : String.fromCharCode(97 + f);
  };
  const labelColor = (cf, cr) => {
    const [f, r] = canvasToFR(cf, cr);
    return (f + r) % 2 === 0 ? BOARD_DARK : BOARD_LIGHT;
  };

  ctx.textAlign = 'right';
  // Koordinaten oben
  for (let cf = 0; cf < 8; cf++) {
    ctx.fillStyle = labelColor(cf, 0);
    ctx.fillText(horizLabel(cf, 0), cf * CELL + CELL - MARGIN_COORDS, MARGIN_COORDS);
  }
  // Koordinaten unten
  for (let cf = 0; cf < 8; cf++) {
    ctx.fillStyle = labelColor(cf, 7);
    ctx.fillText(horizLabel(cf, 7), cf * CELL + CELL - MARGIN_COORDS, BOARD_SIZE - MARGIN_COORDS);
  }
  ctx.textAlign = 'left';
  // Koordinaten links
  for (let cr = 0; cr < 8; cr++) {
    ctx.fillStyle = labelColor(0, cr);
    ctx.fillText(vertLabel(0, cr), MARGIN_COORDS, cr * CELL + MARGIN_COORDS);
  }
  // Koordinaten rechts
  for (let cr = 0; cr < 8; cr++) {
    ctx.fillStyle = labelColor(7, cr);
    ctx.fillText(vertLabel(7, cr), BOARD_SIZE - MARGIN_COORDS * 2, cr * CELL + MARGIN_COORDS);
  }
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
    document.getElementById('score-' + c).textContent = state.scores[i] + (forfeitBonus[i] || 0);
  });
}

function updateStatus() {
  if (state.is_over) {
    const w = state.winner ? t('color.' + state.winner) : t('status.nobody');
    overlay.classList.add('hidden');
    turnEl.innerHTML = t('status.gameOver', { winner: w });
  } else {
    overlay.classList.add('hidden');
    const active = ['Red','Blue','Yellow','Green']
      .filter((_, i) => state.active[i])
      .map(c => t('color.' + c));
    turnEl.innerHTML = t('status.turn', {
      color:  COLOR_HEX[state.to_move],
      name:   t('color.' + state.to_move),
      active: active.join(', '),
    });
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
  if (isBusy) return;
  if (state && state.is_over) return;
  if (previewMv !== null) { clearPreview(); draw(); return; }
  // Skalierung Canvas-Pixel ↔ CSS-Pixel: das Brett kann per CSS verkleinert
  // sein, intern wird aber immer mit BOARD_SIZE gerechnet.
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (BOARD_SIZE / rect.width);
  const y = (e.clientY - rect.top)  * (BOARD_SIZE / rect.height);
  const sq = xy2sq(x, y);
  if (sq === null) return;

  // Sobald der Spieler aktiv mit dem Brett interagiert, sind die alten
  // Engine-Pfeile veraltet → wegräumen, bevor irgendwas neu gezeichnet wird.
  clearEngineCandidates();

  // Zug ausführen wenn Zielfeld einer legalen Bewegung entspricht
  if (selected !== null) {
    const mv = legalMoves.find(m => m.to === sq);
    if (mv) {
      const fromSq = mv.from;
      const movingPiece = state.squares[fromSq];
      recordCaptureFromMove(mv.to);
      if (engine.apply_move(mv.notation)) {
        clearEngineCandidates();
        appendMoveLog(mv);
        commitToReplay(mv.notation);
        state = engine.get_state();
        renderCaptures();
        renderTopMoves();
        selected = null; legalMoves = [];
        setBusy(true);
        animateMove(fromSq, mv.to, movingPiece, () => { draw(); setBusy(false); });
        return;
      }
      // apply_move fehlgeschlagen → letzte recordCapture rückgängig
      undoLastCapture();
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

// Hängt einen Halbzug an die Zugliste an. Vier Halbzüge bilden eine "Runde"
// (Rot, Blau, Gelb, Grün) und werden in einem einzigen <li> zusammengefasst.
// Für den ersten Zug einer Runde wird ein neues <li> angelegt, danach wird
// der gesamte Zeileninhalt re-gerendert (einfacher als einzelne Zellen zu
// patchen, kostet kaum etwas bei nur 4 Spalten).
function pushLoggedEntry(entry) {
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

function appendMoveLog(mv) {
  const text = mv.notation
    + (mv.captures ? ' \u00D7' : '')
    + (mv.promoted ? ' =B' : ''); // chaturaji: Bauer wird beim Promotionsfeld zum Boat (=B)
  const entry = { text, color: PLY_COLORS[plyCount % 4], notation: mv.notation };
  pushLoggedEntry(entry);
  // Läuft eine Aufnahme, wird jeder gespielte Halbzug (Klick oder Engine)
  // mitgeschnitten. mv.notation ist die Engine-Notation (z.B. "c2c3").
  if (recording) { recordedMoves.push({ engine: mv.notation, time: Date.now() }); updateRecordControls(); }
  // Neuer Zug entwertet die Vor-Historie für den Redo-Button.
  playRedoStack.length = 0;
  updatePlayReplayButtons();
}

// ─── Buttons ──────────────────────────────────────────────────────────────────
document.getElementById('btn-engine-move').addEventListener('click', () => {
  if (isBusy) return;
  if (state && state.is_over) return;
  clearPreview();
  clearEngineCandidates();
  const depth = parseInt(depthInput.value);
  engineInfo.textContent = t('engine.searching', { depth });
  const epoch = ++actionEpoch;
  setBusy(true);
  // setTimeout(…, 10) gibt dem Browser einen Tick, um den "Suche…"-Text
  // anzuzeigen, bevor die synchrone (und potenziell sekundenlange) WASM-
  // Suche den Main-Thread blockiert.
  setTimeout(() => {
    if (epoch !== actionEpoch) return; // Reset während Wartezeit -> verworfen
    const useBook = document.getElementById('use-opening-book')?.checked ?? false;
    const t0  = performance.now();
    const res = engine.best_move(depth);
    const ms  = (performance.now() - t0).toFixed(0);

    // Buch-Hit: Engine hat ohne Suche geantwortet (depth=0, nodes=0).
    const isBookMove = res.depth === 0 && (res.nodes || 0) === 0 && res.best_move;

    // Kandidatenpfeile: bei Buch-Treffer + aktivierter Checkbox die Buchzüge
    // verwenden, damit roter Pfeil = gesetzter Zug; sonst Engine-Top-N aus TT.
    let tops;
    if (useBook && isBookMove) {
      tops = Array.from(engine.book_top_moves(3)).map(b => ({ mv: b.mv, pct: b.pct }));
    } else {
      // top_moves nutzt die noch warme Transposition-Tabelle, kostet kaum extra.
      tops = Array.from(engine.top_moves(depth, 3)).map(t => ({ mv: t.mv, pct: t.pct }));
    }

    // Top-3-Pfeile auf das Brett legen, bis der nächste Zug fällt.
    engineCandidates = tops;
    renderEngineCandidates(tops);

    let fromSq = -1, toSq = -1, movingPiece = null;
    if (res.best_move) {
      fromSq = engineSqIdx(res.best_move.slice(0, 2));
      toSq   = engineSqIdx(res.best_move.slice(2, 4));
      movingPiece = state?.squares[fromSq] ?? null;
    }

    const netNote  = res.used_network ? t('net.active') : '';
    const topsLine = tops.length > 0
      ? '  ' + tops.map((tm, i) =>
          (i + 1) + '. ' + engineMoveToDisplay(tm.mv) + ' ' + tm.pct + '%'
        ).join(' \u00B7 ')
      : '';
    if (isBookMove) {
      engineInfo.textContent = t('engine.bookMove', { ms, net: netNote, tops: topsLine });
    } else {
      engineInfo.textContent = t('engine.depthInfo', {
        depth: res.depth,
        nodes: (res.nodes || 0).toLocaleString(),
        ms,
        net: netNote,
        tops: topsLine,
      });
    }

    // Erst Pfeile zeichnen, dann mit kleiner Verzögerung den Zug ausführen
    // \u2014 das Auge bekommt eine Chance, die Alternativen zu sehen.
    draw();
    setTimeout(() => {
      if (epoch !== actionEpoch) return; // Reset während 700ms-Pfeilen -> verworfen
      let applied = false;
      if (res.best_move) {
        recordCaptureFromMove(toSq);
        if (engine.apply_move(res.best_move)) {
          appendMoveLog({ notation: res.best_move, captures: false, promoted: false });
          commitToReplay(res.best_move);
          state = engine.get_state();
          renderCaptures();
          renderTopMoves();
          applied = true;
        } else {
          undoLastCapture();
        }
      }
      selected = null; legalMoves = [];
      if (applied) {
        animateMove(fromSq, toSq, movingPiece, () => { draw(); setBusy(false); });
      } else {
        draw();
        setBusy(false);
      }
    }, 700);
  }, 10);
});

function renderEngineCandidates(tops) {
  const panel = document.getElementById('engine-candidates-panel');
  const list  = document.getElementById('engine-candidates-list');
  if (!panel || !list) return;
  if (!tops || tops.length === 0) { panel.classList.add('hidden'); return; }
  list.innerHTML = '';
  for (let i = 0; i < tops.length; i++) {
    const t = tops[i];
    const style = CANDIDATE_STYLE[i] ?? CANDIDATE_STYLE[2];
    const row = document.createElement('div');
    row.className = 'engine-cand-row';
    row.innerHTML =
      '<span class="engine-cand-marker" style="background:' + style.stroke + '"></span>'
      + '<span class="engine-cand-rank">' + (i + 1) + '.</span>'
      + '<span class="engine-cand-mv">' + engineMoveToDisplay(t.mv) + '</span>'
      + '<div class="engine-cand-bar-wrap"><div class="engine-cand-bar" '
      + 'style="width:' + t.pct + '%;background:' + style.stroke + '"></div></div>'
      + '<span class="engine-cand-pct">' + t.pct + '%</span>';
    list.appendChild(row);
  }
  panel.classList.remove('hidden');
}

function commitToReplay(engineNotation) {
  if (replayMoves.length === 0 || replayIdx >= replayMoves.length) return;
  replayMoves[replayIdx] = {
    ...replayMoves[replayIdx],
    engine:  engineNotation,
    display: engineMoveToDisplay(engineNotation),
  };
  forfeitsAppliedAtIdx[replayIdx] = 0;
  replayIdx++;
  updateReplayPos();
}

function playUndoStep() {
  if (isBusy) return false;
  if (previewMv !== null) { clearPreview(); draw(); return true; }
  clearEngineCandidates();
  if (plyCount === 0) return false;
  const lastEntry = moveLogEntries[moveLogEntries.length - 1];

  // Daten für die Rück-Animation aus letztem Zug holen, BEVOR engine.undo()
  // den Zustand verändert (state.squares[to] wird sonst zur leeren Vorgänger-
  // figur).
  let animFrom = -1, animTo = -1, animPiece = null;
  if (lastEntry && typeof lastEntry.notation === 'string' && lastEntry.notation.length >= 4) {
    animFrom = engineSqIdx(lastEntry.notation.slice(0, 2));
    animTo   = engineSqIdx(lastEntry.notation.slice(2, 4));
    animPiece = state?.squares[animTo] ?? null;
  }

  if (engine.undo()) {
    plyCount--;
    moveLogEntries.pop();
    // Aufnahme: den zuletzt aufgezeichneten Halbzug ebenfalls verwerfen (nur
    // eigene Aufnahme-Züge, nicht den fixen Modus-2-Prefix).
    if (recording && recordedMoves.length > 0) { recordedMoves.pop(); updateRecordControls(); }
    if (plyCount % 4 === 0) {
      if (moveList.lastChild) moveList.removeChild(moveList.lastChild);
    } else {
      const li = moveList.lastElementChild;
      li.innerHTML = roundHtml(moveLogEntries.slice(Math.floor(plyCount / 4) * 4, plyCount));
    }
    selected = null; legalMoves = [];
    undoLastCapture();
    renderCaptures();
    if (lastEntry && lastEntry.notation) playRedoStack.push(lastEntry);
    updatePlayReplayButtons();
    state = engine.get_state();
    if (animPiece && animFrom !== animTo) {
      setBusy(true);
      animateMove(animTo, animFrom, animPiece, () => { draw(); setBusy(false); });
    } else {
      draw();
    }
    return true;
  }
  return false;
}

function playRedoStep() {
  if (isBusy) return;
  if (playRedoStack.length === 0) return;
  clearPreview();
  clearEngineCandidates();
  const entry = playRedoStack[playRedoStack.length - 1];
  const fromSq = engineSqIdx(entry.notation.slice(0, 2));
  const toSq   = engineSqIdx(entry.notation.slice(2, 4));
  const movingPiece = state?.squares[fromSq] ?? null;
  recordCaptureFromMove(toSq);
  if (engine.apply_move(entry.notation)) {
    playRedoStack.pop();
    // Originaltext (inkl. ×, =B) sowie Farbe für aktuelle ply-Position übernehmen.
    pushLoggedEntry({ text: entry.text, color: PLY_COLORS[plyCount % 4], notation: entry.notation });
    state = engine.get_state();
    renderCaptures();
    selected = null; legalMoves = [];
    setBusy(true);
    animateMove(fromSq, toSq, movingPiece, () => { draw(); setBusy(false); });
  } else {
    undoLastCapture();
  }
}

document.getElementById('btn-undo').addEventListener('click', playUndoStep);
document.getElementById('btn-play-prev').addEventListener('click', playUndoStep);
document.getElementById('btn-play-next').addEventListener('click', playRedoStep);

document.getElementById('btn-reset').addEventListener('click', () => {
  // Reset darf jederzeit greifen — laufende Engine-Suche/Animation entwerten.
  actionEpoch++;
  if (animReq !== null) { cancelAnimationFrame(animReq); animReq = null; }
  previewMv = null; previewRow = null;
  clearEngineCandidates();
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  selected = null; legalMoves = [];
  engineInfo.textContent = '';
  replayMoves = []; replayMovesOriginal = []; replayIdx = 0; replayEvals = []; replayAnnotations = [];
  allForfeits = []; appliedForfeitStack.length = 0; forfeitsAppliedAtIdx.length = 0;
  forfeitBonus = [0, 0, 0, 0]; lateForfeitCount = 0;
  resignedKingBonusAtIdx.length = 0;
  playerClocks = [null, null, null, null];
  prevPlayerClocks.length = 0;
  refreshAllScoreTimes();
  document.getElementById('replay-controls').classList.add('hidden');
  document.getElementById('play-replay-controls').classList.remove('hidden');
  playRedoStack.length = 0;
  updatePlayReplayButtons();
  clearCaptures(); renderCaptures();
  document.getElementById('top-moves-panel').classList.add('hidden');
  gameChat = []; gameChatUidToColor = {}; gameChatIdx = 0; maxChatFullMoveNr = 0;
  document.getElementById('chat-messages').innerHTML =
    '<p class="chat-empty" data-i18n="chat.empty">' + t('chat.empty') + '</p>';
  document.getElementById('game-info').innerHTML = '';
  const gameNrEl = document.getElementById('gameNr');
  if (gameNrEl) gameNrEl.textContent = '';
  // Score-Namen wieder auf Default-Farbtitel zurücksetzen und i18n-Marker
  // restaurieren, damit ein Sprachwechsel danach wieder greift.
  const colorKeys = { red: 'color.Red', blue: 'color.Blue', yellow: 'color.Yellow', green: 'color.Green' };
  for (const c of Object.keys(colorKeys)) {
    const el = document.querySelector('.score.' + c + ' .score-name');
    if (el) {
      el.setAttribute('data-i18n', colorKeys[c]);
      el.textContent = t(colorKeys[c]);
    }
  }
  // Kein Spiel mehr geladen → Buch-Checkbox aktivieren und Top-Züge neu rendern
  const bookCb = document.getElementById('use-opening-book');
  if (bookCb && engine.book_info().loaded) { bookCb.checked = true; renderTopMoves(); }
  // Laufende Aufnahme abbrechen und Aufnahme-State zurücksetzen.
  recording = false; recordedMoves = []; recordPrefix = []; lastLoadedGame = null; lastLoadedFile = null;
  updateRecordControls();
  setBusy(false);
  draw();
});

// ─── Aufzeichnen (Recording) ──────────────────────────────────────────────────
// Modus 1: frische Partie ab Zug 1 aufzeichnen. Nutzt den Reset-Handler (leert
// Brett + schaltet auf Play-Modus) und startet dann die Aufnahme.
function startRecordingFresh() {
  document.getElementById('btn-reset').click(); // leert alles + recording=false
  recording = true; recordedMoves = []; recordPrefix = [];
  updateRecordControls();
}

// Modus 2: geladenes Spiel bis zum aktuellen Replay-Zug übernehmen und ab hier
// weiter aufzeichnen (eigene Klick- oder Engine-Züge). Das Brett steht bereits
// auf der Stellung nach `replayIdx` Zügen.
function startRecordingBranch() {
  if (replayMoves.length === 0) return;
  recordPrefix = replayMoves.slice(0, replayIdx).map(m => ({ engine: m.engine, time: m.time }));
  // replayMoves auf den gewählten Zug kürzen → commitToReplay wird zum No-Op,
  // Folgezüge landen nur noch in recordedMoves.
  replayMoves = replayMoves.slice(0, replayIdx);
  replayIdx = replayMoves.length;
  updateReplayPos();
  // Auf Play-Modus umschalten (Replay-Navigation aus, Play-Navigation an).
  document.getElementById('replay-controls').classList.add('hidden');
  document.getElementById('play-replay-controls').classList.remove('hidden');
  recording = true; recordedMoves = [];
  updatePlayReplayButtons();
  updateRecordControls();
}

function stopRecording() {
  recording = false;
  updateRecordControls();
}

// Enable/Disable + Anzeige der Aufnahme-Buttons je nach Zustand.
function updateRecordControls() {
  const $ = id => document.getElementById(id);
  const btnNew = $('btn-record-new');
  const btnBranch = $('btn-record-branch');
  const btnStop = $('btn-record-stop');
  const btnSave = $('btn-record-save');
  const ind = $('record-indicator');
  if (btnNew) btnNew.disabled = false;
  if (btnBranch) btnBranch.disabled = recording || replayMoves.length === 0;
  if (btnStop) btnStop.disabled = !recording;
  if (btnSave) btnSave.disabled = recordedMoves.length === 0;
  if (ind) {
    const total = recordPrefix.length + recordedMoves.length;
    ind.textContent = recording ? t('record.running', { n: total }) : '';
    ind.classList.toggle('hidden', !recording);
  }
}

// Ermittelt den nächsten freien Dateinamen für die Aufzeichnung eines bestehenden
// Spiels: alter Name (ohne evtl. vorhandene "-<Nr>"-Endung) plus fortlaufende Nummer.
// `index` wird durchsucht, um bereits gespeicherte Varianten zu überspringen.
function nextVariantFilename(baseFile, index) {
  const root = baseFile.replace(/\.json$/i, '').replace(/-\d+$/, '');
  const esc  = root.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re   = new RegExp('^' + esc + '-(\\d+)\\.json$', 'i');
  let max = 0;
  for (const e of (index || [])) {
    const m = e && typeof e.file === 'string' ? e.file.match(re) : null;
    if (m) max = Math.max(max, Number(m[1]));
  }
  return root + '-' + (max + 1) + '.json';
}

// Öffnet den Speichern-Dialog mit vorbelegten Feldern. Modus 2 übernimmt die
// Metadaten des geladenen Spiels, Modus 1 startet leer.
function openSaveDialog() {
  if (recordedMoves.length === 0) return;
  const $ = id => document.getElementById(id);
  const g = lastLoadedGame || {};
  // Aufzeichnung eines bestehenden Spiels → alter Name + fortlaufende Nummer,
  // sonst eine frische Partie mit Zeitstempel-Namen.
  $('rec-dlg-filename').value = lastLoadedFile
    ? nextVariantFilename(lastLoadedFile, gamesIndex)
    : 'recording-' + Date.now() + '.json';
  for (let i = 1; i <= 4; i++) {
    $('rec-dlg-name-' + i).value = g['username' + i] || '';
  }
  $('rec-dlg-gamenr').value = g.gameNr || '';
  $('record-save-dialog').classList.remove('hidden');
  $('rec-dlg-filename').focus();
}

function closeSaveDialog() {
  document.getElementById('record-save-dialog').classList.add('hidden');
}

// Baut das Spiel-JSON (gleiches Format wie gespeicherte Spiele) und schreibt es
// serverseitig über save-game.php (lokal: Node-Pendant in server.js). Danach App-Liste neu laden.
async function saveRecording() {
  const $ = id => document.getElementById(id);
  const g = lastLoadedGame || {};
  let filename = ($('rec-dlg-filename').value || '').trim();
  if (!filename) filename = 'recording-' + Date.now() + '.json';
  if (!filename.endsWith('.json')) filename += '.json';
  if (!/^[\w.-]+$/.test(filename)) { alert(t('record.badFilename')); return; }

  const engMoves = [...recordPrefix.map(m => m.engine), ...recordedMoves.map(m => m.engine)];
  const times    = [...recordPrefix.map(m => m.time),   ...recordedMoves.map(m => m.time)];
  const pgn4 = buildPgn4(engMoves, times);

  const gameNrRaw = ($('rec-dlg-gamenr').value || '').trim();
  const game = {
    gameNr: gameNrRaw ? (Number(gameNrRaw) || gameNrRaw) : Date.now(),
    pgn4,
    result: g.result || '',
    chat: [],
  };
  for (let i = 1; i <= 4; i++) {
    game['username' + i] = ($('rec-dlg-name-' + i).value || '').trim();
    game['uid' + i]      = g['uid' + i] || '';
    game['rating' + i]   = g['rating' + i] ?? null;
  }

  try {
    // Aufzeichnung eines bestehenden Spiels: der Server vergibt anhand von
    // baseFile den nächsten freien Namen "alterName-<Nr>.json" (autoritativ, auch
    // bei zwischenzeitlich veraltetem Client-Index).
    const r = await fetch('save-game.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename, game, baseFile: lastLoadedFile || undefined }),
    });
    if (!r.ok) { alert(t('record.saveFailed', { err: r.status + ' ' + await r.text() })); return; }
    let savedFile = filename;
    try { const j = await r.json(); if (j && j.file) savedFile = j.file; } catch { /* Antwort optional */ }
    closeSaveDialog();
    stopRecording();
    // Index neu holen (cache-bust), damit das Spiel sofort in der Liste steht.
    try {
      const ir = await fetch('games_index.json?t=' + Date.now());
      if (ir.ok) { gamesIndex = await ir.json(); renderGameFilterResults(); }
    } catch { /* Liste optional */ }
    alert(t('record.saved', { file: savedFile }));
  } catch (err) {
    alert(t('record.saveFailed', { err: err.message }));
  }
}

document.getElementById('btn-record-new').addEventListener('click', startRecordingFresh);
document.getElementById('btn-record-branch').addEventListener('click', startRecordingBranch);
document.getElementById('btn-record-stop').addEventListener('click', stopRecording);
document.getElementById('btn-record-save').addEventListener('click', openSaveDialog);
document.getElementById('rec-dlg-ok').addEventListener('click', saveRecording);
document.getElementById('rec-dlg-cancel').addEventListener('click', closeSaveDialog);

// ─── Netz laden / entladen ────────────────────────────────────────────────────
document.getElementById('net-file-input').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const err = engine.load_network_json(ev.target.result);
    if (err) {
      netStatus.textContent = t('net.autoLoadFailed', { err });
      netStatus.className = 'err';
    } else {
      const info = engine.network_info();
      netStatus.textContent = t('net.autoLoaded', {
        steps:  info.steps.toLocaleString(),
        params: info.params.toLocaleString(),
      }).replace('weights.json', file.name);
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
  netStatus.textContent = t('net.empty');
  netStatus.className = '';
  unloadBtn.disabled = true;
  netBars.classList.add('hidden');
});

// Beim Start: weights.json vom Server holen und automatisch ins Netz laden.
// Schlägt fehl/leise, wenn die Datei nicht existiert (kein Training gelaufen) \u2013
// File-Picker bleibt als manueller Override.
(async () => {
  try {
    const r = await fetch('weights.json');
    if (!r.ok) return;
    const text = await r.text();
    const err  = engine.load_network_json(text);
    if (err) {
      netStatus.textContent = t('net.autoLoadFailed', { err });
      netStatus.className = 'err';
      return;
    }
    const info = engine.network_info();
    netStatus.textContent = t('net.autoLoaded', {
      steps:  info.steps.toLocaleString(),
      params: info.params.toLocaleString(),
    });
    netStatus.className = 'ok';
    unloadBtn.disabled = false;
    draw();
  } catch { /* offline / kein Server \u2013 egal */ }
})();

// Beim Start: opening_book.json holen und in die Engine laden. Fehlende Datei
// ist OK (Engine sucht dann ohne Buch).
(async () => {
  const bookStatus = document.getElementById('book-status');
  try {
    const r = await fetch('opening_book.json');
    if (!r.ok) return;
    const text = await r.text();
    const err  = engine.load_book_json(text);
    if (err) {
      if (bookStatus) {
        bookStatus.textContent = t('book.autoLoadFailed', { err });
        bookStatus.className = 'err';
      }
      return;
    }
    const info = engine.book_info();
    if (bookStatus) {
      bookStatus.textContent = t('book.autoLoaded', {
        positions: info.positions.toLocaleString(),
      });
      bookStatus.className = 'ok';
    }
    // Buch geladen \u2192 Checkbox standardm\u00e4\u00dfig aktiv (kein Replay-Spiel geladen)
    const bookCb = document.getElementById('use-opening-book');
    if (bookCb) { bookCb.checked = true; renderTopMoves(); }
  } catch { /* offline / kein Server \u2013 egal */ }
})();

// Checkbox: Er\u00f6ffnungsbuch verwenden
document.getElementById('use-opening-book')?.addEventListener('change', () => {
  renderTopMoves();
});

// ─── PGN ─────────────────────────────────────────────────────────────────────
document.getElementById('btn-load-pgn').addEventListener('click', () => {
  const pgn = document.getElementById('pgn-input').value.trim();
  if (!pgn) return;
  previewMv = null; previewRow = null;
  const err = engine.load_pgn(pgn);
  if (err) { alert(t('pgn.error', { err })); return; }
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  selected = null; legalMoves = [];
  draw();
});

document.getElementById('btn-export-pgn').addEventListener('click', () => {
  document.getElementById('pgn-output').textContent = engine.export_pgn();
});

// ─── Replay-Zustand ───────────────────────────────────────────────────────────
let replayMoves         = [];
let replayMovesOriginal = [];   // unveränderliche Kopie der geladenen Partie
let replayIdx           = 0;
let replayEvals       = [];   // vorberechnete Analysedaten (game_analysis/*.json)
let replayAnnotations = [];   // Analyse-Annotationen (annotations-Feld aus game_analysis/*.json)

// Forfeits/Resignations aus dem Chat (Engine erkennt nur Königsschläge).
// `appliedForfeitStack` = bereits angewandte Forfeits in chronologischer Reihen-
// folge; `forfeitsAppliedAtIdx[i]` = wie viele Forfeits direkt vor dem Zug an
// replayIdx=i angewandt wurden (für sauberes Undo).
let allForfeits          = [];
let appliedForfeitStack  = [];
let forfeitsAppliedAtIdx = [];
// Bonuspunkte pro Farbe, die durch Forfeit-Notizen ("(Green +3)") angefallen
// sind — die Engine kennt sie nicht, also addieren wir sie in updateScores().
let forfeitBonus = [0, 0, 0, 0];
// Forfeits, die NACH dem letzten echten Zug angewandt wurden (z.B. Time-Forfeit
// am Spielende ohne weiteren Zug). Werden beim Zurückblättern vom Ende aus
// rückgängig gemacht — getrennt vom regulären forfeitsAppliedAtIdx[], weil sie
// keinem konkreten Zug-Index vorgeschaltet sind.
let lateForfeitCount = 0;
// Bonuspunkte für Königsschläge an bereits ausgestiegenen (aber noch nicht
// eliminierten) Spielern — die Engine vergibt 0, chess.com aber +3. Pro
// replayIdx max. ein Eintrag {mover, points} für sauberes Undo.
let resignedKingBonusAtIdx = [];

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
const moveLogEntries = []; // {text, color, notation?, score?}
const PLY_COLORS = ['red', 'blue', 'yellow', 'green'];

// Erbeutete Figuren je Spieler (Index = PLY_COLORS-Idx). Pro Eintrag
// {color, kind} der geschlagenen Figur. captureHistory hält für jeden
// gespielten Halbzug das Schlag-Resultat, damit Undo/Redo es rückgängig
// machen kann.
const capturedByPlayer = [[], [], [], []];
const captureHistory   = []; // jeder Eintrag: {mover, color, kind} | null

function colorIdx(name) { return PLY_COLORS.indexOf(name.toLowerCase()); }

// Vor jedem engine.apply_move aufrufen: schaut sich das Zielfeld im
// AKTUELLEN state an und merkt sich, was geschlagen wird. Liefert den
// Datensatz oder null zurück (für Logging-Zwecke).
function recordCaptureFromMove(toSq) {
  if (!state) { captureHistory.push(null); return null; }
  const piece    = state.squares[toSq];
  const moverIdx = colorIdx(state.to_move);
  if (piece && moverIdx >= 0 && piece.color !== state.to_move) {
    const rec = { mover: moverIdx, color: piece.color, kind: piece.kind };
    capturedByPlayer[moverIdx].push({ color: rec.color, kind: rec.kind });
    captureHistory.push(rec);
    return rec;
  }
  captureHistory.push(null);
  return null;
}

// Nach engine.undo aufrufen: nimmt den letzten Schlag aus dem History und
// entfernt das zugehörige Icon vom Trophäenstapel.
function undoLastCapture() {
  const rec = captureHistory.pop();
  if (rec) capturedByPlayer[rec.mover].pop();
}

function clearCaptures() {
  for (let i = 0; i < 4; i++) capturedByPlayer[i].length = 0;
  captureHistory.length = 0;
}

// Render-Helper. Eine Zeile pro Figurenart (Königswert oben, Bauern unten).
const KIND_ORDER = { King: 0, Boat: 1, Bishop: 2, Knight: 3, Pawn: 4 };
function renderCaptures() {
  for (let i = 0; i < 4; i++) {
    const el = document.getElementById('captures-' + PLY_COLORS[i]);
    if (!el) continue;
    // Nach Figurentyp gruppieren, Reihenfolge per KIND_ORDER
    const groups = new Map();
    for (const p of capturedByPlayer[i]) {
      if (!groups.has(p.kind)) groups.set(p.kind, []);
      groups.get(p.kind).push(p);
    }
    const sorted = [...groups.entries()]
      .sort((a, b) => (KIND_ORDER[a[0]] ?? 9) - (KIND_ORDER[b[0]] ?? 9));
    let html = '';
    for (const [, pieces] of sorted) {
      html += '<div class="capture-row">';
      for (const p of pieces) {
        html += '<img src="pieces/' + p.color.toLowerCase() + '-' + p.kind.toLowerCase()
              + '.svg" alt="' + p.color + ' ' + p.kind + '">';
      }
      html += '</div>';
    }
    el.innerHTML = html;
  }
}

// Stack rückgängig gemachter Züge im normalen Spielmodus.
// Wird gefüllt durch playUndoStep(), geleert durch jeden neuen Zug.
const playRedoStack = []; // {text, color, notation}

function updatePlayReplayButtons() {
  const prev = document.getElementById('btn-play-prev');
  const next = document.getElementById('btn-play-next');
  if (!prev || !next) return;
  prev.disabled = plyCount === 0;
  next.disabled = playRedoStack.length === 0;
}

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

// Umkehrung von ccSqToEngine: Engine-Feld ("d4", a–h × 1–8) → cc-Feld des
// 14×14-Bretts ("g7"), indem Datei und Reihe je um 3 verschoben werden.
function engSqToCc(sq) {
  const file = sq.charCodeAt(0) - 97;        // a=0 … h=7
  const rank = parseInt(sq.slice(1), 10);    // 1-8
  return String.fromCharCode(97 + file + 3) + (rank + 3);
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

// Umkehrung von parsePgn4Moves: baut aus einer Liste von Engine-Notationen
// (z.B. ["c2c3", "e2e3", …]) einen chess.com-kompatiblen pgn4-String, den
// loadGame()/parsePgn4Moves wieder einlesen kann.
//   - Rundenzähler `1.` `2.` … alle 4 Halbzüge.
//   - Feld-Tokens im 14×14-Raster (engSqToCc), Trenner `x` bei Schlag sonst `-`,
//     optionaler Figurbuchstabe (K/B/N/R, Bauer = keiner) — beides rein
//     kosmetisch, da ccMoveToEngine es beim Reload wieder verwirft.
//   - optionaler `{date=…}`-Kommentar aus den Aufnahme-Zeitstempeln.
// Eine Scratch-Engine bestimmt Schlag/Figur, ohne das UI-Brett zu verändern.
function buildPgn4(engineMoves, times) {
  const PIECE_LETTER = { King: 'K', Bishop: 'B', Knight: 'N', Boat: 'R', Pawn: '' };
  const scratch = new WasmEngine();
  let scratchOk = true;
  const tokens = [];
  for (let i = 0; i < engineMoves.length; i++) {
    if (i % 4 === 0) tokens.push(`${i / 4 + 1}.`);
    const mv = engineMoves[i];
    if (!mv || mv === '--') { tokens.push('--'); continue; }
    const from = mv.slice(0, 2), to = mv.slice(2, 4);
    let letter = '', sep = '-';
    if (scratchOk) {
      const st = scratch.get_state();
      const fromSq = engineSqIdx(from), toSq = engineSqIdx(to);
      const p = st.squares[fromSq];
      if (p) letter = PIECE_LETTER[p.kind] ?? '';
      if (st.squares[toSq]) sep = 'x';
    }
    const ccFrom = engSqToCc(from), ccTo = engSqToCc(to);
    let token = letter + ccFrom + sep + ccTo;
    const ts = times && times[i];
    if (ts) token += ` {date=${new Date(ts).toISOString()}}`;
    tokens.push(token);
    // Scratch-Engine nachziehen (mit Promotions-Fallback wie im Replay); bei
    // Desync (z.B. Chat-Time-Forfeit im Prefix) nur noch schmucklose Tokens.
    if (!(scratch.apply_move(mv) || scratch.apply_move(mv + 'p'))) scratchOk = false;
  }
  return '[Event "Chaturaji"]\n[Result "*"]\n\n' + tokens.join(' ') + '\n';
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
  renderTopMoves();
}

function applyPreview(t, row) {
  if (isBusy) return;
  const wasActive = previewMv === t.mv;
  clearPreview();
  if (wasActive) { draw(); return; }

  const fromSq = engineSqIdx(t.mv.slice(0, 2));
  const toSq   = engineSqIdx(t.mv.slice(2, 4));
  const movingPiece  = state?.squares[fromSq] ?? null;
  const capturePiece = state?.squares[toSq]   ?? null;
  const inReplay = replayMoves.length > 0 && replayIdx < replayMoves.length;

  // Vorberechnete Top-Züge enthalten keinen Promotionssuffix; falls der reine
  // Zug abgelehnt wird, ist es vermutlich eine Bauernumwandlung — Retry mit
  // 'p' (Promotion zu Boat, einzige Promotion in Chaturaji).
  if (inReplay) recordCaptureFromMove(toSq);
  let notation = null;
  if      (engine.apply_move(t.mv))       notation = t.mv;
  else if (engine.apply_move(t.mv + 'p')) notation = t.mv + 'p';
  if (!notation) { if (inReplay) undoLastCapture(); return; }

  if (inReplay) {
    // Replay-Modus: Zug permanent einbuchen, nicht als Vorschau
    appendMoveLog({ notation, captures: capturePiece !== null, promoted: notation.length > 4 });
    state = engine.get_state();
    renderCaptures();
    commitToReplay(notation);
    renderTopMoves();
    setBusy(true);
    animateMove(fromSq, toSq, movingPiece, () => { draw(); setBusy(false); });
  } else {
    previewMv  = t.mv;
    previewRow = row;
    row.classList.add('active');
    state = engine.get_state();
    renderTopMoves();
    setBusy(true);
    animateMove(fromSq, toSq, movingPiece, () => { draw(); setBusy(false); });
  }
}

function renderTopMoves() {
  const panel      = document.getElementById('top-moves-panel');
  const list       = document.getElementById('top-moves-list');
  const sourceEl   = document.getElementById('top-moves-source');
  const bookCb     = document.getElementById('use-opening-book');
  if (!panel || !list) return;

  const useBook = bookCb?.checked ?? false;
  let tops      = null;
  let fromBook  = false;

  if (useBook) {
    // Buchzüge für die aktuelle Stellung holen (Top 3, count >= 1)
    const bookMoves = engine.book_top_moves(3);
    if (bookMoves && bookMoves.length > 0) {
      tops     = bookMoves;
      fromBook = true;
    }
  }

  if (!tops) {
    if (previewMv !== null) { panel.classList.add('hidden'); return; }
    // Fallback: vorberechnete Analysedaten
    const evalEntry = replayEvals[replayIdx] ?? null;
    tops = evalEntry?.top ?? null;
    fromBook = false;
  }

  if (!tops || tops.length === 0) { panel.classList.add('hidden'); return; }

  const st         = engine.get_state();
  const COLOR_CLASS = ['red', 'blue', 'yellow', 'green'];
  const playerIdx  = ['Red', 'Blue', 'Yellow', 'Green'].indexOf(st.to_move);
  const colorClass = COLOR_CLASS[playerIdx] ?? '';
  const barColor   = COLOR_HEX[st.to_move] ?? '#f0c040';

  if (sourceEl) {
    if (fromBook) {
      sourceEl.textContent = t('topMoves.sourceBook');
      sourceEl.classList.remove('hidden');
    } else if (replayEvals.length > 0) {
      sourceEl.textContent = t('topMoves.sourceAnalysis');
      sourceEl.classList.remove('hidden');
    } else {
      sourceEl.classList.add('hidden');
    }
  }

  list.innerHTML = '';
  previewMv = null; previewRow = null;
  for (const mv of tops) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = 'top-move-row';
    const pctLabel = fromBook
      ? mv.count.toLocaleString() + '×'
      : mv.pct + '%';
    row.innerHTML =
      `<span class="top-move-notation ${colorClass}">${engineMoveToDisplay(mv.mv)}</span>` +
      `<div class="top-move-bar-wrap"><div class="top-move-bar" style="width:${mv.pct}%;background:${barColor}"></div></div>` +
      `<span class="top-move-pct">${pctLabel}</span>`;
    row.addEventListener('click', () => applyPreview(mv, row));
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
// Für Forfeits steht oft direkt im Anschluss eine "(<Color> +N)"-Notiz, mit der
// chess.com einem Mitspieler Bonuspunkte gutschreibt — die übernehmen wir, weil
// die Engine selbst keine Forfeit-Boni vergibt.
function extractForfeits(chat, nameToColor) {
  const out = [];
  const cap = (c) => c[0].toUpperCase() + c.slice(1).toLowerCase();
  for (let i = 0; i < chat.length; i++) {
    const m = chat[i];
    if (typeof m?.message !== 'string' || m.fullMoveNr == null) continue;
    let color = null;
    // Variante 1: "<Color> resigned/forfeits"
    let match = m.message.match(/^(Red|Blue|Yellow|Green)\s+(forfeits|resigned)/i);
    if (match) color = cap(match[1]);
    else {
      // Variante 2: "<PlayerName> resigned/forfeits" — Name → Farbe lookup
      match = m.message.match(/^([\w.\-]+)\s+(forfeits|resigned)/i);
      if (match && nameToColor) {
        const c = nameToColor[match[1]] || nameToColor[match[1].toLowerCase()];
        if (c) color = cap(c);
      }
    }
    if (!color) continue;
    // Auf eine direkt folgende "(<Color> +N)"-Notiz schauen
    let bonusColor = null, bonusPoints = 0;
    for (let j = i + 1; j < chat.length && j <= i + 3; j++) {
      const next = chat[j];
      if (next?.fullMoveNr !== m.fullMoveNr) break;
      const bm = (next?.message || '').match(/^\((Red|Blue|Yellow|Green)\s*\+(\d+)\)/i);
      if (bm) { bonusColor = cap(bm[1]); bonusPoints = parseInt(bm[2], 10); break; }
    }
    out.push({ fmn: m.fullMoveNr, color, bonusColor, bonusPoints });
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
  lastLoadedGame = gameData;   // Metadaten-Quelle für "Ab hier aufzeichnen" (Modus 2)
  lastLoadedFile = null;       // wird von loadGameFromIndex() mit dem Dateinamen belegt
  const pgn4  = gameData.pgn4  || '';
  const chat  = gameData.chat  || [];
  const colors = ['Red','Blue','Yellow','Green'];

  const uidToColor  = {};
  const nameToColor = {};
  const players     = [];
  for (let i = 1; i <= 4; i++) {
    const uid      = gameData['uid'      + i];
    const username = gameData['username' + i];
    if (uid)      uidToColor[uid] = colors[i - 1];
    if (username) {
      nameToColor[username]               = colors[i - 1];
      nameToColor[username.toLowerCase()] = colors[i - 1];
      players.push({ color: colors[i - 1].toLowerCase(), username });
    }
  }

  const gameLink = document.getElementById('gameLink');

  if (gameLink) gameLink.setAttribute('href', 'https://www.chess.com/variants/chaturaji/game/' + gameData.gameNr);

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
      // Spielername hat Vorrang vor Sprachwechsel — i18n-Marker entfernen.
      el.removeAttribute('data-i18n');
    }
  }

  replayMoves         = parsePgn4Moves(pgn4);
  replayMovesOriginal = replayMoves.map(m => ({ ...m }));
  replayIdx           = 0;
  allForfeits = extractForfeits(chat, nameToColor);
  appliedForfeitStack.length = 0;
  forfeitsAppliedAtIdx.length = 0;
  forfeitBonus = [0, 0, 0, 0]; lateForfeitCount = 0;
  resignedKingBonusAtIdx.length = 0;
  playerClocks = [null, null, null, null];
  prevPlayerClocks.length = 0;
  refreshAllScoreTimes();

  previewMv = null; previewRow = null;
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  selected = null; legalMoves = [];

  document.getElementById('replay-controls').classList.remove('hidden');
  document.getElementById('play-replay-controls').classList.add('hidden');
  playRedoStack.length = 0;
  updatePlayReplayButtons();
  clearCaptures(); renderCaptures();
  updateReplayPos();
  draw();

  // Spiel geladen → Eröffnungsbuch-Checkbox deaktivieren (Analyse-Daten bevorzugt)
  const bookCb = document.getElementById('use-opening-book');
  if (bookCb) bookCb.checked = false;

  // Frisch geladenes Spiel → evtl. laufende Aufnahme beenden, "Ab hier
  // aufzeichnen" ist jetzt verfügbar.
  recording = false; recordedMoves = []; recordPrefix = [];
  updateRecordControls();
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
        replayEvals       = data.evals;
        replayAnnotations = Array.isArray(data.annotations) ? data.annotations : [];
        renderTopMoves();
        renderAnnotationPanel();
      } else {
        try {
          const r = await fetch('game_analysis/' + file.name);
          if (r.ok) {
            const ana = await r.json();
            replayEvals       = ana.evals        || [];
            replayAnnotations = Array.isArray(ana.annotations) ? ana.annotations : [];
            renderTopMoves();
            renderAnnotationPanel();
          }
        } catch  (err) { alert(t('error.loadGame', { file: file.name })); }
      }
    } catch  (err) { alert(t('error.loadGame', { file: file.name })); }
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
      replayEvals       = data.evals        || [];
      replayAnnotations = Array.isArray(data.annotations) ? data.annotations : [];
      renderTopMoves();
      renderAnnotationPanel();
    } catch (err) { alert(t('error.loadAnalysis', { err: err.message })); }
  };
  reader.readAsText(file);
  e.target.value = '';
});

// ─── Analyse-Annotationen ────────────────────────────────────────────────────
const COLOR_CLASS_ANN = ['red', 'blue', 'yellow', 'green'];

function renderAnnotationPanel() {
  const panel = document.getElementById('annotation-panel');
  const list  = document.getElementById('annotation-list');
  if (!panel || !list) return;
  if (!replayAnnotations || replayAnnotations.length === 0) {
    panel.classList.add('hidden');
    return;
  }
  list.innerHTML = '';
  for (const ann of replayAnnotations) {
    const li = document.createElement('li');
    const colorClass = COLOR_CLASS_ANN[ann.player] ?? '';
    li.className = 'annotation-entry ' + colorClass;
    li.innerHTML =
      '<span class="ann-dot">●</span>' +
      '<span class="ann-msg">' + (ann.message || '') + '</span>';
    li.addEventListener('click', () => replayJumpTo(ann.moveIdx ?? 0));
    list.appendChild(li);
  }
  panel.classList.remove('hidden');
}

function replayJumpTo(targetIdx) {
  if (animReq !== null) { cancelAnimationFrame(animReq); animReq = null; }
  previewMv = null; previewRow = null;
  clearEngineCandidates();
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  replayIdx = 0;
  appliedForfeitStack.length = 0;
  forfeitsAppliedAtIdx.length = 0;
  forfeitBonus = [0, 0, 0, 0]; lateForfeitCount = 0;
  resignedKingBonusAtIdx.length = 0;
  playerClocks = [null, null, null, null];
  prevPlayerClocks.length = 0;
  refreshAllScoreTimes();
  selected = null; legalMoves = [];
  gameChatIdx = 0;
  document.getElementById('chat-messages').innerHTML = '';

  const target = Math.min(targetIdx, replayMoves.length);
  while (replayIdx < target) replayApplyNext();

  state = engine.get_state();
  updateReplayPos();
  renderTopMoves();
  draw();
}

// ─── Replay-Navigation ────────────────────────────────────────────────────────
function replayApplyNext() {
  // Laufende Animation sofort beenden
  if (animReq !== null) { cancelAnimationFrame(animReq); animReq = null; draw(); }
  clearPreview();
  clearEngineCandidates();

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
        applyForfeitBonus(f);
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
    recordCaptureFromMove(toSq);
    // Promotionsfallback (siehe applyPreview): chess.com-PGN4 markiert
    // Bauernumwandlungen nicht explizit, daher ggf. mit 'p' nachreichen.
    if (!(engine.apply_move(mv.engine) || engine.apply_move(mv.engine + 'p'))) {
      undoLastCapture();
    } else {
      // Königsschlag eines ausgestiegenen (active=false), aber noch nicht
      // eliminierten (König stand noch) Spielers: die Engine vergibt 0 Punkte
      // (Capture-Score-Check liest active[]), chess.com aber +3. Bonus
      // nachpflegen. `stBefore` ist garantiert nach Forfeit-Anwendung,
      // aber vor dem Zug abgegriffen — auch im seltenen Fall, dass Forfeit
      // und Königsschlag im selben fullMoveNr passieren.
      const cap = captureHistory[captureHistory.length - 1];
      if (cap && cap.kind === 'King' && stBefore) {
        const ci = ['Red','Blue','Yellow','Green'].indexOf(cap.color);
        if (ci >= 0 && !stBefore.active[ci]) {
          forfeitBonus[cap.mover] += 3;
          resignedKingBonusAtIdx[replayIdx] = { mover: cap.mover, points: 3 };
        }
      }
    }
    renderCaptures();
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

  // Nach dem letzten Zug noch ausstehende Forfeits anwenden (z.B. Time-Forfeit
  // OHNE Folgezug — fmn liegt dann hinter dem letzten echten Zug). Das schaltet
  // die Bonuspunkte aus "(Color +N)"-Notizen frei, die sonst nicht zur Anzeige
  // kommen würden.
  const isLastMove = replayIdx === replayMoves.length;
  if (isLastMove) {
    for (const f of allForfeits) {
      if (appliedForfeitStack.includes(f)) continue;
      if (engine.forfeit_color(f.color)) {
        appliedForfeitStack.push(f);
        lateForfeitCount++;
        applyForfeitBonus(f);
      }
    }
  }

  // Sobald die letzte Runde mit Chat erreicht ist, alle restlichen Nachrichten
  // freigeben — sonst würden Post-Game-Kommentare (deren `time` nach dem
  // letzten Zug liegt) bis zum Spielende stumm bleiben.
  const chatUpTo =
    isLastMove ? Infinity
    : (maxChatFullMoveNr > 0 && mv.fullMoveNr >= maxChatFullMoveNr) ? Infinity
    : mv.time;
  if (chatUpTo != null) renderChatUpTo(chatUpTo);

  // Zustand nach dem Zug holen und Figur animiert bewegen
  state = engine.get_state();
  animateMove(fromSq, toSq, movingPiece, () => draw());
  renderTopMoves();
}

// Forfeit-Boni aus "(Color +N)"-Chat-Notizen zur Anzeige addieren/entfernen.
function applyForfeitBonus(f) {
  if (!f?.bonusColor || !f.bonusPoints) return;
  const idx = ['Red','Blue','Yellow','Green'].indexOf(f.bonusColor);
  if (idx >= 0) forfeitBonus[idx] += f.bonusPoints;
}
function undoForfeitBonus(f) {
  if (!f?.bonusColor || !f.bonusPoints) return;
  const idx = ['Red','Blue','Yellow','Green'].indexOf(f.bonusColor);
  if (idx >= 0) forfeitBonus[idx] -= f.bonusPoints;
}

function replayUndoPrev() {
  if (previewMv !== null) { clearPreview(); draw(); return; }
  clearEngineCandidates();
  if (replayIdx <= 0) return;

  // Wenn wir gerade vom Ende zurückblättern: zuerst etwaige Late-Forfeits
  // (die NACH dem letzten Zug angewandt wurden) rückgängig machen, dann den
  // letzten Zug. Stack-Reihenfolge: …, move, late_1, …, late_K → top zuerst.
  if (replayIdx === replayMoves.length && lateForfeitCount > 0) {
    for (let i = 0; i < lateForfeitCount; i++) {
      engine.undo();
      const popped = appliedForfeitStack.pop();
      undoForfeitBonus(popped);
    }
    lateForfeitCount = 0;
  }

  replayIdx--;
  // Anim-Daten für die Rückbewegung BEVOR engine.undo() den Zustand kippt.
  let animFrom = -1, animTo = -1, animPiece = null;
  const undoMv = replayMoves[replayIdx];
  if (undoMv.engine !== '--' && typeof undoMv.engine === 'string' && undoMv.engine.length >= 4) {
    animFrom = engineSqIdx(undoMv.engine.slice(0, 2));
    animTo   = engineSqIdx(undoMv.engine.slice(2, 4));
    animPiece = state?.squares[animTo] ?? null;
  }
  if (undoMv.engine !== '--') {
    engine.undo();
    undoLastCapture();
    renderCaptures();
    // Bonus für Königsschlag an einem ausgestiegenen Spieler zurücknehmen
    const rkb = resignedKingBonusAtIdx[replayIdx];
    if (rkb) {
      forfeitBonus[rkb.mover] -= rkb.points;
      delete resignedKingBonusAtIdx[replayIdx];
    }
  }
  // Forfeits zurücknehmen, die unmittelbar vor diesem Zug angewandt wurden.
  // Reihenfolge im Engine-History-Stack: …, forfeit_n, …, forfeit_1, move →
  // erst der move (oben) wurde gerade gepoppt, jetzt rückwärts die Forfeits.
  const fc = forfeitsAppliedAtIdx[replayIdx] || 0;
  for (let i = 0; i < fc; i++) {
    engine.undo();
    const popped = appliedForfeitStack.pop();
    undoForfeitBonus(popped);
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
  state = engine.get_state();
  if (animPiece && animFrom !== animTo) {
    animateMove(animTo, animFrom, animPiece, () => draw());
  } else {
    draw();
  }
  renderTopMoves();
}

document.getElementById('btn-replay-start').addEventListener('click', () => {
  previewMv = null; previewRow = null;
  clearEngineCandidates();
  engine.reset();
  moveList.innerHTML = '';
  plyCount = 0; moveLogEntries.length = 0;
  replayMoves = replayMovesOriginal.map(m => ({ ...m }));
  replayIdx = 0;
  appliedForfeitStack.length = 0;
  forfeitsAppliedAtIdx.length = 0;
  forfeitBonus = [0, 0, 0, 0]; lateForfeitCount = 0;
  resignedKingBonusAtIdx.length = 0;
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
    const r = await fetch('game_analysis/' + filename);
    if (!r.ok) { alert(t('error.fileNotReachable', { file: filename })); return; }
    const data = await r.json();
    loadGame(data);
    lastLoadedFile = filename;   // Basis für Aufzeichnungs-Dateiname (alter Name + Nr.)
    if (Array.isArray(data.evals)) {
      replayEvals       = data.evals;
      replayAnnotations = Array.isArray(data.annotations) ? data.annotations : [];
      renderTopMoves();
      renderAnnotationPanel();
    }
  } catch (err) {
    alert(t('error.cannotLoad', { file: filename, err: err.message }));
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

// URL-Parameter ?game=<nr> oder Pfad /<nr> → entsprechendes Analyse-File auto-laden.
// Akzeptiert mit oder ohne ".json"-Suffix. Zusätzliches ?m=<nr> spult nach dem
// Laden bis zum Halbzug <nr> vor (0 = Startstellung, wird auf die Zuglänge begrenzt).
(async () => {
  const search  = new URLSearchParams(window.location.search);
  const qParam  = search.get('game');
  const pathSeg = window.location.pathname.replace(/^\//, '').replace(/\/$/, '');
  const param   = qParam || (/^\d+$/.test(pathSeg) ? pathSeg : null);
  if (!param) return;
  const filename = param.endsWith('.json') ? param : param + '.json';
  await loadGameFromIndex(filename);

  const mRaw = search.get('m');
  if (mRaw !== null && replayMoves.length > 0) {
    const m = Number(mRaw);
    if (Number.isFinite(m) && m > 0) replayJumpTo(Math.floor(m));
  }
})();

// ─── Brettorientierung ────────────────────────────────────────────────────────
// viewRot = 0 → Rot unten, 1 → Blau unten, 2 → Gelb unten, 3 → Grün unten.
// Tabelle: Welche Farbe sitzt für jede Rotation in welchem Eckslot?
//   layouts[viewRot] = [TL, TR, BL, BR]
const SCORE_LAYOUTS = [
  ['blue',   'yellow', 'red',    'green'],   // viewRot=0  (Rot unten)
  ['yellow', 'green',  'blue',   'red'],     // viewRot=1  (Blau unten)
  ['green',  'red',    'yellow', 'blue'],    // viewRot=2  (Gelb unten)
  ['red',    'blue',   'green',  'yellow'],  // viewRot=3  (Grün unten)
];

function applyOrientation() {
  const top    = document.getElementById('score-top');
  const bottom = document.getElementById('score-bottom');
  const [tl, tr, bl, br] = SCORE_LAYOUTS[viewRot];
  const box = c => document.querySelector('.score.' + c);
  const cap = c => document.getElementById('captures-' + c);

  // Slot-Klassen aktualisieren (links = border-left, rechts = border-right + row-reverse)
  for (const c of ['red', 'blue', 'yellow', 'green']) {
    const el = box(c);
    if (!el) continue;
    el.classList.remove('left-slot', 'right-slot');
    el.classList.add((c === tl || c === bl) ? 'left-slot' : 'right-slot');
  }
  // Score-Boxen in die richtigen Container hängen (Reihenfolge: links-zuerst)
  top.append(box(tl), box(tr));
  bottom.append(box(bl), box(br));

  // Captures-Boxen folgen ihrer Score-Box in den passenden Eck-Slot.
  const slots = {
    TL: document.getElementById('cap-slot-TL'),
    TR: document.getElementById('cap-slot-TR'),
    BL: document.getElementById('cap-slot-BL'),
    BR: document.getElementById('cap-slot-BR'),
  };
  if (slots.TL && cap(tl)) slots.TL.appendChild(cap(tl));
  if (slots.TR && cap(tr)) slots.TR.appendChild(cap(tr));
  if (slots.BL && cap(bl)) slots.BL.appendChild(cap(bl));
  if (slots.BR && cap(br)) slots.BR.appendChild(cap(br));
}

function setBottomLeftColor(color) {
  const next = SCORE_LAYOUTS.findIndex(l => l[2] === color);
  if (next < 0 || next === viewRot) return;
  viewRot = next;
  try { localStorage.setItem('viewRot', String(viewRot)); } catch (_) {}
  applyOrientation();
  draw();
}

// Doppelklick auf einen Score-Namen → den Spieler nach unten links drehen.
document.querySelectorAll('.score .score-name').forEach(el => {
  el.addEventListener('dblclick', () => {
    const scoreBox = el.closest('.score');
    if (!scoreBox) return;
    const color = ['red', 'blue', 'yellow', 'green']
      .find(c => scoreBox.classList.contains(c));
    if (color) setBottomLeftColor(color);
  });
});

// ─── Sprachumschalter ─────────────────────────────────────────────────────────
(function setupLangSwitcher() {
  const sel = document.getElementById('lang-select');
  if (!sel) return;
  sel.innerHTML = '';
  for (const { code, label, flag } of SUPPORTED_LANGS) {
    const o = document.createElement('option');
    o.value = code;
    o.textContent = (flag ? flag + ' ' : '') + label;
    if (code === getLang()) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => { setLang(sel.value); });
})();

// Bei Sprachwechsel UI-Bereiche neu rendern, deren Inhalt dynamisch generiert
// wird (also nicht über data-i18n zurückgesetzt werden kann).
onLanguageChange(() => {
  if (state) updateStatus();
  // Net-Status: nur "leer"-Default re-übersetzen, sonst bleibt aktueller Inhalt.
  if (netStatus && !netStatus.className) netStatus.textContent = t('net.empty');
  const bookStatus = document.getElementById('book-status');
  if (bookStatus && !bookStatus.className) bookStatus.textContent = t('book.empty');
});

function scaleBoardFrame() {
  const frame = document.getElementById('board-frame');
  const parent = frame.parentElement;
  const available = parent.clientWidth;
  const natural = 660 + 2 * (110 + 8); // board + 2×(captures-col + gap)
  const scale = Math.min(1, available / natural);
  frame.style.transform = `scale(${scale})`;
  // Verhindert Leerraum unter dem skalierten Frame
  const scaledH = (660 + 80) * scale; // Brett + Score-Boxen
  frame.style.marginBottom =
    `calc(${scaledH}px - ${660 + 80}px + var(--rem-100))`;
}
window.addEventListener('resize', scaleBoardFrame);
document.addEventListener('DOMContentLoaded', scaleBoardFrame);


// ─── Replay-Controls auf Mobile über das Brett verschieben ───────────────────

const replayControls = document.getElementById('replay-controls');
const boardFrame     = document.getElementById('board-frame');
const mainLeft       = document.querySelector('.main-left');
// Originaler Eltern-Container und Nachfolger merken (für Rückkehr)
const replayOrigParent = replayControls.parentElement;
const replayOrigNextSibling = replayControls.nextSibling;

let replayMoved = false;

function repositionReplayControls() {
  const isMobile = window.innerWidth <= 880;

  if (isMobile && !replayMoved) {
    // Vor #board-frame in .main-left einfügen
    mainLeft.insertBefore(replayControls, boardFrame);
    replayMoved = true;
  } else if (!isMobile && replayMoved) {
    // Zurück an Originalposition
    replayOrigParent.insertBefore(replayControls, replayOrigNextSibling);
    replayMoved = false;
  }
}

window.addEventListener('resize', repositionReplayControls);
repositionReplayControls(); // Einmalig beim Load

// ─── Start ────────────────────────────────────────────────────────────────────
applyOrientation();
updateRecordControls(); // Aufnahme-Buttons initial setzen (nach allen let-Deklarationen)
draw();
