//! WASM bindings für den Chaturaji Engine + neuronales Netz.

mod network;

use wasm_bindgen::prelude::*;
use serde::Serialize;

use chaturaji_core::board::Board;
use chaturaji_core::notation::{move_to_str, parse_move, GameRecord};
use chaturaji_core::piece::Color;
use chaturaji_core::rules::Rules;
use chaturaji_engine::book::OpeningBook;
use chaturaji_engine::search::{Engine as SearchEngine, SearchAlgo};
use network::Network;

// ─── JS-facing types ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct BoardState {
    pub squares: Vec<Option<PieceInfo>>,
    pub to_move: String,
    pub scores:  [i32; 4],
    pub active:  [bool; 4],
    pub is_over: bool,
    pub winner:  Option<String>,
}

#[derive(Serialize)]
pub struct PieceInfo {
    pub kind:  String,
    pub color: String,
}

#[derive(Serialize)]
pub struct MoveInfo {
    pub from:     u8,
    pub to:       u8,
    pub notation: String,
    pub captures: bool,
    pub promoted: bool,
}

#[derive(Serialize)]
pub struct TopMove {
    pub mv:    String,   // engine notation, e.g. "d2d3"
    pub score: i32,      // current player's raw score for this move
    pub pct:   u8,       // 0-100: score relative to best move (best = 100)
}

#[derive(Serialize)]
pub struct EngineResult {
    pub best_move:    Option<String>,
    pub scores:       [i32; 4],
    pub net_values:   Option<[f32; 4]>,
    pub depth:        u8,
    pub nodes:        u64,
    pub used_network: bool,
}

#[derive(Serialize)]
pub struct NetworkInfo {
    pub loaded: bool,
    pub steps:  u64,
    pub lr:     f32,
    pub params: usize,
}

#[derive(Serialize)]
pub struct BookInfo {
    pub loaded:    bool,
    pub positions: usize,
}

#[derive(Serialize)]
pub struct BookMove {
    pub mv:    String,  // engine notation, e.g. "d2d3"
    pub count: u32,
    pub pct:   u8,      // 0-100 relativ zum häufigsten Zug
}

// ─── Engine handle ────────────────────────────────────────────────────────────

/// Which search algorithm the engine entry points use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    /// Best-Reply Search — only the most dangerous opponent replies.
    Brs,
    /// All three opponents minimise the root player's score.
    Paranoid,
}

#[wasm_bindgen]
pub struct WasmEngine {
    board:   Board,
    engine:  SearchEngine,
    history: Vec<Board>,
    network: Option<Network>,
    algo:    Algorithm,
}

#[wasm_bindgen]
impl WasmEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmEngine {
        WasmEngine {
            board:   Board::default(),
            engine:  SearchEngine::new(16),
            history: Vec::new(),
            network: None,
            algo:    Algorithm::Brs,
        }
    }

    /// Pick the search algorithm: `"brs"` (default) or `"paranoid"`.
    /// Unknown values are ignored and reported as `false`.
    ///
    /// Note that `depth` means different things to the two: paranoid spends
    /// four plies per game round, BRS two. BRS at depth 4 therefore looks two
    /// full rounds ahead where paranoid at depth 4 looks one — and does it in
    /// a fraction of the nodes (see `examples/brs_bench.rs`).
    pub fn set_algorithm(&mut self, name: &str) -> bool {
        match name {
            "brs"      => { self.algo = Algorithm::Brs;      true }
            "paranoid" => { self.algo = Algorithm::Paranoid; true }
            _          => false,
        }
    }

    pub fn algorithm(&self) -> String {
        match self.algo {
            Algorithm::Brs      => "brs".to_string(),
            Algorithm::Paranoid => "paranoid".to_string(),
        }
    }

    // ── Board ─────────────────────────────────────────────────────────────────

    pub fn get_state(&self) -> JsValue {
        let squares: Vec<Option<PieceInfo>> = (0u8..64).map(|sq| {
            self.board.piece_at(sq).map(|p| PieceInfo {
                kind:  format!("{:?}", p.kind),
                color: p.color.name().to_string(),
            })
        }).collect();
        let state = BoardState {
            squares,
            to_move: self.board.to_move.name().to_string(),
            scores:  self.board.scores.as_array(),
            active:  self.board.active,
            is_over: Rules::is_game_over(&self.board),
            winner:  Rules::winner(&self.board).map(|c| c.name().to_string()),
        };
        serde_wasm_bindgen::to_value(&state).unwrap()
    }

    pub fn legal_moves_from(&self, from: u8) -> JsValue {
        let moves: Vec<MoveInfo> = Rules::legal_moves(&self.board)
            .into_iter()
            .filter(|mv| mv.from == from)
            .map(|mv| MoveInfo {
                from:     mv.from,
                to:       mv.to,
                notation: move_to_str(&mv),
                captures: mv.captured.is_some(),
                promoted: mv.promoted,
            })
            .collect();
        serde_wasm_bindgen::to_value(&moves).unwrap()
    }

    // ── Züge ──────────────────────────────────────────────────────────────────

    pub fn apply_move(&mut self, notation: &str) -> bool {
        match parse_move(&self.board, notation) {
            Ok(mv) => {
                self.history.push(self.board.clone());
                self.board = Rules::apply_with_effects(&self.board, mv);
                true
            }
            Err(_) => false,
        }
    }

    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.history.pop() {
            self.board = prev; true
        } else { false }
    }

    /// Markiert einen Spieler als ausgeschieden (z.B. nach Time-Forfeit), die
    /// im Gegensatz zum Königsschlag nicht aus dem Brett ableitbar sind.
    /// Schiebt `to_move` weiter, falls der ausgeschiedene Spieler am Zug war.
    /// Pusht den vorherigen Zustand auf den Undo-Stack.
    pub fn forfeit_color(&mut self, color: &str) -> bool {
        let c = match color.to_ascii_lowercase().as_str() {
            "red"    => Color::Red,
            "blue"   => Color::Blue,
            "yellow" => Color::Yellow,
            "green"  => Color::Green,
            _ => return false,
        };
        if !self.board.active[c.idx()] { return false; }
        self.history.push(self.board.clone());
        self.board.active[c.idx()] = false;
        if self.board.to_move == c {
            let mut next = c.next();
            for _ in 0..4 {
                if self.board.active[next.idx()] { break; }
                next = next.next();
            }
            self.board.to_move = next;
        }
        true
    }

    // ── Engine ────────────────────────────────────────────────────────────────

    pub fn best_move(&mut self, depth: u8) -> JsValue {
        let net_values = self.network.as_ref().map(|net| {
            net.forward(&self.board)
        });
        // Split field borrows so the network closure and engine mutation can coexist.
        let algo    = self.algo;
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let result = match algo {
            Algorithm::Brs      => engine.search_brs(board, depth, net_eval),
            Algorithm::Paranoid => engine.search_paranoid(board, depth, net_eval),
        };
        let er = EngineResult {
            best_move:    result.best_move.map(|mv| move_to_str(&mv)),
            scores:       result.scores,
            net_values,
            depth:        result.depth,
            nodes:        result.nodes,
            used_network: self.network.is_some(),
        };
        serde_wasm_bindgen::to_value(&er).unwrap()
    }

    /// Sucht mit Zeitbudget statt mit fester Tiefe.
    ///
    /// Eine feste Tiefe sagt nichts darüber, wie lange der Zug dauern wird:
    /// gemessen kostete eine Runde mehr Vorausschau je nach Stellung das
    /// Hundertfache an Rechenzeit. Wer den Tiefenregler höherstellt, wartet
    /// deshalb unkalkulierbar lang. Mit einem Budget wird die Wartezeit zur
    /// Vorgabe und die Tiefe zum Ergebnis — `depth` im Rückgabewert sagt, wie
    /// weit es gereicht hat.
    ///
    /// `max_depth` bleibt als Deckel bestehen, damit die Engine in einer
    /// ausgedünnten Endstellung nicht sinnlos weitervertieft.
    pub fn best_move_timed(&mut self, budget_ms: f64, max_depth: u8) -> JsValue {
        let net_values = self.network.as_ref().map(|net| net.forward(&self.board));

        // js_sys::Date::now() statt std::time::Instant: letzteres paniziert
        // unter wasm32-unknown-unknown.
        let deadline = js_sys::Date::now() + budget_ms;
        self.engine.set_stop_check(move || js_sys::Date::now() >= deadline);

        let algo    = self.algo;
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let search_algo = match algo {
            Algorithm::Brs      => SearchAlgo::Brs,
            Algorithm::Paranoid => SearchAlgo::Paranoid,
        };
        let result = engine.search_deepening(board, search_algo, max_depth, net_eval);
        self.engine.clear_stop_check();

        let er = EngineResult {
            best_move:    result.best_move.map(|mv| move_to_str(&mv)),
            scores:       result.scores,
            net_values,
            depth:        result.depth,
            nodes:        result.nodes,
            used_network: self.network.is_some(),
        };
        serde_wasm_bindgen::to_value(&er).unwrap()
    }

    /// Returns the top-`n` moves at the current position as a JS array of
    /// `{mv, score, pct}` objects.  `pct` is 0–100 with 100 = best move.
    pub fn top_moves(&mut self, depth: u8, n: u8) -> JsValue {
        let algo    = self.algo;
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let ranked = match algo {
            Algorithm::Brs      => engine.top_n_brs(board, depth, n as usize, net_eval),
            Algorithm::Paranoid => engine.top_n_paranoid(board, depth, n as usize, net_eval),
        };
        let mover_idx = self.board.to_move.idx();

        let best_score = ranked.first()
            .map(|r| r.scores[mover_idx])
            .unwrap_or(1);
        let best_score = if best_score == 0 { 1 } else { best_score };

        let top: Vec<TopMove> = ranked.iter().map(|r| {
            let raw = r.scores[mover_idx];
            let pct = if best_score > 0 {
                ((raw.max(0) as f64 / best_score.max(1) as f64) * 100.0).round().min(100.0) as u8
            } else { 0 };
            TopMove {
                mv:    move_to_str(&r.mv),
                score: raw,
                pct,
            }
        }).collect();

        serde_wasm_bindgen::to_value(&top).unwrap()
    }

    pub fn engine_move(&mut self, depth: u8) -> bool {
        let algo    = self.algo;
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let result = match algo {
            Algorithm::Brs      => engine.search_brs(board, depth, net_eval),
            Algorithm::Paranoid => engine.search_paranoid(board, depth, net_eval),
        };
        if let Some(mv) = result.best_move {
            self.history.push(self.board.clone());
            self.board = Rules::apply_with_effects(&self.board, mv);
            true
        } else { false }
    }

    pub fn evaluate_position(&self) -> JsValue {
        match &self.network {
            Some(net) => serde_wasm_bindgen::to_value(
                &net.forward(&self.board)
            ).unwrap(),
            None => JsValue::NULL,
        }
    }

    // ── Netz ──────────────────────────────────────────────────────────────────

    pub fn load_network_json(&mut self, json: &str) -> Option<String> {
        match serde_json::from_str::<Network>(json) {
            Ok(net) => {
                if let Err(e) = net.validate() {
                    return Some(format!("Netzwerk-Architektur passt nicht: {e}"));
                }
                self.network = Some(net);
                None
            }
            Err(e)  => Some(format!("Fehler: {e}")),
        }
    }

    pub fn network_info(&self) -> JsValue {
        let info = match &self.network {
            Some(net) => NetworkInfo { loaded: true,  steps: net.steps, lr: net.lr, params: net.param_count() },
            None      => NetworkInfo { loaded: false, steps: 0,         lr: 0.0,    params: 0 },
        };
        serde_wasm_bindgen::to_value(&info).unwrap()
    }

    pub fn unload_network(&mut self) { self.network = None; }

    // ── Eröffnungsbuch ────────────────────────────────────────────────────────

    /// Lädt ein vom Trainer geschriebenes Buch (JSON) in die Engine. Solange
    /// das Buch geladen ist und die aktuelle Stellung darin steht, gibt
    /// `best_move` einen Buchzug zurück (depth=0, nodes=0).
    pub fn load_book_json(&mut self, json: &str) -> Option<String> {
        match serde_json::from_str::<OpeningBook>(json) {
            Ok(book) => { self.engine.set_book(book); None }
            Err(e)   => Some(format!("Fehler: {e}")),
        }
    }

    pub fn unload_book(&mut self) { self.engine.clear_book(); }

    pub fn book_info(&self) -> JsValue {
        let info = BookInfo {
            loaded:    self.engine.has_book(),
            positions: self.engine.book_len().unwrap_or(0),
        };
        serde_wasm_bindgen::to_value(&info).unwrap()
    }

    /// Top-N Buchzüge für die aktuelle Stellung, nach demselben Score sortiert
    /// wie die Engine-Buchauswahl (bester Zug = Index 0 = roter Pfeil).
    pub fn book_top_moves(&self, n: usize) -> JsValue {
        let mut entries = self.engine.book_entries(&self.board, self.engine.book_min_count());
        entries.truncate(n);
        let max_count = entries.iter().map(|e| e.2).max().unwrap_or(1);
        let moves: Vec<BookMove> = entries.iter().map(|(from, to, count)| {
            let mv  = format!("{}{}", sq_to_eng(*from), sq_to_eng(*to));
            let pct = ((*count as f64 / max_count as f64) * 100.0).round() as u8;
            BookMove { mv, count: *count, pct }
        }).collect();
        serde_wasm_bindgen::to_value(&moves).unwrap()
    }

    // ── PGN ───────────────────────────────────────────────────────────────────

    pub fn load_pgn(&mut self, pgn: &str) -> Option<String> {
        match GameRecord::replay(pgn) {
            Ok(board) => {
                self.history.clear();
                self.board = board;
                self.engine.new_game();
                None
            }
            Err(e) => Some(e),
        }
    }

    pub fn export_pgn(&self) -> String {
        "[Event \"Chaturaji\"]\n[Result \"*\"]\n\n*\n".to_string()
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.board = Board::default();
        self.engine.new_game();
    }
}

fn sq_to_eng(sq: u8) -> String {
    let file = (b'a' + (sq & 7)) as char;
    let rank = (sq >> 3) + 1;
    format!("{}{}", file, rank)
}
