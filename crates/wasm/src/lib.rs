//! WASM bindings für den Chaturaji Engine + neuronales Netz.
//!
//! Das Netz kommt aus `chaturaji_engine::nnue_network`. Diese Kiste hatte bis
//! zum 2026-09-15 eine **eigene** Kopie von Netz und Merkmalsextraktor (56
//! Eingaben, eigene Serde-Form). Die Kopie und der Trainer liefen auseinander,
//! ohne dass es auffiel: die ausgelieferte `weights.json` gab in jeder Stellung
//! denselben Vektor zurück, weil die Aktivierungen davonliefen und tanh am
//! Anschlag stand. Für die Alpha-Beta-Suchen war das noch zu ertragen, für die
//! Baumsuche nicht — ohne Unterschiede zwischen den Blättern wählt PUCT nur
//! noch nach dem Zug-Prior. Deshalb jetzt ein Extraktor und ein Forward-Pass
//! für Trainer, Arena und Browser.

use wasm_bindgen::prelude::*;
use serde::Serialize;

use chaturaji_core::board::Board;
use chaturaji_core::notation::{move_to_str, parse_move, GameRecord};
use chaturaji_core::piece::Color;
use chaturaji_core::rules::Rules;
use chaturaji_engine::book::OpeningBook;
use chaturaji_engine::mcts::{Fpu, Mcts, MctsConfig, RootChoice, DEFAULT_C_PUCT};
use chaturaji_engine::policy::PolicyNet;
use chaturaji_engine::search::{Engine as SearchEngine, SearchAlgo};
use chaturaji_engine::nnue_network::NnueNetwork as Network;

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
    /// Monte-Carlo-Baumsuche mit gelerntem Zug-Prior.
    ///
    /// Gemessen am 2026-09-14 gegen Max^n Tiefe 4 / Beam 6, je 576 Partien mit
    /// demselben Netz auf beiden Seiten: **+0,54 Platzwert bei gleicher
    /// Rechenzeit** (1.600 Simulationen), **+0,18 bei einem Viertel** (400).
    /// Das ist der größte gemessene Stärkegewinn des Projekts.
    ///
    /// Braucht ein geladenes Netz — ohne Blattbewertung hat MCTS nichts, was
    /// es zurücktragen könnte. Die Einstiegspunkte fallen dann auf BRS zurück.
    Mcts,
}

/// Die beiden Alpha-Beta-Verfahren, ohne MCTS.
///
/// Ein eigener Typ, damit die Suchpfade darüber vollständig verzweigen können:
/// `Algorithm::Mcts` ist dort ausgeschlossen, und das soll der Compiler sehen
/// statt es einem toten `match`-Arm zu überlassen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AlphaBeta {
    Brs,
    Paranoid,
}

#[wasm_bindgen]
pub struct WasmEngine {
    board:   Board,
    engine:  SearchEngine,
    history: Vec<Board>,
    network: Option<Network>,
    algo:    Algorithm,
    /// Der Baum wird zwischen den Zügen behalten, damit nicht bei jedem Zug
    /// neu alloziert wird; `search` leert ihn selbst.
    mcts:    Mcts,
    /// Der Zug-Prior der Baumsuche.
    ///
    /// Das Policy-Netz statt der Linearform: gemessen +0,104 Platzwert über
    /// 2.304 Partien und vier Seeds, bei gleichem Bewertungsnetz und gleicher
    /// Suche. Bei den Budgets, die im Browser realistisch sind, folgen die
    /// Besuche überwiegend dem Prior — er ist also die Stellschraube, nicht
    /// eine unter vielen.
    model:   PolicyNet,
    iters:   u32,
    c_puct:  f32,
    root:    RootChoice,
    fpu:     Fpu,
    /// Baum zwischen den Zügen weiterverwenden. Braucht, dass bei **jeder**
    /// Brettänderung `advance` gerufen wird; die Stellungsprüfung in
    /// `Mcts::search` fängt ein Versäumnis ab, kostet dann aber den Baum.
    reuse:   bool,
    /// Gemessene Simulationen je Millisekunde, für `best_move_timed`.
    ///
    /// Der Startwert ist eine Schätzung; nach dem ersten Zug steht die
    /// gemessene Rate darin. Sie hängt vom Gerät ab und von der Stellung — ein
    /// leergeräumtes Brett hat weniger legale Züge und rechnet schneller —,
    /// deshalb wird sie fortlaufend nachgeführt statt einmal festgelegt.
    sims_per_ms: f32,
    /// Besuchsverteilung der letzten Baumsuche an der aktuellen Stellung.
    ///
    /// Die Oberfläche ruft nach `best_move` noch `top_moves` für die
    /// Kandidatenpfeile. Bei Alpha-Beta kostet das kaum etwas, weil die
    /// Transpositionstabelle noch warm ist; MCTS hat keine und würde den
    /// ganzen Baum ein zweites Mal bauen — also wird die Verteilung behalten
    /// und bei jeder Brettänderung verworfen.
    mcts_cache: Option<Vec<(chaturaji_core::board::Move, u32)>>,
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
            mcts:    Mcts::new(),
            model:   PolicyNet::default(),
            iters:   800,
            c_puct:  DEFAULT_C_PUCT,
            root:    RootChoice::Visits,
            // Beide gemessen: Wiederverwendung +0,064, FPU +0,067, zusammen
            // +0,112 Platzwert über vier Seeds — bei messbar null Kosten.
            fpu:     Fpu::ElternMinus(0.2),
            reuse:   true,
            sims_per_ms: 3.0,
            mcts_cache:  None,
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
            "mcts"     => { self.algo = Algorithm::Mcts;     true }
            _          => false,
        }
    }

    pub fn algorithm(&self) -> String {
        match self.algo {
            Algorithm::Brs      => "brs".to_string(),
            Algorithm::Paranoid => "paranoid".to_string(),
            Algorithm::Mcts     => "mcts".to_string(),
        }
    }

    /// Aufwand der Baumsuche: Simulationen je Zug.
    ///
    /// Anders als bei `depth` ist der Zusammenhang zur Wartezeit hier linear —
    /// doppelt so viele Simulationen kosten doppelt so lange. 1.600 entspricht
    /// ungefähr Max^n Tiefe 4 / Beam 6; der Standard 800 ist bewusst darunter,
    /// weil im Browser eine Sekunde Bedenkzeit unangenehmer wirkt als am
    /// Messrechner. Werte unter 2 ergeben keinen Baum.
    pub fn set_mcts_iterations(&mut self, iters: u32) -> bool {
        if iters < 2 { return false; }
        self.iters = iters;
        true
    }

    pub fn mcts_iterations(&self) -> u32 { self.iters }

    /// Erkundungsgewicht der PUCT-Formel. Größer heißt mehr Vertrauen in den
    /// Zug-Prior, kleiner mehr in die eigenen Simulationen. Der Standard 1,5
    /// passt zur Skala der Netzausgabe [−1, 1] und ist gemessen; Verstellen
    /// lohnt nur zum Experimentieren.
    pub fn set_c_puct(&mut self, c: f32) -> bool {
        if !c.is_finite() || c < 0.0 { return false; }
        self.c_puct = c;
        true
    }

    pub fn c_puct(&self) -> f32 { self.c_puct }


    // ── MCTS ──────────────────────────────────────────────────────────────────

    /// Eine Baumsuche für die aktuelle Stellung, oder `None`, wenn MCTS hier
    /// nicht zuständig ist: anderes Verfahren gewählt, kein Netz geladen, oder
    /// keine legalen Züge.
    ///
    /// Ohne Netz gibt es keine Blattbewertung — MCTS hätte nichts, was es
    /// zurücktragen könnte, und liefe auf eine reine Prior-Sortierung hinaus.
    /// Die Aufrufer fallen in dem Fall auf ihr bisheriges Verfahren zurück,
    /// statt kommentarlos schwächer zu spielen.
    fn mcts_search(&mut self) -> Option<chaturaji_engine::mcts::SearchResult> {
        if self.algo != Algorithm::Mcts { return None; }
        let net = self.network.as_ref()?;
        let cfg = MctsConfig {
            iterations: self.iters, c_puct: self.c_puct, root: self.root,
            fpu: self.fpu, reuse: self.reuse,
        };
        // Feldweise ausleihen: der Baum wird verändert, während das Netz
        // gelesen wird.
        let board = &self.board;
        let model = &self.model;
        let r = self.mcts.search(&|b: &Board| net.forward(b), model, board, &cfg);
        self.mcts_cache = r.as_ref().map(|r| r.visits.clone());
        r
    }

    /// Verwirft die zwischengespeicherte Besuchsverteilung.
    ///
    /// Muss von jeder Stelle gerufen werden, die `self.board` ändert — sonst
    /// zeigten die Kandidatenpfeile die Züge der vorigen Stellung.
    fn invalidate_mcts(&mut self) { self.mcts_cache = None; }

    /// Nach einem gespielten Zug: Baum auf das entsprechende Kind umsetzen.
    ///
    /// Muss nach **jedem** Halbzug geschehen, auch nach denen der Gegner — die
    /// nächste eigene Entscheidung liegt vier Halbzüge weiter, und nur wer alle
    /// dazwischen mitgeht, findet seinen Teilbaum wieder. `board_davor` ist die
    /// Stellung **vor** dem Zug.
    fn mcts_advance(&mut self, board_davor: &Board, mv: chaturaji_core::board::Move) {
        self.mcts_cache = None;
        if self.reuse { self.mcts.advance(board_davor, mv); } else { self.mcts.reset(); }
    }

    /// Bei allem, was nicht ein einzelner Zug vorwärts ist: Baum wegwerfen.
    fn mcts_reset(&mut self) {
        self.mcts_cache = None;
        self.mcts.reset();
    }

    /// Buchzug, falls die Stellung im geladenen Buch steht.
    ///
    /// Die Alpha-Beta-Suchen fragen das Buch selbst ab. MCTS tut das nicht,
    /// also muss es hier geschehen — sonst schaltete die Wahl von MCTS das
    /// Eröffnungsbuch stillschweigend ab.
    /// Verfahren für die Alpha-Beta-Pfade.
    ///
    /// Diese Pfade werden bei MCTS nur noch erreicht, wenn die Baumsuche nicht
    /// zuständig war — praktisch: kein Netz geladen. Dann ist BRS das bessere
    /// von beidem, also wird darauf zurückgefallen, statt den Zug zu verweigern.
    fn alpha_beta_algo(&self) -> AlphaBeta {
        match self.algo {
            Algorithm::Paranoid => AlphaBeta::Paranoid,
            // MCTS ohne Netz: BRS ist der bessere Rückfall.
            Algorithm::Brs | Algorithm::Mcts => AlphaBeta::Brs,
        }
    }

    fn book_move_str(&self) -> Option<String> {
        self.engine.book_move(&self.board).map(|mv| move_to_str(&mv))
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
                let davor = self.board.clone();
                self.history.push(davor.clone());
                self.board = Rules::apply_with_effects(&davor, mv);
                self.mcts_advance(&davor, mv);
                true
            }
            Err(_) => false,
        }
    }

    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.history.pop() {
            self.board = prev;
            // Zurücknehmen ist kein Schritt vorwärts — der Baum passt nicht mehr.
            self.mcts_reset();
            true
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
        self.mcts_reset();
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
        if self.algo == Algorithm::Mcts {
            if let Some(mv) = self.book_move_str() {
                let er = EngineResult {
                    best_move: Some(mv), scores: self.board.scores.as_array(),
                    net_values, depth: 0, nodes: 0,
                    used_network: self.network.is_some(),
                };
                return serde_wasm_bindgen::to_value(&er).unwrap();
            }
            if let Some(r) = self.mcts_search() {
                let er = EngineResult {
                    best_move:    Some(move_to_str(&r.best)),
                    scores:       self.board.scores.as_array(),
                    net_values,
                    depth:        r.depth,
                    nodes:        r.nodes,
                    used_network: true,
                };
                return serde_wasm_bindgen::to_value(&er).unwrap();
            }
        }
        // Split field borrows so the network closure and engine mutation can coexist.
        let algo    = self.alpha_beta_algo();
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let result = match algo {
            AlphaBeta::Brs      => engine.search_brs(board, depth, net_eval),
            AlphaBeta::Paranoid => engine.search_paranoid(board, depth, net_eval),
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

        // MCTS hat keine iterative Vertiefung, die man abbrechen könnte. Statt
        // das Budget zu ignorieren, wird es in Simulationen umgerechnet: die
        // Kosten je Simulation sind annähernd konstant, anders als bei einer
        // Tiefe. Gemessen wird dafür an der laufenden Suche — der erste Zug
        // einer Sitzung kalibriert sich an einem kurzen Probelauf.
        if self.algo == Algorithm::Mcts && self.network.is_some() {
            if let Some(mv) = self.book_move_str() {
                let er = EngineResult {
                    best_move: Some(mv), scores: self.board.scores.as_array(),
                    net_values, depth: 0, nodes: 0, used_network: true,
                };
                return serde_wasm_bindgen::to_value(&er).unwrap();
            }
            // Aus der gemessenen Rate die Zahl der Simulationen schätzen. Die
            // Untergrenze verhindert, dass ein winziges Budget einen Baum
            // ergibt, der nichts aussagt; die Obergrenze fängt eine kaputte
            // Rate ab.
            let geplant = (budget_ms as f32 * self.sims_per_ms)
                .clamp(32.0, 200_000.0) as u32;
            let merk = self.iters;
            self.iters = geplant;
            let start = js_sys::Date::now();
            let r = self.mcts_search();
            let gebraucht = (js_sys::Date::now() - start).max(0.001);
            self.iters = merk;
            // Rate nachführen, geglättet: ein einzelner Ausreißer (Tab im
            // Hintergrund, GC) soll die nächste Suche nicht verreißen.
            let gemessen = geplant as f32 / gebraucht as f32;
            if gemessen.is_finite() && gemessen > 0.0 {
                self.sims_per_ms = 0.7 * self.sims_per_ms + 0.3 * gemessen;
            }
            if let Some(r) = r {
                let er = EngineResult {
                    best_move:    Some(move_to_str(&r.best)),
                    scores:       self.board.scores.as_array(),
                    net_values,
                    depth:        r.depth,
                    nodes:        r.nodes,
                    used_network: true,
                };
                return serde_wasm_bindgen::to_value(&er).unwrap();
            }
        }

        // js_sys::Date::now() statt std::time::Instant: letzteres paniziert
        // unter wasm32-unknown-unknown.
        let deadline = js_sys::Date::now() + budget_ms;
        self.engine.set_stop_check(move || js_sys::Date::now() >= deadline);

        let algo    = self.alpha_beta_algo();
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let search_algo = match algo {
            AlphaBeta::Brs      => SearchAlgo::Brs,
            AlphaBeta::Paranoid => SearchAlgo::Paranoid,
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
        if self.algo == Algorithm::Mcts {
            // Die Verteilung der letzten Suche an dieser Stellung, sonst neu
            // suchen.
            let visits = match self.mcts_cache.take() {
                Some(v) => Some(v),
                None    => self.mcts_search().map(|r| r.visits),
            };
            if let Some(visits) = visits {
                // Bei MCTS ist die Besuchszahl das Maß, nicht ein Score: sie
                // ist das Ergebnis der Suche selbst. `score` trägt deshalb die
                // Besuche, `pct` ihren Anteil am meistbesuchten Zug.
                let max = visits.first().map(|(_, v)| *v).unwrap_or(1).max(1);
                let top: Vec<TopMove> = visits.iter().take(n as usize).map(|(mv, v)| TopMove {
                    mv:    move_to_str(mv),
                    score: *v as i32,
                    pct:   ((*v as f64 / max as f64) * 100.0).round() as u8,
                }).collect();
                self.mcts_cache = Some(visits);
                return serde_wasm_bindgen::to_value(&top).unwrap();
            }
        }
        let algo    = self.alpha_beta_algo();
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let ranked = match algo {
            AlphaBeta::Brs      => engine.top_n_brs(board, depth, n as usize, net_eval),
            AlphaBeta::Paranoid => engine.top_n_paranoid(board, depth, n as usize, net_eval),
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
        if self.algo == Algorithm::Mcts {
            let gewaehlt = self.engine.book_move(&self.board)
                .or_else(|| self.mcts_search().map(|r| r.best));
            if let Some(mv) = gewaehlt {
                let davor = self.board.clone();
                self.history.push(davor.clone());
                self.board = Rules::apply_with_effects(&davor, mv);
                self.mcts_advance(&davor, mv);
                return true;
            }
            // Kein Netz geladen: unten weiter mit BRS statt gar nicht zu ziehen.
        }
        let algo    = self.alpha_beta_algo();
        let engine  = &mut self.engine;
        let network = &self.network;
        let board   = &self.board;
        let f;
        let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match network.as_ref() {
            Some(net) => { f = |b: &Board| net.forward(b); Some(&f) }
            None      => None,
        };
        let result = match algo {
            AlphaBeta::Brs      => engine.search_brs(board, depth, net_eval),
            AlphaBeta::Paranoid => engine.search_paranoid(board, depth, net_eval),
        };
        if let Some(mv) = result.best_move {
            let davor = self.board.clone();
            self.history.push(davor.clone());
            self.board = Rules::apply_with_effects(&davor, mv);
            self.mcts_advance(&davor, mv);
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
            Ok(mut net) => {
                if let Err(e) = net.validate() {
                    return Some(format!("Netzwerk-Architektur passt nicht: {e}"));
                }
                // Checkpoints von vor den dichten Merkmalen haben kürzere
                // L1-Zeilen; die fehlenden Spalten mit Null aufzufüllen ist
                // genau das, was der Trainer beim Weiterlernen auch tut.
                net.ensure_input_size();
                self.network = Some(net);
                // Anderes Netz heißt andere Bewertungen — der alte Baum ist wertlos.
                self.mcts_reset();
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

    pub fn unload_network(&mut self) { self.network = None; self.mcts_reset(); }

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
                self.mcts_reset();
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
        self.mcts_reset();
    }
}

fn sq_to_eng(sq: u8) -> String {
    let file = (b'a' + (sq & 7)) as char;
    let rank = (sq >> 3) + 1;
    format!("{}{}", file, rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lädt das Netz, das die Seite tatsächlich ausliefert.
    ///
    /// `www/weights.json` steht in `.gitignore` — 4 MB Gewichte gehören nicht
    /// in den Verlauf, sie liegen im Release `nnue-state`. In einem frischen
    /// Klon fehlt die Datei deshalb, und dann prüfen diese Tests eben nichts,
    /// statt fehlzuschlagen. Das ist die ehrliche Variante: ein roter Test, der
    /// nur eine fehlende Datei meldet, wird nach dem zweiten Mal ignoriert und
    /// deckt dann auch echte Fehler nicht mehr auf.
    fn ausgeliefertes_netz() -> Option<Network> {
        let pfad = concat!(env!("CARGO_MANIFEST_DIR"), "/www/weights.json");
        let json = match std::fs::read_to_string(pfad) {
            Ok(j) => j,
            Err(_) => {
                eprintln!("übersprungen: {pfad} fehlt (steht in .gitignore)");
                return None;
            }
        };
        let mut netz: Network = serde_json::from_str(&json)
            .expect("www/weights.json muss ladbar sein");
        netz.validate().expect("Architektur muss zur einkompilierten passen");
        netz.ensure_input_size();
        Some(netz)
    }

    /// Eine Bewertungsfunktion, die überall dasselbe liefert, ist wertlos — und
    /// für die Baumsuche schlimmer als wertlos: ohne Unterschiede zwischen den
    /// Blättern ist Q überall gleich, PUCT wählt dann nur noch nach dem
    /// Zug-Prior und sucht gar nicht mehr.
    ///
    /// Am 2026-09-15 war genau das der Fall, unbemerkt über Monate: die alte,
    /// in dieser Kiste gepflegte Netzkopie gab in jeder Stellung
    /// [−1, −1, +1, −1] zurück. Merkmale in [0, 1] wie vorgesehen, aber a2 max
    /// 107,7 und vor tanh [−31,8, −45,6, +49,6, −79,8] — die Aktivierungen
    /// liefen davon, tanh stand am Anschlag. Diese Prüfung hätte das sofort
    /// gezeigt.
    #[test]
    fn das_ausgelieferte_netz_unterscheidet_stellungen() {
        let Some(netz) = ausgeliefertes_netz() else { return };
        let mut brett = Board::default();
        let mut gesehen = vec![netz.forward(&brett)];
        for _ in 0..6 {
            let zug = Rules::legal_moves(&brett)[0];
            brett = Rules::apply_with_effects(&brett, zug);
            gesehen.push(netz.forward(&brett));
        }
        let erste = gesehen[0];
        let unterschiedlich = gesehen.iter()
            .any(|v| v.iter().zip(&erste).any(|(a, b)| (a - b).abs() > 1e-4));
        assert!(unterschiedlich, "Netz liefert in allen Stellungen dasselbe: {erste:?}");
    }

    /// Eine gesättigte Ausgabe trägt keine Information: tanh ist dort flach,
    /// verschiedene Stellungen fallen auf denselben Wert zusammen.
    #[test]
    fn das_ausgelieferte_netz_ist_nicht_gesaettigt() {
        let Some(netz) = ausgeliefertes_netz() else { return };
        let v = netz.forward(&Board::default());
        let am_anschlag = v.iter().filter(|x| x.abs() > 0.999).count();
        assert!(am_anschlag == 0,
            "Ausgaben am Anschlag: {v:?} — tanh ist gesättigt");
    }

    /// Die Summe der vier Platzwerte ist 0. Ein Netz, das darauf trainiert ist,
    /// muss in der symmetrischen Startstellung nahe bei null liegen; läuft die
    /// Summe weg, ist die Zielkodierung nicht die, für die die Suche rechnet.
    #[test]
    fn die_startstellung_ist_ungefaehr_ausgeglichen() {
        let Some(netz) = ausgeliefertes_netz() else { return };
        let v = netz.forward(&Board::default());
        let summe: f32 = v.iter().sum();
        assert!(summe.abs() < 0.5, "Startstellung nicht ausgeglichen: {v:?}, Summe {summe:.3}");
    }
}
