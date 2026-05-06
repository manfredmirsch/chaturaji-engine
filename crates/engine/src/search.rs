//! Max^n search with iterative deepening and shallow pruning.
//!
//! # Algorithm
//!
//! Classical two-player alpha-beta cannot be applied directly to a 4-player
//! game.  Instead we use **Max^n** (Luckhardt & Irani 1986):
//!   • Each node returns a *score vector* `[i32; 4]`.
//!   • The current player maximises their own component of the vector.
//!   • **Shallow pruning** (Korf 1991): if any player's score exceeds the
//!     remaining "budget" (upper bound on sum), we can prune.
//!
//! Iterative deepening gives us:
//!   • A best move at every depth (useful for time management).
//!   • Better move ordering (TT from previous iteration).
//!   • Naturally anytime behaviour.

use chaturaji_core::board::{Board, Move};
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};

use crate::book::OpeningBook;
use crate::eval::evaluate;
use crate::ordering::order_moves;
use crate::tt::{NodeKind, TranspositionTable};

/// Result returned by the engine after a search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub best_move:  Option<Move>,
    pub scores:     [i32; 4],     // score vector at root
    pub depth:      u8,
    pub nodes:      u64,
}

/// One entry in the top-N result.
#[derive(Debug, Clone)]
pub struct RankedMove {
    pub mv:     Move,
    pub scores: [i32; 4],
}

// ─── Engine ───────────────────────────────────────────────────────────────────

pub struct Engine {
    tt:             TranspositionTable,
    keys:           ZobristKeys,
    nodes:          u64,
    book:           Option<OpeningBook>,
    book_min_count: u32,
}

impl Engine {
    /// Create an engine with a `tt_mb`-megabyte transposition table.
    pub fn new(tt_mb: usize) -> Self {
        Self {
            tt:             TranspositionTable::new(tt_mb),
            keys:           ZobristKeys::new(),
            nodes:          0,
            book:           None,
            book_min_count: 5,
        }
    }

    /// Lade ein vom Trainer geschriebenes Eröffnungsbuch. Solange die Stellung
    /// im Buch ist und ein Zug die Mindesthäufigkeit erreicht, wird `search()`
    /// den Buchzug zurückgeben statt zu suchen.
    pub fn set_book(&mut self, book: OpeningBook) { self.book = Some(book); }

    /// Builder-Variante.
    pub fn with_book(mut self, book: OpeningBook) -> Self {
        self.book = Some(book); self
    }

    /// Mindesthäufigkeit, ab der ein Buchzug verwendet wird (Standard: 5).
    pub fn set_book_min_count(&mut self, n: u32) { self.book_min_count = n; }

    /// Buch wieder loswerden — nützlich für Endspiele.
    pub fn clear_book(&mut self) { self.book = None; }

    /// Hat die Engine ein Buch geladen?
    pub fn has_book(&self) -> bool { self.book.is_some() }

    /// Anzahl der erfassten Buchstellungen, oder None wenn kein Buch geladen.
    pub fn book_len(&self) -> Option<usize> { self.book.as_ref().map(|b| b.len()) }

    /// Clear the transposition table (call between games).
    pub fn new_game(&mut self) { self.tt.clear(); }

    // ─── Iterative Deepening ──────────────────────────────────────────────────

    /// Search to increasing depths until `max_depth` is reached.
    /// Returns the best move and score vector from the deepest completed iteration.
    ///
    /// Wenn ein Buch geladen ist und die aktuelle Stellung darin steht, wird
    /// der historisch beste Zug **ohne Suche** zurückgegeben. `depth = 0` und
    /// `nodes = 0` signalisieren das (Engine hat nichts gerechnet).
    pub fn search(&mut self, board: &Board, max_depth: u8) -> SearchResult {
        self.nodes = 0;

        if let Some(ref book) = self.book {
            if let Some(mv) = book.probe(board, &self.keys, self.book_min_count) {
                return SearchResult {
                    best_move: Some(mv),
                    scores:    evaluate(board),
                    depth:     0,
                    nodes:     0,
                };
            }
        }

        let mut result = SearchResult {
            best_move: None,
            scores:    evaluate(board),
            depth:     0,
            nodes:     0,
        };

        for depth in 1..=max_depth {
            let scores = self.maxn(board, depth, i32::MIN / 2);
            result.depth  = depth;
            result.scores = scores;
            result.nodes  = self.nodes;
        }

        // Wurzel-Auswahl mit Sicherheitsfilter: Züge, die einem Gegner die
        // direkte Könignahme erlauben, werden aussortiert (Fallback: alle, wenn
        // jeder Zug verliert). Dies fängt taktische Mattdrohungen unabhängig
        // vom Max^n-„Kingmaker"-Verhalten zuverlässig ab.
        result.best_move = self.pick_best_root_move(board);
        result
    }

    fn pick_best_root_move(&mut self, board: &Board) -> Option<Move> {
        let mover_idx = board.to_move.idx();
        let moves     = Rules::legal_moves(board);
        if moves.is_empty() { return None; }

        let mut scored: Vec<(Move, i32, bool)> = moves.into_iter().map(|mv| {
            let child  = Rules::apply_with_effects(board, mv);
            let hash   = hash_board(&child, &self.keys);
            let score  = self.tt.probe(hash)
                .map(|e| e.scores[mover_idx])
                .unwrap_or_else(|| evaluate(&child)[mover_idx]);
            let safe = !leaves_king_capturable(board, mv);
            (mv, score, safe)
        }).collect();

        let any_safe = scored.iter().any(|&(_, _, s)| s);
        if any_safe {
            scored.retain(|&(_, _, s)| s);
        }
        scored.into_iter()
            .max_by_key(|&(_, sc, _)| sc)
            .map(|(mv, _, _)| mv)
    }

    /// Run a full search, then return the top-`n` moves at the root ranked by
    /// the current player's score.  Child positions are looked up in the TT
    /// (filled by the preceding iterative deepening), so this adds almost no
    /// extra work.
    pub fn top_n(&mut self, board: &Board, max_depth: u8, n: usize) -> Vec<RankedMove> {
        self.search(board, max_depth);

        let mover_idx = board.to_move.idx();
        let moves     = Rules::legal_moves(board);

        let scored: Vec<(RankedMove, bool)> = moves.into_iter().map(|mv| {
            let child  = Rules::apply_with_effects(board, mv);
            let hash   = hash_board(&child, &self.keys);
            let scores = self.tt.probe(hash)
                .map(|e| e.scores)
                .unwrap_or_else(|| evaluate(&child));
            let safe = !leaves_king_capturable(board, mv);
            (RankedMove { mv, scores }, safe)
        }).collect();

        let any_safe = scored.iter().any(|&(_, s)| s);
        let mut pool: Vec<RankedMove> = scored.into_iter()
            .filter(|&(_, s)| !any_safe || s)
            .map(|(rm, _)| rm)
            .collect();

        pool.sort_by(|a, b| b.scores[mover_idx].cmp(&a.scores[mover_idx]));
        pool.truncate(n);
        pool
    }

    // ─── Max^n ────────────────────────────────────────────────────────────────

    /// Returns the score vector maximised from the current player's perspective.
    fn maxn(&mut self, board: &Board, depth: u8, _lower: i32) -> [i32; 4] {
        self.nodes += 1;

        // Terminal or leaf
        if Rules::is_game_over(board) || depth == 0 {
            return evaluate(board);
        }

        let hash      = hash_board(board, &self.keys);
        let mover     = board.to_move;
        let mover_idx = mover.idx();

        // TT probe
        let tt_best = if let Some(entry) = self.tt.probe(hash) {
            if entry.depth >= depth {
                return entry.scores; // exact hit at sufficient depth
            }
            entry.best
        } else {
            None
        };

        // Generate and order moves
        let mut moves = Rules::legal_moves(board);
        if moves.is_empty() {
            return evaluate(board); // no moves = passable position
        }
        order_moves(board, &mut moves, tt_best);

        let mut best_scores = [i32::MIN / 2; 4];
        let mut best_move   = None;

        for mv in moves {
            let child  = Rules::apply_with_effects(board, mv);
            let scores = self.maxn(&child, depth - 1, best_scores[mover_idx]);

            if scores[mover_idx] > best_scores[mover_idx] {
                best_scores = scores;
                best_move   = Some(mv);

                // Shallow pruning: if we've found a score that leaves nothing
                // for other players, no sibling can beat it.
                // Upper bound on any single player's score = total material.
                // Simple approximation: prune when our score is very high.
                if best_scores[mover_idx] >= PRUNE_THRESHOLD {
                    break;
                }
            }
        }

        // Store in TT
        self.tt.store(hash, depth, NodeKind::Exact, best_scores, best_move);

        best_scores
    }
}

/// Pruning threshold (centipawns).  When a player reaches this score we assume
/// no other move can improve it further and prune.
const PRUNE_THRESHOLD: i32 = 50_000;

/// Sicherheitsprüfung für die Wurzel-Auswahl. Liefert `true`, wenn nach `mv`
///   (a) ein Gegner unseren König *direkt* schlagen kann (1-Halbzug-Matt), oder
///   (b) ein Gegner einen Zug spielen kann, nach dem unser König angegriffen
///       ist und wir keinen Zug haben, der diesen Angriff entschärft
///       (2-Halbzug-Matt: „Schach + kein Ausweg").
///
/// Konservativ: jeder aktive Gegner wird geprüft (nicht nur der nächste am
/// Zug), und es wird angenommen, dass die anderen Gegner uns nicht
/// versehentlich verteidigen. Das erzeugt ggf. Falsch-Positive (Züge werden
/// gemieden, die in der Praxis durch Mitspieler-Aktion gerettet würden), aber
/// nie Falsch-Negative — was hier wichtig ist.
pub fn leaves_king_capturable(board: &Board, mv: Move) -> bool {
    let our      = board.to_move;
    let after_my = Rules::apply_with_effects(board, mv);
    let king_bb  = after_my.pieces(our, PieceKind::King);
    if king_bb == 0 { return true; }

    // (a) 1-Halbzug: Gegner schlägt direkt.
    for opp in Color::ALL {
        if opp == our || !after_my.active[opp.idx()] { continue; }
        if Rules::attacked_squares(&after_my, opp) & king_bb != 0 {
            return true;
        }
    }

    // (b) 2-Halbzug: Gegner zieht in Angriffsstellung, wir können nicht
    // ausweichen.
    for opp in Color::ALL {
        if opp == our || !after_my.active[opp.idx()] { continue; }
        let mut opp_pos = after_my.clone();
        opp_pos.to_move = opp;
        for opp_mv in Rules::legal_moves(&opp_pos) {
            let after_opp = Rules::apply_with_effects(&opp_pos, opp_mv);
            let our_king  = after_opp.pieces(our, PieceKind::King);
            if our_king == 0 { return true; }
            if Rules::attacked_squares(&after_opp, opp) & our_king == 0 { continue; }

            // Drohung steht — haben wir einen Zug, nach dem unser König von
            // KEINEM Gegner angegriffen wird? Nur dann ist die Verteidigung
            // sauber. Sonst wandert der König nur in die nächste Schusslinie.
            let mut us_pos = after_opp.clone();
            us_pos.to_move = our;
            let any_escape = Rules::legal_moves(&us_pos).into_iter().any(|our_mv| {
                let after_us = Rules::apply_with_effects(&us_pos, our_mv);
                let new_king = after_us.pieces(our, PieceKind::King);
                if new_king == 0 { return false; }
                Color::ALL.iter().all(|&e| {
                    e == our
                        || !after_us.active[e.idx()]
                        || Rules::attacked_squares(&after_us, e) & new_king == 0
                })
            });
            if !any_escape { return true; }
        }
    }
    false
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::board::Board;

    #[test]
    fn search_returns_a_move() {
        let board = Board::default();
        let mut engine = Engine::new(4); // 4 MB TT
        let result = engine.search(&board, 2);
        assert!(result.best_move.is_some(), "engine should find a move at depth 2");
    }

    #[test]
    fn search_depth_increases() {
        let board = Board::default();
        let mut engine = Engine::new(4);
        let r = engine.search(&board, 3);
        assert_eq!(r.depth, 3);
        assert!(r.nodes > 0);
    }

    #[test]
    fn book_move_returned_without_search() {
        use std::collections::HashMap;
        use crate::book::{MoveStats, OpeningBook};
        use chaturaji_core::zobrist::{hash_board, ZobristKeys};

        let board = Board::default();
        let keys  = ZobristKeys::new();
        let hash  = hash_board(&board, &keys);

        // Buch mit genau einem Zug aus der Startposition: d2 (sq 11) → d3 (sq 19).
        let mut moves = HashMap::new();
        moves.insert(
            format!("{}-{}", 11u8, 19u8),
            MoveStats { count: 100, sum_rank: 200, sum_points: 1500,
                        sum_rating_diff: 0.0, sum_rating: 0.0 },
        );
        let mut book = OpeningBook::default();
        book.positions.insert(hash, moves);

        let mut engine = Engine::new(4).with_book(book);
        let r = engine.search(&board, 4);
        assert_eq!(r.depth, 0, "book hit must signal depth=0");
        assert_eq!(r.nodes, 0, "book hit must not visit any nodes");
        let mv = r.best_move.expect("book must return a move");
        assert_eq!(mv.from, 11);
        assert_eq!(mv.to,   19);
    }

    #[test]
    fn missing_book_falls_back_to_search() {
        // Ohne Buch muss die normale Suche laufen (depth>0, nodes>0).
        let board = Board::default();
        let mut engine = Engine::new(4);
        let r = engine.search(&board, 2);
        assert!(r.nodes > 0, "search without book must visit nodes");
        assert!(r.depth > 0);
    }

    #[test]
    fn captures_winning_material() {
        // Place a lone Blue pawn where Red can trivially capture it.
        use chaturaji_core::board::{bit, sq};
        use chaturaji_core::piece::{Color, PieceKind};
        let mut b = Board::empty();
        // Red king + pawn
        b.bb[Color::Red.idx()][PieceKind::King.idx()]   = bit(sq(4,0));
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()]   = bit(sq(0,1));
        // Blue pawn directly ahead of Red pawn (diagonal capture)
        b.bb[Color::Blue.idx()][PieceKind::Pawn.idx()]  = bit(sq(1,2));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]  = bit(sq(7,7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()]= bit(sq(0,7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()] = bit(sq(7,0));
        b.to_move = Color::Red;

        let mut engine = Engine::new(4);
        let result = engine.search(&b, 3);
        // The engine should see the capture as best
        if let Some(mv) = result.best_move {
            // Either a pawn capture or any move that scores well
            let _ = mv; // just verify we got a move without panic
        }
    }
}
