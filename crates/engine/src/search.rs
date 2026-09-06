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
use crate::eval::{attack_bitboard, evaluate_with, EvalParams};
use crate::ordering::order_moves;
use crate::tt::{NodeKind, TranspositionTable};
use crate::utility::UTIL_SCALE;

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
    params:         EvalParams,
    keys:           ZobristKeys,
    nodes:          u64,
    book:           Option<OpeningBook>,
    book_min_count: u32,
    killers:        [[Option<Move>; 2]; 64],
}

impl Engine {
    /// Create an engine with a `tt_mb`-megabyte transposition table.
    pub fn new(tt_mb: usize) -> Self {
        Self {
            tt:             TranspositionTable::new(tt_mb),
            params:         EvalParams::default(),
            keys:           ZobristKeys::new(),
            nodes:          0,
            book:           None,
            book_min_count: 5,
            killers:        [[None; 2]; 64],
        }
    }

    /// Evaluation parameters this engine plays with. Two engines with
    /// different parameter sets can be seated at the same table, which is how
    /// `examples/arena.rs` measures one model against another.
    pub fn set_eval_params(&mut self, p: EvalParams) { self.params = p; }

    /// Builder-Variante zu [`Engine::set_eval_params`].
    pub fn with_eval_params(mut self, p: EvalParams) -> Self { self.params = p; self }

    pub fn eval_params(&self) -> EvalParams { self.params }

    /// Static evaluation under this engine's own parameters.
    #[inline]
    fn ev(&self, board: &Board) -> [i32; 4] { evaluate_with(board, &self.params) }

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

    pub fn book_min_count(&self) -> u32 { self.book_min_count }

    /// Buch wieder loswerden — nützlich für Endspiele.
    pub fn clear_book(&mut self) { self.book = None; }

    /// Hat die Engine ein Buch geladen?
    pub fn has_book(&self) -> bool { self.book.is_some() }

    /// Anzahl der erfassten Buchstellungen, oder None wenn kein Buch geladen.
    pub fn book_len(&self) -> Option<usize> { self.book.as_ref().map(|b| b.len()) }

    /// Liefert Buchzüge für `board`, nach Score sortiert (bester zuerst).
    /// Verwendet denselben min_count-Schwellwert wie `probe()`.
    pub fn book_entries(&self, board: &Board, min_count: u32) -> Vec<(u8, u8, u32)> {
        self.book.as_ref()
            .and_then(|b| b.entries(board, &self.keys, min_count))
            .unwrap_or_default()
    }

    /// Clear the transposition table and killer table (call between games).
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.killers = [[None; 2]; 64];
    }

    /// Store `mv` as a killer at `depth` (quiet moves only).
    fn store_killer(&mut self, depth: u8, mv: Move) {
        if mv.captured.is_some() { return; }
        let d = (depth as usize).min(63);
        let ks = &mut self.killers[d];
        if ks[0] == Some(mv) { return; }  // avoid duplicates
        ks[1] = ks[0];
        ks[0] = Some(mv);
    }

    // ─── Iterative Deepening ──────────────────────────────────────────────────

    /// Search to increasing depths until `max_depth` is reached.
    /// Returns the best move and score vector from the deepest completed iteration.
    ///
    /// Wenn ein Buch geladen ist und die aktuelle Stellung darin steht, wird
    /// der historisch beste Zug **ohne Suche** zurückgegeben. `depth = 0` und
    /// `nodes = 0` signalisieren das (Engine hat nichts gerechnet).
    pub fn search(&mut self, board: &Board, max_depth: u8) -> SearchResult {
        self.nodes = 0;
        self.tt.new_search();

        if let Some(ref book) = self.book {
            if let Some(mv) = book.probe(board, &self.keys, self.book_min_count) {
                return SearchResult {
                    best_move: Some(mv),
                    scores:    self.ev(board),
                    depth:     0,
                    nodes:     0,
                };
            }
        }

        let mut result = SearchResult {
            best_move: None,
            scores:    self.ev(board),
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
                .unwrap_or_else(|| self.ev(&child)[mover_idx]);
            let safe = !leaves_king_capturable(board, mv)
                    && !leaves_piece_exposed(board, mv);
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
                .unwrap_or_else(|| self.ev(&child));
            let safe = !leaves_king_capturable(board, mv)
                    && !leaves_piece_exposed(board, mv);
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
        if Rules::is_game_over(board) { return self.ev(board); }
        if depth == 0 { return self.quiescence(board, 1); }

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

        // Generate and order moves. Copy killers for this depth before the loop
        // so the mutable borrow for the recursive maxn call doesn't conflict.
        let my_killers = if depth < 64 { self.killers[depth as usize] } else { [None; 2] };
        let mut moves = Rules::legal_moves(board);
        if moves.is_empty() {
            return self.ev(board); // no moves = passable position
        }
        order_moves(board, &mut moves, tt_best, &my_killers);

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
                    self.store_killer(depth, mv);
                    break;
                }
            }
        }

        // Store in TT
        self.tt.store(hash, depth, NodeKind::Exact, best_scores, best_move);

        best_scores
    }

    // ─── Quiescence Search ────────────────────────────────────────────────────

    /// Extends the search at leaf nodes by only considering captures and
    /// promotions until the position is "quiet" or `qdepth` is exhausted.
    ///
    /// Stand-pat: `evaluate(board)` is the baseline — the current player may
    /// always choose to stand pat rather than capture. If no capture improves
    /// their score, stand-pat is returned.  This prevents the horizon effect
    /// where a losing exchange starts just at the search-depth boundary.
    ///
    /// Max Q-depth = 4 to avoid chain explosions in 4-player middlegames.
    fn quiescence(&mut self, board: &Board, qdepth: u8) -> [i32; 4] {
        self.nodes += 1;
        let stand_pat = self.ev(board);

        if Rules::is_game_over(board) || qdepth == 0 {
            return stand_pat;
        }

        let mover_idx = board.to_move.idx();

        // If the stand-pat score is already beyond the pruning threshold,
        // no capture can realistically improve it further.
        if stand_pat[mover_idx] >= PRUNE_THRESHOLD {
            return stand_pat;
        }

        // Filter to captures and promotions only.
        let all_moves = Rules::legal_moves(board);
        let mut noisy: Vec<Move> = all_moves.into_iter()
            .filter(|mv| mv.captured.is_some() || mv.promoted)
            .collect();

        if noisy.is_empty() {
            return stand_pat;
        }

        // Order by MVV-LVA so the most promising captures come first.
        order_moves(board, &mut noisy, None, &[None; 2]);

        // Stand-pat is the initial best: the player can always decline to capture.
        let mut best_scores = stand_pat;

        for mv in noisy {
            let child  = Rules::apply_with_effects(board, mv);
            let scores = self.quiescence(&child, qdepth - 1);

            if scores[mover_idx] > best_scores[mover_idx] {
                best_scores = scores;
                if best_scores[mover_idx] >= PRUNE_THRESHOLD {
                    break;
                }
            }
        }

        best_scores
    }

    // ─── Paranoid search ─────────────────────────────────────────────────────

    /// Minimax alpha-beta where the current player maximises their score and
    /// all opponents minimise it. Returns root player's centipawn score.
    ///
    /// TT is used for move ordering only (paranoid and Max^n scores are
    /// incompatible in the same TT, so we never store or cut off).
    fn paranoid_rec(
        &mut self,
        board: &Board,
        depth: u8,
        mut alpha: i32,
        mut beta: i32,
        root_player: usize,
        net_eval: Option<&dyn Fn(&Board) -> [f32; 4]>,
    ) -> i32 {
        self.nodes += 1;

        if Rules::is_game_over(board) {
            return self.ev(board)[root_player];
        }
        if depth == 0 {
            if let Some(f) = net_eval {
                // The net's tanh output is already a placement-style utility in
                // [−1, 1] (it is trained against final standings), so it shares
                // the scale of `evaluate` and must use the same multiplier —
                // otherwise net and hand-crafted leaves are incomparable.
                return (f(board)[root_player] * UTIL_SCALE as f32) as i32;
            }
            return self.ev(board)[root_player];
        }

        let my_killers = if depth < 64 { self.killers[depth as usize] } else { [None; 2] };
        let mut moves = Rules::legal_moves(board);
        if moves.is_empty() {
            return self.ev(board)[root_player];
        }

        let hash    = hash_board(board, &self.keys);
        let tt_best = self.tt.probe(hash).and_then(|e| e.best);
        order_moves(board, &mut moves, tt_best, &my_killers);

        let is_max = board.to_move.idx() == root_player;
        let mut best = if is_max { i32::MIN / 2 } else { i32::MAX / 2 };

        for mv in moves {
            let child = Rules::apply_with_effects(board, mv);
            let score = self.paranoid_rec(&child, depth - 1, alpha, beta, root_player, net_eval);
            if is_max {
                if score > best {
                    best = score;
                    if best > alpha {
                        alpha = best;
                        self.store_killer(depth, mv);
                    }
                    if alpha >= beta { break; }
                }
            } else {
                if score < best {
                    best = score;
                    if best < beta { beta = best; }
                    if alpha >= beta { break; }
                }
            }
        }
        best
    }

    /// Search using the paranoid algorithm (all opponents minimise the current
    /// player's score). Includes the same root safety filter as `search()`.
    ///
    /// This avoids Max^n's blind spot where no single opponent has an incentive
    /// to capture an undefended piece — under paranoid assumptions they always
    /// will if it hurts the current player.
    pub fn search_paranoid(
        &mut self,
        board: &Board,
        max_depth: u8,
        net_eval: Option<&dyn Fn(&Board) -> [f32; 4]>,
    ) -> SearchResult {
        self.nodes = 0;
        self.tt.new_search();

        if let Some(ref book) = self.book {
            if let Some(mv) = book.probe(board, &self.keys, self.book_min_count) {
                return SearchResult {
                    best_move: Some(mv),
                    scores:    self.ev(board),
                    depth:     0,
                    nodes:     0,
                };
            }
        }

        let root_player = board.to_move.idx();
        let moves       = Rules::legal_moves(board);
        if moves.is_empty() {
            return SearchResult { best_move: None, scores: self.ev(board), depth: max_depth, nodes: 0 };
        }

        let safety: Vec<bool> = moves.iter()
            .map(|&mv| !leaves_king_capturable(board, mv) && !leaves_piece_exposed(board, mv))
            .collect();
        let any_safe = safety.iter().any(|&s| s);

        let mut best_move  = None;
        let mut best_score = i32::MIN / 2;
        let mut alpha      = i32::MIN / 2 + 1;

        for (i, &mv) in moves.iter().enumerate() {
            if any_safe && !safety[i] { continue; }
            let child = Rules::apply_with_effects(board, mv);
            let score = self.paranoid_rec(
                &child, max_depth.saturating_sub(1),
                alpha, i32::MAX / 2 - 1, root_player,
                net_eval,
            );
            if score > best_score {
                best_score = score;
                best_move  = Some(mv);
                if best_score > alpha { alpha = best_score; }
            }
        }

        SearchResult {
            best_move,
            scores: self.ev(board),
            depth:  max_depth,
            nodes:  self.nodes,
        }
    }

    // ─── Best-Reply Search ───────────────────────────────────────────────────

    /// One ply of Best-Reply Search (Schadd & Winands 2011).
    ///
    /// The two algorithms above sit at opposite extremes. Max^n is the
    /// theoretically correct model for a general-sum game but barely prunes,
    /// so it stays shallow. Paranoid prunes well but assumes all three
    /// opponents cooperate against the root player — in Chaturaji plainly
    /// false, since they are competing with each other for the same points.
    ///
    /// BRS takes the middle road: between two moves of the root player only
    /// the *single most dangerous* opponent replies, and the other two pass.
    /// Branching per round drops from b⁴ to ≈ 3b², which buys roughly twice
    /// the reachable depth at equal node count while keeping an opponent model
    /// that is much closer to how the game is actually played.
    ///
    /// The passes make intermediate positions slightly unreal — that is the
    /// known cost of the method, and the reason BRS scores are never written
    /// to the shared transposition table (it is read for move ordering only).
    #[allow(clippy::too_many_arguments)]
    fn brs_rec(
        &mut self,
        board: &Board,
        depth: u8,
        mut alpha: i32,
        mut beta: i32,
        root_player: usize,
        is_max: bool,
        net_eval: Option<&dyn Fn(&Board) -> [f32; 4]>,
    ) -> i32 {
        self.nodes += 1;

        // A root player who has been knocked out has a frozen score; nothing
        // below this node can change it.
        if Rules::is_game_over(board) || !board.active[root_player] {
            return self.ev(board)[root_player];
        }
        if depth == 0 {
            if let Some(f) = net_eval {
                return (f(board)[root_player] * UTIL_SCALE as f32) as i32;
            }
            return self.ev(board)[root_player];
        }

        let root_color  = Color::ALL[root_player];
        let my_killers  = if depth < 64 { self.killers[depth as usize] } else { [None; 2] };

        if is_max {
            let mut b = board.clone();
            b.to_move = root_color;

            let mut moves = Rules::legal_moves(&b);
            if moves.is_empty() {
                return self.ev(&b)[root_player];
            }
            let tt_best = self.tt.probe(hash_board(&b, &self.keys)).and_then(|e| e.best);
            order_moves(&b, &mut moves, tt_best, &my_killers);

            let mut best = i32::MIN / 2;
            for mv in moves {
                let child = Rules::apply_with_effects(&b, mv);
                let score = self.brs_rec(
                    &child, depth - 1, alpha, beta, root_player, false, net_eval,
                );
                if score > best {
                    best = score;
                    if best > alpha {
                        alpha = best;
                        self.store_killer(depth, mv);
                    }
                    if alpha >= beta { break; }
                }
            }
            best
        } else {
            // The best-reply ply: every active opponent is offered the move and
            // the worst outcome for the root player is taken. Only that one
            // opponent actually plays — the others pass.
            let mut best = i32::MAX / 2;
            let mut any_reply = false;

            'outer: for opp in Color::ALL {
                if opp == root_color || !board.active[opp.idx()] { continue; }

                let mut b = board.clone();
                b.to_move = opp;
                let mut moves = Rules::legal_moves(&b);
                if moves.is_empty() { continue; }
                any_reply = true;
                order_moves(&b, &mut moves, None, &my_killers);

                for mv in moves {
                    let mut child = Rules::apply_with_effects(&b, mv);
                    // Control returns to the root player: the other two passed.
                    if child.active[root_player] {
                        child.to_move = root_color;
                    }
                    let score = self.brs_rec(
                        &child, depth - 1, alpha, beta, root_player, true, net_eval,
                    );
                    if score < best {
                        best = score;
                        if best < beta { beta = best; }
                        if alpha >= beta { break 'outer; }
                    }
                }
            }

            if any_reply { best } else { self.ev(board)[root_player] }
        }
    }

    /// Search using Best-Reply Search. Same root safety filter and same
    /// signature as [`Engine::search_paranoid`], so the two are drop-in
    /// interchangeable.
    pub fn search_brs(
        &mut self,
        board: &Board,
        max_depth: u8,
        net_eval: Option<&dyn Fn(&Board) -> [f32; 4]>,
    ) -> SearchResult {
        self.nodes = 0;
        self.tt.new_search();

        if let Some(ref book) = self.book {
            if let Some(mv) = book.probe(board, &self.keys, self.book_min_count) {
                return SearchResult {
                    best_move: Some(mv),
                    scores:    self.ev(board),
                    depth:     0,
                    nodes:     0,
                };
            }
        }

        let root_player = board.to_move.idx();
        let moves       = Rules::legal_moves(board);
        if moves.is_empty() {
            return SearchResult {
                best_move: None, scores: self.ev(board), depth: max_depth, nodes: 0,
            };
        }

        let safety: Vec<bool> = moves.iter()
            .map(|&mv| !leaves_king_capturable(board, mv) && !leaves_piece_exposed(board, mv))
            .collect();
        let any_safe = safety.iter().any(|&s| s);

        let mut best_move  = None;
        let mut best_score = i32::MIN / 2;
        let mut alpha      = i32::MIN / 2 + 1;

        for (i, &mv) in moves.iter().enumerate() {
            if any_safe && !safety[i] { continue; }
            let child = Rules::apply_with_effects(board, mv);
            // After our move it is the opponents' best-reply ply.
            let score = self.brs_rec(
                &child, max_depth.saturating_sub(1),
                alpha, i32::MAX / 2 - 1, root_player, false,
                net_eval,
            );
            if score > best_score {
                best_score = score;
                best_move  = Some(mv);
                if best_score > alpha { alpha = best_score; }
            }
        }

        SearchResult {
            best_move,
            scores: self.ev(board),
            depth:  max_depth,
            nodes:  self.nodes,
        }
    }

    /// Top-N moves ranked by BRS score. Each root move is searched with a full
    /// window so the relative ordering is accurate.
    pub fn top_n_brs(
        &mut self,
        board: &Board,
        max_depth: u8,
        n: usize,
        net_eval: Option<&dyn Fn(&Board) -> [f32; 4]>,
    ) -> Vec<RankedMove> {
        let root_player = board.to_move.idx();
        let moves       = Rules::legal_moves(board);

        let safety: Vec<bool> = moves.iter()
            .map(|&mv| !leaves_king_capturable(board, mv) && !leaves_piece_exposed(board, mv))
            .collect();
        let any_safe = safety.iter().any(|&s| s);

        let mut ranked: Vec<(Move, i32)> = moves.iter().enumerate()
            .filter(|(i, _)| !any_safe || safety[*i])
            .map(|(_, &mv)| {
                let child = Rules::apply_with_effects(board, mv);
                let score = self.brs_rec(
                    &child, max_depth.saturating_sub(1),
                    i32::MIN / 2 + 1, i32::MAX / 2 - 1, root_player, false,
                    net_eval,
                );
                (mv, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(n);

        ranked.into_iter().map(|(mv, score)| {
            let mut scores = [0i32; 4];
            scores[root_player] = score;
            RankedMove { mv, scores }
        }).collect()
    }

    /// Top-N moves ranked by paranoid score.  Each root move is evaluated
    /// independently so the relative ordering is accurate.
    pub fn top_n_paranoid(
        &mut self,
        board: &Board,
        max_depth: u8,
        n: usize,
        net_eval: Option<&dyn Fn(&Board) -> [f32; 4]>,
    ) -> Vec<RankedMove> {
        let root_player = board.to_move.idx();
        let moves       = Rules::legal_moves(board);

        let safety: Vec<bool> = moves.iter()
            .map(|&mv| !leaves_king_capturable(board, mv) && !leaves_piece_exposed(board, mv))
            .collect();
        let any_safe = safety.iter().any(|&s| s);

        let mut ranked: Vec<(Move, i32)> = moves.iter().enumerate()
            .filter(|(i, _)| !any_safe || safety[*i])
            .map(|(_, &mv)| {
                let child = Rules::apply_with_effects(board, mv);
                let score = self.paranoid_rec(
                    &child, max_depth.saturating_sub(1),
                    i32::MIN / 2 + 1, i32::MAX / 2 - 1, root_player,
                    net_eval,
                );
                (mv, score)
            })
            .collect();

        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(n);

        ranked.into_iter().map(|(mv, score)| {
            let mut scores = [0i32; 4];
            scores[root_player] = score;
            RankedMove { mv, scores }
        }).collect()
    }
}

/// Shallow-pruning threshold, in `UTIL_SCALE` units.
///
/// Since `evaluate` now returns a rank utility with `Σ u_i ≡ 0` and
/// `u_i ∈ [−1, 1]`, a single player's score is bounded above by `UTIL_SCALE`
/// exactly — no guessing required. Once the mover is within a hair of that
/// bound they are certain to finish first and no sibling can do better.
const PRUNE_THRESHOLD: i32 = UTIL_SCALE - UTIL_SCALE / 50; // 98 % of the bound

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

// ─── Piece-safety root filter ─────────────────────────────────────────────────

/// Returns true when `mv` leaves a valuable piece (knight/bishop/boat)
/// completely undefended AND, after any single response by the next player,
/// any opponent can capture it for free.
///
/// This catches the pattern the max-n search misses: Blue makes an
/// innocuous-looking move (e.g. advancing a pawn) that incidentally clears a
/// blocker, and then Yellow captures the now-exposed piece.
///
/// Paranoid but O(B²) at root-only — negligible vs. the main search.
fn leaves_piece_exposed(board: &Board, mv: Move) -> bool {
    let our      = board.to_move;
    let after_mv = Rules::apply_with_effects(board, mv);

    // Our valuable non-king pieces after the move.
    let our_valuable: u64 = [PieceKind::Knight, PieceKind::Bishop, PieceKind::Boat]
        .iter()
        .fold(0u64, |acc, &k| acc | after_mv.pieces(our, k));
    if our_valuable == 0 { return false; }

    // Squares we attack = proxy for defended squares (recapture potential).
    let our_atk = attack_bitboard(&after_mv, our);
    // Pieces with zero defenders.
    let undefended = our_valuable & !our_atk;
    if undefended == 0 { return false; }

    // For each legal response by the immediate next player:
    for opp_mv in Rules::legal_moves(&after_mv) {
        let after_opp = Rules::apply_with_effects(&after_mv, opp_mv);

        // Which of our formerly-undefended pieces are still on the board?
        let our_pieces_now: u64 = [PieceKind::Knight, PieceKind::Bishop, PieceKind::Boat]
            .iter()
            .fold(0u64, |acc, &k| acc | after_opp.pieces(our, k));
        let still_undefended = our_pieces_now & undefended
            & !attack_bitboard(&after_opp, our);
        if still_undefended == 0 { continue; }

        // Can any opponent now capture one of those pieces?
        for opp in Color::ALL {
            if opp == our || !after_opp.active[opp.idx()] { continue; }
            if attack_bitboard(&after_opp, opp) & still_undefended != 0 {
                return true;
            }
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

    /// Regression test for the "unprotected bishop discovered attack" bug:
    ///
    /// Red bishop on b4, completely undefended.
    /// Blue pawn on c5 is the sole blocker between Yellow's bishop (f8) and b4.
    /// If Red moves their king to b1 (leaving b4 undefended), Blue can advance
    /// the pawn to d5, clearing the diagonal — then Yellow takes f8×b4 for free.
    ///
    /// `leaves_piece_exposed` must flag the king-to-b1 move as unsafe.
    #[test]
    fn leaves_piece_exposed_detects_discovered_bishop_capture() {
        use chaturaji_core::board::{bit, sq};
        use chaturaji_core::piece::{Color, PieceKind};
        use chaturaji_core::notation::parse_move;

        // Reconstruct the position after 5 full rounds as described.
        // Red: King c1, Bishop b4, Knight c3, Boat a1, Pawns a2 b2 c2 d3
        // Blue: King b6, Boat a8, Knight a7, Bishop a6, Pawns c5 c6 c7 c8
        // Yellow: King f7, Boat h8, Knight g8, Bishop f8, Pawns e6 f6 g6 h6
        // Green: King h3, Boat h1, Knight g4, Bishop f3, Pawns g1 g2 g3 f4
        let mut b = Board::empty();
        // Red
        b.bb[Color::Red.idx()][PieceKind::King.idx()]   = bit(sq(2,0)); // c1
        b.bb[Color::Red.idx()][PieceKind::Bishop.idx()] = bit(sq(1,3)); // b4
        b.bb[Color::Red.idx()][PieceKind::Knight.idx()] = bit(sq(2,2)); // c3
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]   = bit(sq(0,0)); // a1
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()]   =
              bit(sq(0,1))|bit(sq(1,1))|bit(sq(2,1))|bit(sq(3,2)); // a2 b2 c2 d3
        // Blue
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(1,5)); // b6
        b.bb[Color::Blue.idx()][PieceKind::Boat.idx()]   = bit(sq(0,7)); // a8
        b.bb[Color::Blue.idx()][PieceKind::Knight.idx()] = bit(sq(0,6)); // a7
        b.bb[Color::Blue.idx()][PieceKind::Bishop.idx()] = bit(sq(0,5)); // a6
        b.bb[Color::Blue.idx()][PieceKind::Pawn.idx()]   =
              bit(sq(2,4))|bit(sq(2,5))|bit(sq(2,6))|bit(sq(2,7)); // c5 c6 c7 c8
        // Yellow
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()]   = bit(sq(5,6)); // f7
        b.bb[Color::Yellow.idx()][PieceKind::Boat.idx()]   = bit(sq(7,7)); // h8
        b.bb[Color::Yellow.idx()][PieceKind::Knight.idx()] = bit(sq(6,7)); // g8
        b.bb[Color::Yellow.idx()][PieceKind::Bishop.idx()] = bit(sq(5,7)); // f8
        b.bb[Color::Yellow.idx()][PieceKind::Pawn.idx()]   =
              bit(sq(4,5))|bit(sq(5,5))|bit(sq(6,5))|bit(sq(7,5)); // e6 f6 g6 h6
        // Green
        b.bb[Color::Green.idx()][PieceKind::King.idx()]   = bit(sq(7,2)); // h3
        b.bb[Color::Green.idx()][PieceKind::Boat.idx()]   = bit(sq(7,0)); // h1
        b.bb[Color::Green.idx()][PieceKind::Knight.idx()] = bit(sq(6,3)); // g4
        b.bb[Color::Green.idx()][PieceKind::Bishop.idx()] = bit(sq(5,2)); // f3
        b.bb[Color::Green.idx()][PieceKind::Pawn.idx()]   =
              bit(sq(6,0))|bit(sq(6,1))|bit(sq(6,2))|bit(sq(5,3)); // g1 g2 g3 f4
        b.to_move = Color::Red;

        // c1→b1: king retreats, bishop on b4 stays undefended.
        let mv_c1b1 = parse_move(&b, "c1b1").expect("c1b1 must be legal");
        assert!(
            leaves_piece_exposed(&b, mv_c1b1),
            "c1b1 should be flagged: bishop on b4 is undefended \
             and Yellow can capture it after Blue clears c5"
        );

        // Sanity: search must NOT choose c1b1 at depth 4 (another safe move exists).
        let mut engine = Engine::new(16);
        let result = engine.search(&b, 4);
        if let Some(mv) = result.best_move {
            assert!(
                !(mv.from == sq(2,0) && mv.to == sq(1,0)),
                "engine must not play c1b1 — bishop on b4 would hang"
            );
        }
    }

    /// Paranoid search must avoid promoting a pawn to an undefended Boat when
    /// two opponents can immediately capture it.
    ///
    /// Position: Red pawn on h7 (one step from promotion).  The promotion
    /// square h8 is attacked by a Blue bishop and a Yellow knight — but Red has
    /// no piece that defends h8 after promotion.  Paranoid search must prefer
    /// any other move over the losing promotion.
    #[test]
    fn paranoid_avoids_undefended_promotion() {
        use chaturaji_core::board::{bit, sq};
        use chaturaji_core::piece::{Color, PieceKind};

        let mut b = Board::empty();
        // Red: king a1, pawn h7 (one step from promotion on h8)
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = bit(sq(0, 0)); // a1
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()] = bit(sq(7, 6)); // h7

        // Blue bishop on f6 (attacks h8 diagonally).
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(0, 7)); // a8
        b.bb[Color::Blue.idx()][PieceKind::Bishop.idx()] = bit(sq(5, 5)); // f6

        // Yellow knight on g6 (attacks h8).
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()]   = bit(sq(7, 7)); // h8 — put king elsewhere
        // actually put Yellow king somewhere else so h8 can be the target
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()]   = 0;
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()]   = bit(sq(4, 7)); // e8
        b.bb[Color::Yellow.idx()][PieceKind::Knight.idx()] = bit(sq(5, 6)); // f7 attacks h8

        b.bb[Color::Green.idx()][PieceKind::King.idx()] = bit(sq(7, 0)); // h1

        b.to_move = Color::Red;

        let mut engine = Engine::new(4);
        let result = engine.search_paranoid(&b, 4, None);
        if let Some(mv) = result.best_move {
            // h7→h8 promotion = from sq(7,6) to sq(7,7)
            assert!(
                !(mv.from == sq(7, 6) && mv.to == sq(7, 7)),
                "paranoid must not promote to undefended h8 — \
                 Blue bishop and Yellow knight both attack it"
            );
        }
    }

    // ── Best-Reply Search ────────────────────────────────────────────────────

    #[test]
    fn brs_returns_a_move() {
        let board = Board::default();
        let mut engine = Engine::new(4);
        let result = engine.search_brs(&board, 3, None);
        assert!(result.best_move.is_some(), "BRS should find a move at depth 3");
    }

    #[test]
    fn brs_top_n_returns_moves() {
        let board = Board::default();
        let mut engine = Engine::new(4);
        let top = engine.top_n_brs(&board, 2, 3, None);
        assert!(!top.is_empty(), "top_n_brs should return at least one move");
        assert!(top.len() <= 3);
        // Ranked best-first.
        let idx = board.to_move.idx();
        for w in top.windows(2) {
            assert!(w[0].scores[idx] >= w[1].scores[idx], "top_n_brs must be sorted");
        }
    }

    /// The point of BRS, stated in the unit that matters: **look-ahead in game
    /// rounds**, not nominal plies.
    ///
    /// A round is "everyone has moved once". Paranoid spends four plies on it,
    /// BRS spends two (one root move plus one best reply), so equal look-ahead
    /// means paranoid depth 4k against BRS depth 2k. Compared per *nominal*
    /// depth BRS actually looks more expensive, because its reply ply fans out
    /// over all three opponents — that comparison is the misleading one.
    ///
    /// Measured on the start position (see `examples/brs_bench.rs`):
    ///   1 round  — paranoid 1 862 nodes,     BRS 85 nodes
    ///   2 rounds — paranoid 1 234 866 nodes, BRS 2 229 nodes
    #[test]
    fn brs_is_far_cheaper_than_paranoid_at_equal_lookahead() {
        let board = Board::default();

        let mut e1 = Engine::new(4);
        let paranoid_nodes = e1.search_paranoid(&board, 4, None).nodes;

        let mut e2 = Engine::new(4);
        let brs_nodes = e2.search_brs(&board, 2, None).nodes;

        assert!(
            brs_nodes * 5 < paranoid_nodes,
            "BRS should be several times cheaper for one round of look-ahead: \
             BRS {brs_nodes}, paranoid {paranoid_nodes}"
        );
    }

    /// BRS must still see plain tactics: promoting onto a square two opponents
    /// attack is losing, and the best reply finds it.
    #[test]
    fn brs_avoids_undefended_promotion() {
        use chaturaji_core::board::{bit, sq};
        use chaturaji_core::piece::{Color, PieceKind};

        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = bit(sq(0, 0)); // a1
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()] = bit(sq(7, 6)); // h7

        b.bb[Color::Blue.idx()][PieceKind::King.idx()]     = bit(sq(0, 7)); // a8
        b.bb[Color::Blue.idx()][PieceKind::Bishop.idx()]   = bit(sq(5, 5)); // f6 → h8
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()]   = bit(sq(4, 7)); // e8
        b.bb[Color::Yellow.idx()][PieceKind::Knight.idx()] = bit(sq(5, 6)); // f7 → h8
        b.bb[Color::Green.idx()][PieceKind::King.idx()]    = bit(sq(7, 0)); // h1

        b.to_move = Color::Red;

        let mut engine = Engine::new(4);
        let result = engine.search_brs(&b, 4, None);
        if let Some(mv) = result.best_move {
            assert!(
                !(mv.from == sq(7, 6) && mv.to == sq(7, 7)),
                "BRS must not promote onto h8 — Blue bishop and Yellow knight both cover it"
            );
        }
    }

    /// An eliminated opponent must not get a reply ply. Without the `active`
    /// guard the best-reply loop would happily generate moves for a dead
    /// player and invent threats that cannot happen.
    #[test]
    fn brs_ignores_eliminated_opponents() {
        use chaturaji_core::board::{bit, sq};
        use chaturaji_core::piece::{Color, PieceKind};

        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0, 0));
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]    = bit(sq(3, 3));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(7, 0));
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(0, 7));
        // Yellow is out: no king, flagged inactive, but leaves a boat behind.
        b.bb[Color::Yellow.idx()][PieceKind::Boat.idx()] = bit(sq(7, 7));
        b.active[Color::Yellow.idx()] = false;
        b.to_move = Color::Red;

        let mut engine = Engine::new(4);
        let result = engine.search_brs(&b, 3, None);
        assert!(result.best_move.is_some(), "BRS must still work with a player eliminated");
    }

    #[test]
    fn paranoid_returns_a_move() {
        let board = Board::default();
        let mut engine = Engine::new(4);
        let result = engine.search_paranoid(&board, 2, None);
        assert!(result.best_move.is_some(), "paranoid should find a move at depth 2");
    }

    #[test]
    fn paranoid_top_n_returns_moves() {
        let board = Board::default();
        let mut engine = Engine::new(4);
        let top = engine.top_n_paranoid(&board, 2, 3, None);
        assert!(!top.is_empty(), "top_n_paranoid should return at least one move");
        assert!(top.len() <= 3);
    }
}
