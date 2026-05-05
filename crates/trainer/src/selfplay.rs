//! Self-Play: Engine spielt gegen sich selbst.
//!
//! Zugselektion:
//!   • Exploration (ε):   zufälliger Zug
//!   • Exploitation (1-ε): bester Zug laut Engine-Suche (Tiefe = engine_depth)
//!
//! Die Engine (Max^n + Transpositionstabelle) bewertet Züge deutlich stärker
//! als ein reiner 1-Ply-Net-Lookahead.  Das Training-Signal (TD-Updates) kommt
//! weiterhin ausschließlich vom neuronalen Netz.

use rand::Rng;
use chaturaji_core::board::Board;
use chaturaji_core::piece::Color;
use chaturaji_core::rules::Rules;
use chaturaji_core::notation::move_to_str;
use chaturaji_core::zobrist::ZobristKeys;
use chaturaji_engine::book::OpeningBook;
use chaturaji_engine::search::Engine;
use crate::features::extract;
use crate::network::Network;

pub struct SelfPlayConfig {
    pub epsilon_start: f32,
    pub epsilon_end:   f32,
    pub epsilon_decay: f32,
    pub max_moves:     usize,
    /// Suchtiefe der Engine bei der Zugselektion (Exploitation).
    pub engine_depth:  u8,
    /// Wie viele Halbzüge lang das Eröffnungsbuch konsultiert wird.
    pub book_max_plies: usize,
    /// Mindestanzahl Beobachtungen, damit ein Buchzug verwendet wird.
    pub book_min_count: u32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            epsilon_start:  0.3,
            epsilon_end:    0.05,
            epsilon_decay:  0.995,
            max_moves:      300,
            engine_depth:   3,
            book_max_plies: 16,
            book_min_count: 2,
        }
    }
}

impl Clone for SelfPlayConfig {
    fn clone(&self) -> Self {
        Self {
            epsilon_start:  self.epsilon_start,
            epsilon_end:    self.epsilon_end,
            epsilon_decay:  self.epsilon_decay,
            max_moves:      self.max_moves,
            engine_depth:   self.engine_depth,
            book_max_plies: self.book_max_plies,
            book_min_count: self.book_min_count,
        }
    }
}

pub struct Step {
    pub board:    Board,
    pub features: Vec<f32>,
    pub value:    [f32; 4],
}

pub struct GameResult {
    pub steps:       Vec<Step>,
    pub final_board: Board,
    pub move_log:    Vec<String>,
    pub winner:      Option<Color>,
}

pub fn play_game(
    net:     &Network,
    cfg:     &SelfPlayConfig,
    epsilon: f32,
    rng:     &mut impl Rng,
    book:    Option<&OpeningBook>,
    keys:    &ZobristKeys,
) -> GameResult {
    let mut board    = Board::default();
    let mut steps    = Vec::with_capacity(cfg.max_moves);
    let mut move_log = Vec::with_capacity(cfg.max_moves);
    // Engine-Instanz pro Partie; TT-Größe klein halten (2 MB) für schnelles Training.
    let mut engine   = Engine::new(2);

    for ply in 0..cfg.max_moves {
        if Rules::is_game_over(&board) { break; }

        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }

        let features = extract(&board);
        let value    = net.forward(&features);
        steps.push(Step { board: board.clone(), features, value });

        // Eröffnungsbuch zuerst: in den ersten Halbzügen, count-gewichtetes Sampling
        // → diverse, menschenähnliche Eröffnungen statt Engine-Tunneling.
        let book_move = book
            .filter(|_| ply < cfg.book_max_plies)
            .and_then(|b| b.entries(&board, keys, cfg.book_min_count))
            .and_then(|entries| sample_book_move(&entries, &moves, rng));

        let chosen = if let Some(mv) = book_move {
            mv
        } else if rng.gen::<f32>() < epsilon {
            let idx = rng.gen_range(0..moves.len());
            moves[idx]
        } else {
            // Engine-Suche liefert den stärksten Zug bis zur konfigurierten Tiefe.
            match engine.search(&board, cfg.engine_depth).best_move {
                Some(mv) => mv,
                None     => moves[0],
            }
        };

        move_log.push(move_to_str(&chosen));
        board = Rules::apply_with_effects(&board, chosen);
    }

    let winner = Rules::winner(&board);
    GameResult { steps, final_board: board, move_log, winner }
}

fn sample_book_move(
    entries: &[(u8, u8, u32)],
    legal:   &[chaturaji_core::board::Move],
    rng:     &mut impl Rng,
) -> Option<chaturaji_core::board::Move> {
    let total: u64 = entries.iter().map(|(_, _, c)| *c as u64).sum();
    if total == 0 { return None; }
    let mut pick = rng.gen_range(0..total);
    for (from, to, count) in entries {
        let c = *count as u64;
        if pick < c {
            return legal.iter().find(|m| m.from == *from && m.to == *to).copied();
        }
        pick -= c;
    }
    None
}

pub fn final_targets(board: &Board) -> [f32; 4] {
    let scores    = board.scores.as_array();
    let max_score = *scores.iter().max().unwrap_or(&1) as f32;
    let min_score = *scores.iter().min().unwrap_or(&0) as f32;
    let range     = (max_score - min_score).max(1.0);
    let mut targets = [0.0f32; 4];
    for i in 0..4 {
        targets[i] = (2.0 * (scores[i] as f32 - min_score) / range) - 1.0;
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_short_game() {
        let net = Network::new(0.001, 0.9);
        let cfg = SelfPlayConfig { max_moves: 20, ..Default::default() };
        let mut rng = rand::thread_rng();
        let keys = ZobristKeys::new();
        let result = play_game(&net, &cfg, 1.0, &mut rng, None, &keys);
        assert!(!result.steps.is_empty());
        assert!(!result.move_log.is_empty());
    }

    #[test]
    fn final_targets_in_range() {
        let board = Board::default();
        for v in final_targets(&board) {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }
}
