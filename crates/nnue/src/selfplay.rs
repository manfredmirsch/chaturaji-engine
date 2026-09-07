//! Self-Play für NNUE-Training.
//!
//! Verwendet das NNUE-Netz direkt als Evaluierungsfunktion (Max^n mit TT).
//!
//! Beam-Suche: Interne Knoten sortieren Züge per billigem Material-Heuristik
//! (Schlagwert) und rekursieren nur in die besten `beam_width` davon.
//! NNUE-Evals laufen ausschließlich an Blattknoten → ~beam_width^(depth-1)
//! Evals pro Wurzel-Zug statt B^(depth-1).
//!
//! Kosten (approx., B=30 Wurzelzüge):
//!   depth 1, beam 0 – ~30   Evals/Zug  (greedy, sehr schnell)
//!   depth 2, beam 0 – ~900  Evals/Zug
//!   depth 4, beam 3 – ~810  Evals/Zug  (30 × 3³)
//!   depth 4, beam 5 – ~3750 Evals/Zug  (30 × 5³)

use std::collections::HashMap;
use std::cmp::Reverse;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use chaturaji_core::board::{Board, Move};
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;
use chaturaji_core::notation::move_to_str;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};
use chaturaji_engine::book::OpeningBook;
use crate::network::NnueNetwork;

pub struct SelfPlayConfig {
    pub epsilon_start:   f32,
    pub epsilon_end:     f32,
    pub epsilon_decay:   f32,
    /// Halbzüge, nach denen eine Partie abgebrochen wird.
    ///
    /// Der Wert ist keine Notbremse, sondern bestimmt mit, *welches Spiel* das
    /// Netz lernt. `Rules::is_game_over` ist allein „höchstens ein Spieler
    /// aktiv"; unter dieser Bedingung enden Selbstspiel-Partien praktisch nie
    /// von selbst — sie laufen bis zum Limit. Menschen dagegen geben auf oder
    /// spielen eine entschiedene Stellung nicht aus.
    ///
    /// Ausgezählt über 4000 echte Partien aus `game_data/`: Median 94
    /// Halbzüge, 90 % unter 152, nur gut 1 % erreichen 200. Ein Limit von 300
    /// füllte das Training daher zu einem großen Teil mit Stellungen, die in
    /// echten Partien nicht vorkommen.
    ///
    /// Der Abbruch verfälscht das Ziel nicht: `outcome::place_values` liest den
    /// Punktestand zum Abbruchzeitpunkt, und der ist auch bei unbeendeter
    /// Partie eine gültige Rangfolge.
    pub max_moves:       usize,
    pub engine_depth:    u8,
    /// Beam width at internal nodes (0 = unbegrenzt).
    /// Beim Beam werden Züge zuerst per 1-Ply-NNUE sortiert; nur die besten
    /// `beam_width` Züge werden rekursiv untersucht.
    /// Empfehlung: depth 1–2 → 0, depth 3 → 8, depth 4 → 6.
    pub beam_width:      usize,
    pub book_max_plies:  usize,
    pub book_min_count:  u32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            epsilon_start:  0.3,
            epsilon_end:    0.05,
            epsilon_decay:  0.995,
            // 150 ≈ 90 %-Quantil echter Partien (152).
            max_moves:      150,
            engine_depth:   1,
            beam_width:     0,
            book_max_plies: 16,
            book_min_count: 2,
        }
    }
}

pub struct Step {
    /// Die vollständige Stellung, nicht nur die Bitboards: das Netz braucht
    /// auch Punktestand, Zugrecht und Spielphase (siehe `features::dense_features`).
    pub board: Board,
    pub value: [f32; 4],
}

pub struct GameResult {
    pub steps:       Vec<Step>,
    pub final_board: Board,
    pub move_log:    Vec<String>,
    pub winner:      Option<Color>,
}

// ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

/// Billige Material-Heuristik für Beam-Sortierung (kein NNUE-Aufruf nötig).
#[inline]
fn move_priority(mv: Move) -> i32 {
    let capture = match mv.captured.map(|p| p.kind) {
        Some(PieceKind::King)   => 100,
        Some(PieceKind::Boat)   => 50,
        Some(PieceKind::Knight) => 30,
        Some(PieceKind::Bishop) => 30,
        Some(PieceKind::Pawn)   => 10,
        None                     => 0,
    };
    let promotion = if mv.promoted { 20 } else { 0 };
    capture + promotion
}

// ─── NNUE Max^n mit Transpositionstabelle ────────────────────────────────────

/// Rekursiver Max^n mit NNUE-Blattbewertung und optionalem Beam.
///
/// `beam_width > 0`: Interne Knoten sortieren Züge per `move_priority`
/// (Schlagwert, kein NNUE) und rekursieren nur in die besten `beam_width`.
/// NNUE-Evals laufen ausschließlich an Blattknoten (depth == 0).
///
/// Die TT speichert (Tiefe, Scorevektor); ein Eintrag wird nur verwendet,
/// wenn die gespeicherte Tiefe ≥ der angefragten Tiefe ist.
fn nnue_maxn(
    net:        &NnueNetwork,
    board:      &Board,
    depth:      u8,
    beam_width: usize,
    tt:         &mut HashMap<u64, (u8, [f32; 4])>,
    keys:       &ZobristKeys,
) -> [f32; 4] {
    if Rules::is_game_over(board) || depth == 0 {
        return net.forward(board);
    }

    let hash = hash_board(board, keys);
    if let Some(&(d, scores)) = tt.get(&hash) {
        if d >= depth { return scores; }
    }

    let mut all_moves = Rules::legal_moves(board);
    if all_moves.is_empty() {
        return net.forward(board);
    }

    let mover_idx = board.to_move.idx();

    // Beam: billige Heuristik-Sortierung, dann auf beam_width begrenzen.
    if beam_width > 0 && all_moves.len() > beam_width {
        all_moves.sort_by_key(|&mv| Reverse(move_priority(mv)));
        all_moves.truncate(beam_width);
    }
    let moves = all_moves;

    let mut best = [f32::NEG_INFINITY; 4];

    for mv in moves {
        let child  = Rules::apply_with_effects(board, mv);
        let scores = nnue_maxn(net, &child, depth - 1, beam_width, tt, keys);
        if scores[mover_idx] > best[mover_idx] {
            best = scores;
        }
    }

    tt.insert(hash, (depth, best));
    best
}

/// Bewertet alle legalen Züge mit NNUE Max^n und gibt den besten zurück.
/// Die Wurzel betrachtet immer alle legalen Züge (kein Beam auf Wurzelebene).
/// Die TT wird vor jedem Aufruf geleert (korrektes Tiefenhandling über Züge hinweg).
fn nnue_best_move(
    net:        &NnueNetwork,
    board:      &Board,
    moves:      &[Move],
    depth:      u8,
    beam_width: usize,
    keys:       &ZobristKeys,
    tt:         &mut HashMap<u64, (u8, [f32; 4])>,
) -> Move {
    let mover_idx = board.to_move.idx();
    let d1 = depth.saturating_sub(1);

    moves.iter().copied()
        .map(|mv| {
            let child = Rules::apply_with_effects(board, mv);
            let score = nnue_maxn(net, &child, d1, beam_width, tt, keys)[mover_idx];
            (mv, score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(mv, _)| mv)
        .unwrap_or(moves[0])
}

// ─── play_game ────────────────────────────────────────────────────────────────

pub fn play_game(
    net:     &NnueNetwork,
    cfg:     &SelfPlayConfig,
    epsilon: f32,
    rng:     &mut impl Rng,
    book:    Option<&OpeningBook>,
    keys:    &ZobristKeys,
) -> GameResult {
    let mut board    = Board::default();
    let mut steps    = Vec::with_capacity(cfg.max_moves);
    let mut move_log = Vec::with_capacity(cfg.max_moves);
    let mut tt: HashMap<u64, (u8, [f32; 4])> = HashMap::new();

    for ply in 0..cfg.max_moves {
        if Rules::is_game_over(&board) { break; }

        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }

        let value = net.forward(&board);
        steps.push(Step { board: board.clone(), value });

        let book_move = book
            .filter(|_| ply < cfg.book_max_plies)
            .and_then(|b| b.entries(&board, keys, cfg.book_min_count))
            .and_then(|entries| sample_book_move(&entries, &moves, rng));

        let chosen = if let Some(mv) = book_move {
            mv
        } else if rng.gen::<f32>() < epsilon {
            moves[rng.gen_range(0..moves.len())]
        } else {
            tt.clear();
            nnue_best_move(net, &board, &moves, cfg.engine_depth, cfg.beam_width, keys, &mut tt)
        };

        move_log.push(move_to_str(&chosen));
        board = Rules::apply_with_effects(&board, chosen);
    }

    let winner = Rules::winner(&board);
    GameResult { steps, final_board: board, move_log, winner }
}

// ─── Paralleles Self-Play ─────────────────────────────────────────────────────

/// Eine zu spielende Partie: laufende Nummer, RNG-Seed und ε.
///
/// Der Seed hängt allein an der globalen Partienummer, nicht an der Reihenfolge
/// der Abarbeitung. Damit liefert derselbe Lauf dasselbe Ergebnis, egal auf wie
/// vielen Kernen oder in wie vielen Shards er läuft — ohne das wäre ein
/// verteilter Lauf nicht reproduzierbar und ein Fehler nicht nachstellbar.
#[derive(Clone, Copy)]
pub struct GameJob {
    pub index:   u64,
    pub seed:    u64,
    pub epsilon: f32,
}

/// Ableitung des Partie-Seeds aus Lauf-Seed und Partienummer.
pub fn game_seed(run_seed: u64, index: u64) -> u64 {
    // SplitMix64-Finalizer: streut benachbarte Indizes weit auseinander, damit
    // Partie 7 und Partie 8 nicht mit verwandten Zufallsfolgen starten.
    let mut z = run_seed
        .wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Spielt einen Stapel Partien parallel gegen dasselbe, eingefrorene Netz.
///
/// Das Netz wird nur gelesen (`forward` nimmt `&self`), deshalb braucht es
/// keinen Lock. Die Rückgabe ist nach `GameJob::index` sortiert, damit die
/// anschließenden TD-Updates in fester Reihenfolge laufen.
pub fn play_batch(
    net:  &NnueNetwork,
    cfg:  &SelfPlayConfig,
    jobs: &[GameJob],
    book: Option<&OpeningBook>,
    keys: &ZobristKeys,
) -> Vec<(u64, GameResult)> {
    let mut out: Vec<(u64, GameResult)> = jobs
        .par_iter()
        .map(|job| {
            let mut rng = SmallRng::seed_from_u64(job.seed);
            (job.index, play_game(net, cfg, job.epsilon, &mut rng, book, keys))
        })
        .collect();
    out.sort_by_key(|(idx, _)| *idx);
    out
}

fn sample_book_move(
    entries: &[(u8, u8, u32)],
    legal:   &[Move],
    rng:     &mut impl Rng,
) -> Option<Move> {
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

/// Endstand → Zielvektor für das Training: die Platzwertung des Endergebnisses.
///
/// Dieselbe Kodierung wie im PGN-/JSON-Import (`outcome::place_values`), damit
/// Self-Play und Supervised Learning dasselbe lernen.
pub fn final_targets(board: &Board) -> [f32; 4] {
    // `Rules::final_scores` statt `board.scores`: bleibt genau ein Spieler
    // übrig, gehören ihm die 3 Punkte je nie geschlagenem König. Im Self-Play
    // ändert das nichts — dort scheidet man nur durch den Verlust des Königs
    // aus, es bleibt also keiner stehen. Beide Wege sollen aber dieselbe
    // Rechnung benutzen, damit Self-Play und echte Partien dasselbe Ziel
    // lernen.
    crate::outcome::place_values(Rules::final_scores(board))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_short_game_produces_steps() {
        let net = NnueNetwork::new(0.001, 0.9);
        let cfg = SelfPlayConfig { max_moves: 20, ..Default::default() };
        let mut rng = rand::thread_rng();
        let keys = ZobristKeys::new();
        let result = play_game(&net, &cfg, 1.0, &mut rng, None, &keys);
        assert!(!result.steps.is_empty());
        assert!(!result.move_log.is_empty());
    }

    #[test]
    fn nnue_greedy_picks_a_move() {
        let net   = NnueNetwork::new(0.001, 0.9);
        let board = Board::default();
        let moves = Rules::legal_moves(&board);
        let keys  = ZobristKeys::new();
        let mut tt = HashMap::new();
        let mv = nnue_best_move(&net, &board, &moves, 1, 0, &keys, &mut tt);
        assert!(moves.contains(&mv));
    }

    #[test]
    fn nnue_depth2_picks_a_move() {
        let net   = NnueNetwork::new(0.001, 0.9);
        let board = Board::default();
        let moves = Rules::legal_moves(&board);
        let keys  = ZobristKeys::new();
        let mut tt = HashMap::new();
        let mv = nnue_best_move(&net, &board, &moves, 2, 0, &keys, &mut tt);
        assert!(moves.contains(&mv));
    }

    #[test]
    fn nnue_depth4_beam_picks_a_move() {
        let net   = NnueNetwork::new(0.001, 0.9);
        let board = Board::default();
        let moves = Rules::legal_moves(&board);
        let keys  = ZobristKeys::new();
        let mut tt = HashMap::new();
        let mv = nnue_best_move(&net, &board, &moves, 4, 6, &keys, &mut tt);
        assert!(moves.contains(&mv));
    }

    #[test]
    fn final_targets_in_range() {
        for v in final_targets(&Board::default()) {
            assert!((-1.0..=1.0).contains(&v));
        }
    }
}
