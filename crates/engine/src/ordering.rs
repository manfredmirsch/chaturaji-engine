//! Move ordering for better alpha-beta pruning.
//!
//! Priority (descending):
//!   1. TT best move from a previous iteration
//!   2. Captures, scored by MVV-LVA (most-valuable-victim / least-valuable-attacker)
//!   3. Non-captures (quiet moves)

use chaturaji_core::board::{Board, Move};
use chaturaji_core::piece::PieceKind;

/// Assign an ordering score to a move.  Higher = searched first.
pub fn score_move(
    board:   &Board,
    mv:      &Move,
    tt_best: Option<Move>,
    killers: &[Option<Move>; 2],
) -> i32 {
    // 1. TT best move
    if let Some(best) = tt_best {
        if mv.from == best.from && mv.to == best.to { return 100_000; }
    }

    // 2. Captures: MVV-LVA
    if let Some(cap) = mv.captured {
        let mover_kind = board.piece_at(mv.from)
            .map(|p| p.kind)
            .unwrap_or(PieceKind::Pawn);

        let victim_val   = cap.kind.capture_value();
        let attacker_val = mover_kind.capture_value();

        // MVV-LVA: high victim value is good, low attacker value is good
        return 10_000 + victim_val * 10 - attacker_val;
    }

    // 3. Killer moves (quiet refutations from the same depth in prior iterations)
    if killers[0] == Some(*mv) { return 9_500; }
    if killers[1] == Some(*mv) { return 9_000; }

    // 4. Quiet moves – prefer promotions
    if mv.promoted { return 5_000; }

    0
}

/// Sort moves in-place, best first.
pub fn order_moves(
    board:   &Board,
    moves:   &mut Vec<Move>,
    tt_best: Option<Move>,
    killers: &[Option<Move>; 2],
) {
    moves.sort_unstable_by(|a, b| {
        score_move(board, b, tt_best, killers)
            .cmp(&score_move(board, a, tt_best, killers))
    });
}
