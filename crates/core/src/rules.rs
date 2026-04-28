//! High-level rule enforcement for chess.com Chaturaji.
//!
//! This layer sits above movegen and handles:
//!   • Boat triumph (four boats form a 2×2 → capture all three others)
//!   • Double-check (+1) and triple-check (+5) scoring bonus
//!   • Game-over detection
//!   • Filtering out moves that leave the mover with no king (not needed in
//!     Chaturaji, since there is no check rule – kept as a stub for clarity)

use crate::board::{bit, Board, Move};
use crate::movegen::MoveGen;
use crate::piece::{Color, PieceKind};

/// Stateless rule-checker.
pub struct Rules;

impl Rules {
    // ─── Legal move enumeration ───────────────────────────────────────────────

    /// Returns all fully legal moves for the current player, including
    /// post-move side effects (boat triumph checks happen after application).
    pub fn legal_moves(board: &Board) -> Vec<Move> {
        // In Chaturaji there is no check/checkmate, so all pseudo-legal moves
        // are legal.  We keep this wrapper so the engine always calls Rules,
        // not MoveGen directly, making future rule additions easy.
        MoveGen::generate(board)
    }

    // ─── Move application with full rule enforcement ───────────────────────────

    /// Apply `mv` and return the new board with all rule effects resolved.
    pub fn apply_with_effects(board: &Board, mv: Move) -> Board {
        let mut next = board.apply_move(mv);

        // 1. Boat triumph
        Self::resolve_boat_triumph(&mut next, mv.mover);

        // 2. Check bonuses
        Self::apply_check_bonus(&mut next, mv.mover);

        next
    }

    // ─── Boat triumph ─────────────────────────────────────────────────────────
    //
    // When a boat moves such that a 2×2 square filled with boats is formed,
    // the moving player's boat captures all three other boats in that square.
    // This is evaluated AFTER the move has been applied.

    fn resolve_boat_triumph(board: &mut Board, mover: Color) {
        // Collect all boat squares across all active players.
        let mut all_boats: u64 = 0;
        for c in Color::ALL {
            if board.active[c.idx()] {
                all_boats |= board.pieces(c, PieceKind::Boat);
            }
        }

        // Check every possible 2×2 top-left corner (files 0-6, ranks 0-6).
        for r in 0..7u8 {
            for f in 0..7u8 {
                let sq00 = r * 8 + f;
                let sq10 = r * 8 + f + 1;
                let sq01 = (r+1) * 8 + f;
                let sq11 = (r+1) * 8 + f + 1;
                let mask = bit(sq00) | bit(sq10) | bit(sq01) | bit(sq11);

                if all_boats & mask == mask {
                    // All four squares are boats – check if mover owns at least one.
                    let mover_boats = board.pieces(mover, PieceKind::Boat) & mask;
                    if mover_boats != 0 {
                        // Capture the three foreign boats.
                        for c in Color::ALL {
                            if c == mover { continue; }
                            if !board.active[c.idx()] { continue; }
                            let foreign = board.bb[c.idx()][PieceKind::Boat.idx()] & mask;
                            if foreign != 0 {
                                board.bb[c.idx()][PieceKind::Boat.idx()] &= !foreign;
                                let n = foreign.count_ones() as i32;
                                board.scores.add(mover, n * PieceKind::Boat.capture_value());
                            }
                        }
                    }
                }
            }
        }
    }

    // ─── Check bonus ──────────────────────────────────────────────────────────
    //
    // After a move, count how many active opponents' kings are attacked
    // by the mover's pieces.
    //   1 king attacked → no bonus (single check)
    //   2 kings attacked → +1 (double check)
    //   3 kings attacked → +5 (triple check)
    //
    // "Attacked" means the mover has at least one legal move to the king's square.

    fn apply_check_bonus(board: &mut Board, mover: Color) {
        let attacked_kings = Self::count_attacked_kings(board, mover);
        let bonus = match attacked_kings {
            2 => 1,
            3 => 5,
            _ => 0,
        };
        if bonus > 0 {
            board.scores.add(mover, bonus);
        }
    }

    /// Count how many *active* opponents' kings the `mover` attacks.
    pub fn count_attacked_kings(board: &Board, mover: Color) -> usize {
        let attacked = Self::attacked_squares(board, mover);
        Color::ALL
            .iter()
            .filter(|&&c| c != mover && board.active[c.idx()])
            .filter(|&&c| board.pieces(c, PieceKind::King) & attacked != 0)
            .count()
    }

    /// Bitboard of all squares the `mover` attacks (used for check detection).
    /// This is a fast approximation: generates pseudo-legal moves and marks destinations.
    pub fn attacked_squares(board: &Board, mover: Color) -> u64 {
        // Temporarily set to_move to `mover` so MoveGen generates their moves.
        let mut tmp = board.clone();
        tmp.to_move = mover;
        MoveGen::generate(&tmp)
            .iter()
            .fold(0u64, |acc, mv| acc | bit(mv.to))
    }

    // ─── Game-over ────────────────────────────────────────────────────────────

    /// Returns true when the game has ended (≤1 active player).
    pub fn is_game_over(board: &Board) -> bool {
        board.is_terminal()
    }

    /// The winning player (sole survivor), or None if still running.
    pub fn winner(board: &Board) -> Option<Color> {
        board.winner()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{sq, Board};
    use crate::piece::Color;

    #[test]
    fn legal_moves_non_empty_at_start() {
        let b = Board::default();
        assert!(!Rules::legal_moves(&b).is_empty());
    }

    #[test]
    fn boat_triumph_captures_three_boats() {
        // Set up four boats in a 2×2 square: d4, e4, d5, e5
        // Red owns d4, others own the remaining three.
        let mut b = Board::empty();

        let d4 = sq(3,3); let e4 = sq(4,3);
        let d5 = sq(3,4); let e5 = sq(4,4);

        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]    = bit(d4);
        b.bb[Color::Blue.idx()][PieceKind::Boat.idx()]   = bit(e4);
        b.bb[Color::Yellow.idx()][PieceKind::Boat.idx()] = bit(d5);
        b.bb[Color::Green.idx()][PieceKind::Boat.idx()]  = bit(e5);

        // Add kings so the board is valid
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0,0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(7,7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0,7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7,0));
        b.to_move = Color::Red;

        // Trigger triumph resolution directly
        Rules::resolve_boat_triumph(&mut b, Color::Red);

        // Red should now have all four squares; others should have no boats
        assert_eq!(b.bb[Color::Blue.idx()][PieceKind::Boat.idx()], 0);
        assert_eq!(b.bb[Color::Yellow.idx()][PieceKind::Boat.idx()], 0);
        assert_eq!(b.bb[Color::Green.idx()][PieceKind::Boat.idx()], 0);
        assert_eq!(b.scores.get(Color::Red), 15); // 3 boats × 5 points
    }
}
