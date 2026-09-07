//! Move generator for chess.com Chaturaji.
//!
//! Piece movement rules:
//!   King   – one step in any of 8 directions (as standard chess)
//!   Knight – standard chess knight (L-shape)
//!   Bishop – standard chess bishop (diagonal slider)
//!   Boat   – moves like a rook (orthogonal slider, any distance)
//!   Pawn   – moves one step *forward* (direction depends on player),
//!            captures one step forward-diagonally; no initial double step;
//!            promotes to Boat upon reaching the back rank.
//!
//! Pawn directions (all relative to the pawn's color):
//!   Red    → moves North  (+rank)
//!   Blue   → moves West   (-file)
//!   Yellow → moves South  (-rank)
//!   Green  → moves East   (+file)

use crate::board::{bit, file_of, rank_of, sq, Board, Move};
use crate::piece::{Color, PieceKind};

/// Move generator.  Stateless – operates purely on a `Board` reference.
pub struct MoveGen;

impl MoveGen {
    /// Generate all pseudo-legal moves for the active player.
    /// (Pseudo-legal: does not filter moves that walk into check, as
    ///  Chaturaji has no check rule – kings are simply captured.)
    pub fn generate(board: &Board) -> Vec<Move> {
        let mover = board.to_move;
        if !board.active[mover.idx()] {
            return vec![];
        }

        let mut moves = Vec::with_capacity(64);

        let friendly = board.occupied_by(mover);

        // Generate per piece kind
        Self::gen_pawns(board, mover, friendly, &mut moves);
        Self::gen_knights(board, mover, friendly, &mut moves);
        Self::gen_bishops(board, mover, friendly, &mut moves);
        Self::gen_boats(board, mover, friendly, &mut moves);
        Self::gen_kings(board, mover, friendly, &mut moves);

        moves
    }

    // ─── Pawn ─────────────────────────────────────────────────────────────────

    fn gen_pawns(board: &Board, mover: Color, friendly: u64, moves: &mut Vec<Move>) {
        let pawns = board.pieces(mover, PieceKind::Pawn);
        let all_occ = board.all_occupied();

        let mut bb = pawns;
        while bb != 0 {
            let from = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            let f = file_of(from) as i8;
            let r = rank_of(from) as i8;

                        // Zugrichtungen (chess.com):
            //   Rot   → Nord  (+rank), Promotion rank 7
            //   Blau  → Ost   (+file), Promotion file 7
            //   Gelb  → Süd   (-rank), Promotion rank 0
            //   Grün  → West  (-file), Promotion file 0
            let (fwd, cap_dirs): (_, &[(i8,i8)]) = match mover {
                Color::Red    => ((f,   r+1), &[(-1, 1),(1, 1)]),  // Nord,  schlägt NW+NO
                Color::Blue   => ((f+1, r),   &[(1, -1),(1,  1)]), // Ost,   schlägt SO+NO
                Color::Yellow => ((f,   r-1), &[(-1,-1),(1,-1)]),  // Süd,   schlägt SW+SO
                Color::Green  => ((f-1, r),   &[(-1,-1),(-1, 1)]), // West,  schlägt SW+NW
            };

            // Ruhiger Vorwärtszug
            if let Some(to) = Self::try_sq(fwd.0, fwd.1) {
                if bit(to) & all_occ == 0 {
                    let promoted = Self::promotes(mover, to);
                    let mut mv = Move::new(from, to, mover);
                    mv.promoted = promoted;
                    moves.push(mv);
                }
            }

            // Schlagzüge (aktive und Zombie-Figuren)
            for &(df, dr) in cap_dirs {
                if let Some(to) = Self::try_sq(f + df, r + dr) {
                    if bit(to) & friendly == 0 {
                        if let Some(cap) = board.piece_at(to) {
                            let promoted = Self::promotes(mover, to);
                            let mut mv = Move::new(from, to, mover);
                            mv.captured = Some(cap);
                            mv.promoted = promoted;
                            moves.push(mv);
                        }
                    }
                }
            }
        }
    }

    /// Promotionsbedingung: Bauer erreicht die hinterste Reihe des Gegners.
    fn promotes(mover: Color, to: u8) -> bool {
        match mover {
            Color::Red    => rank_of(to) == 7, // Rang 8 (Nord-Rand)
            Color::Blue   => file_of(to) == 7, // File h (Ost-Rand)
            Color::Yellow => rank_of(to) == 0, // Rang 1 (Süd-Rand)
            Color::Green  => file_of(to) == 0, // File a (West-Rand)
        }
    }

    // ─── Knight ───────────────────────────────────────────────────────────────

    fn gen_knights(board: &Board, mover: Color, friendly: u64, moves: &mut Vec<Move>) {
        const DELTAS: [(i8,i8);8] = [
            (-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)
        ];
        Self::gen_leaper(board, mover, PieceKind::Knight, friendly, &DELTAS, moves);
    }

    // ─── King ─────────────────────────────────────────────────────────────────

    fn gen_kings(board: &Board, mover: Color, friendly: u64, moves: &mut Vec<Move>) {
        const DELTAS: [(i8,i8);8] = [
            (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)
        ];
        Self::gen_leaper(board, mover, PieceKind::King, friendly, &DELTAS, moves);
    }

    /// Generic leaper generator (knight, king).
    fn gen_leaper(
        board: &Board,
        mover: Color,
        kind: PieceKind,
        friendly: u64,
        deltas: &[(i8,i8)],
        moves: &mut Vec<Move>,
    ) {
        let mut bb = board.pieces(mover, kind);
        while bb != 0 {
            let from = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            let f = file_of(from) as i8;
            let r = rank_of(from) as i8;
            for &(df, dr) in deltas {
                if let Some(to) = Self::try_sq(f+df, r+dr) {
                    if bit(to) & friendly != 0 { continue; }
                    let mut mv = Move::new(from, to, mover);
                    mv.captured = board.piece_at(to);
                    moves.push(mv);
                }
            }
        }
    }

    // ─── Bishop (slider) ──────────────────────────────────────────────────────

    fn gen_bishops(board: &Board, mover: Color, friendly: u64, moves: &mut Vec<Move>) {
        const DIRS: [(i8,i8);4] = [(-1,-1),(-1,1),(1,-1),(1,1)];
        Self::gen_slider(board, mover, PieceKind::Bishop, friendly, &DIRS, moves);
    }

    /// Generic sliding piece generator.
    fn gen_slider(
        board: &Board,
        mover: Color,
        kind: PieceKind,
        friendly: u64,
        dirs: &[(i8,i8)],
        moves: &mut Vec<Move>,
    ) {
        let all_occ = board.all_occupied();
        let mut bb = board.pieces(mover, kind);
        while bb != 0 {
            let from = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            let f0 = file_of(from) as i8;
            let r0 = rank_of(from) as i8;
            for &(df, dr) in dirs {
                let (mut f, mut r) = (f0 + df, r0 + dr);
                while (0..8).contains(&f) && (0..8).contains(&r) {
                    let to = sq(f as u8, r as u8);
                    if bit(to) & friendly != 0 { break; }
                    let mut mv = Move::new(from, to, mover);
                    mv.captured = board.piece_at(to);
                    moves.push(mv);
                    if bit(to) & all_occ != 0 { break; } // blocked after capture
                    f += df; r += dr;
                }
            }
        }
    }

    // ─── Boat (rook: orthogonal slider) ──────────────────────────────────────

    fn gen_boats(board: &Board, mover: Color, friendly: u64, moves: &mut Vec<Move>) {
        const DIRS: [(i8,i8);4] = [(-1,0),(1,0),(0,-1),(0,1)];
        Self::gen_slider(board, mover, PieceKind::Boat, friendly, &DIRS, moves);
    }

    // ─── Utility ─────────────────────────────────────────────────────────────

    /// Returns `Some(square)` if (f, r) is on the board.
    #[inline]
    fn try_sq(f: i8, r: i8) -> Option<u8> {
        if (0..8).contains(&f) && (0..8).contains(&r) {
            Some(sq(f as u8, r as u8))
        } else {
            None
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn red_has_moves_at_start() {
        let b = Board::default();
        let moves = MoveGen::generate(&b);
        assert!(!moves.is_empty(), "Red must have legal moves at start");
    }

    #[test]
    fn red_pawn_forward_counts() {
        let b = Board::default();
        // Red has 4 pawns on rank 2, each can advance to rank 3 (no captures possible).
        let pawn_moves: Vec<_> = MoveGen::generate(&b)
            .into_iter()
            .filter(|m| {
                b.piece_at(m.from).map(|p| p.kind == PieceKind::Pawn).unwrap_or(false)
            })
            .collect();
        assert_eq!(pawn_moves.len(), 4, "Red should have exactly 4 pawn pushes at start");
    }

    #[test]
    fn boat_moves_like_rook() {
        // Red boat on d4 (file=3, rank=3), isolated.
        // Should reach all squares on file 3 and rank 3 (except its own square).
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()] = bit(sq(3,3)); // d4
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = bit(sq(0,0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()] = bit(sq(7,7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0,7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()] = bit(sq(7,0));
        b.to_move = Color::Red;

        let moves = MoveGen::generate(&b);
        let boat_moves: Vec<_> = moves.iter().filter(|m| m.from == sq(3,3)).collect();
        // 7 squares along file 3 + 7 squares along rank 3 = 14 moves (minus own king on a1)
        // a1 is sq(0,0) which is on rank 0 and file 0 — not on rank 3 or file 3, so 14 moves
        assert_eq!(boat_moves.len(), 14, "boat on d4 should have 14 rook moves on empty board");
        // must reach d1 (file=3, rank=0) and h4 (file=7, rank=3)
        assert!(boat_moves.iter().any(|m| m.to == sq(3,0)), "boat must reach d1");
        assert!(boat_moves.iter().any(|m| m.to == sq(7,3)), "boat must reach h4");
        // must NOT reach e5 (diagonal)
        assert!(!boat_moves.iter().any(|m| m.to == sq(4,4)), "boat must NOT move diagonally");
    }

    #[test]
    fn boat_blocked_by_piece() {
        // Red boat on d4 (file=3, rank=3), friendly pawn on d6 (file=3, rank=5).
        // Boat must not reach d7 or d8, but can reach d5 (the square before the pawn is on d6).
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()] = bit(sq(3,3)); // d4
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()] = bit(sq(3,5)); // d6 blocks
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = bit(sq(0,0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()] = bit(sq(7,7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0,7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()] = bit(sq(7,0));
        b.to_move = Color::Red;

        let moves = MoveGen::generate(&b);
        let boat_moves: Vec<_> = moves.iter().filter(|m| m.from == sq(3,3)).collect();
        assert!(boat_moves.iter().any(|m| m.to == sq(3,4)),  "boat must reach d5");
        assert!(!boat_moves.iter().any(|m| m.to == sq(3,5)), "boat must NOT capture own pawn on d6");
        assert!(!boat_moves.iter().any(|m| m.to == sq(3,6)), "boat must NOT reach d7 (blocked)");
    }

    #[test]
    fn king_cannot_capture_friendly() {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::King.idx()]   = bit(sq(0,0));
        b.bb[Color::Red.idx()][PieceKind::Bishop.idx()] = bit(sq(1,0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]  = bit(sq(7,7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0,7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()] = bit(sq(7,0));
        b.to_move = Color::Red;

        let moves = MoveGen::generate(&b);
        let king_to_b1: Vec<_> = moves.iter()
            .filter(|m| m.from == sq(0,0) && m.to == sq(1,0))
            .collect();
        assert!(king_to_b1.is_empty(), "king must not capture friendly bishop");
    }
}
