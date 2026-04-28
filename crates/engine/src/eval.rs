//! Static evaluation of a Chaturaji position.
//!
//! Returns a score vector `[i32; 4]` – one value per player.  Positive means
//! "good for that player".  The engine's Max^n search maximises its own index.
//!
//! Components:
//!   1. Material difference (using chess.com piece values)
//!   2. Pawn advancement (encourage pushing pawns toward promotion)
//!   3. Mobility (number of legal moves ≈ activity)
//!   4. King safety (penalise a king with many attackers)

use chaturaji_core::board::{file_of, rank_of, Board};
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;

// ─── Piece-square tables ──────────────────────────────────────────────────────
//
// Flat 64-element tables, indexed by square (a1=0 .. h8=63).
// Used for Red (faces North). Other players' tables are obtained by rotating.

#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
   10, 10, 20, 30, 30, 20, 10, 10,
   50, 50, 50, 50, 50, 50, 50, 50,
    0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const KING_PST: [i32; 64] = [
   20, 30, 10,  0,  0, 10, 30, 20,
   20, 20,  0,  0,  0,  0, 20, 20,
  -10,-20,-20,-20,-20,-20,-20,-10,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
];

// Boats move like rooks: equal mobility from every square → flat table.
const BOAT_PST: [i32; 64] = [0; 64];

// ─── Rotation helpers ──────────────────────────────────────────────────────────

/// Rotate a square 90° clockwise (for Blue's perspective).
#[allow(dead_code)]
fn rot90(sq: u8) -> u8 {
    let f = file_of(sq);
    let r = rank_of(sq);
    // 90° CW: (f,r) → (7-r, f)
    (7 - r) * 8 + f   // ← corrected: new_file = 7-r, new_rank = f  
    // Actually: rot90 of (file=f, rank=r) → (file=r, rank=7-f)
    // sq = rank*8+file → new_sq = (7-f)*8 + r
}

fn rotate_sq(sq: u8, color: Color) -> u8 {
    match color {
        Color::Red    => sq,
        Color::Blue   => {
            let f = file_of(sq); let r = rank_of(sq);
            (7 - f) * 8 + r          // 90° CW for Blue (faces East)
        }
        Color::Yellow => 63 - sq,    // 180° for Yellow (faces South)
        Color::Green  => {
            let f = file_of(sq); let r = rank_of(sq);
            f * 8 + (7 - r)          // 270° CW for Green (faces West)
        }
    }
}

fn pst_value(pst: &[i32; 64], sq: u8, color: Color) -> i32 {
    pst[rotate_sq(sq, color) as usize]
}

// ─── Public evaluation ────────────────────────────────────────────────────────

/// Weights (tunable).
const W_MATERIAL:  i32 = 100;
const W_PST:       i32 = 1;
const W_MOBILITY:  i32 = 5;
const W_KING_SAFE: i32 = 20;

/// Evaluate the position and return a score vector [Red, Blue, Yellow, Green].
/// Scores are in centipawns relative to each player.
pub fn evaluate(board: &Board) -> [i32; 4] {
    let mut scores = [0i32; 4];

    for c in Color::ALL {
        let ci = c.idx();
        if !board.active[ci] {
            // Eliminated player has a very bad score.
            scores[ci] = -10_000;
            continue;
        }

        // 1. Accumulated game points (from captures + check bonuses)
        let game_pts = board.scores.get(c) * W_MATERIAL;

        // 2. Material on board
        let material: i32 = [
            PieceKind::Pawn, PieceKind::Knight, PieceKind::Bishop,
            PieceKind::Boat, PieceKind::King,
        ].iter().map(|&k| {
            let count = board.pieces(c, k).count_ones() as i32;
            count * k.capture_value() * W_MATERIAL
        }).sum();

        // 3. Piece-square table bonus
        let pst_bonus: i32 = pst_for_player(board, c);

        // 4. Mobility (number of legal moves)
        let mobility = {
            let mut tmp = board.clone();
            tmp.to_move = c;
            use chaturaji_core::movegen::MoveGen;
            MoveGen::generate(&tmp).len() as i32 * W_MOBILITY
        };

        // 5. King safety (penalise if king is attacked)
        let king_safety = {
            let attackers: i32 = Color::ALL.iter()
                .filter(|&&opp| opp != c && board.active[opp.idx()])
                .map(|&opp| {
                    let attacked = Rules::attacked_squares(board, opp);
                    if board.pieces(c, PieceKind::King) & attacked != 0 { 1 } else { 0 }
                })
                .sum();
            -attackers * W_KING_SAFE
        };

        scores[ci] = game_pts + material + pst_bonus + mobility + king_safety;
    }

    scores
}

fn pst_for_player(board: &Board, c: Color) -> i32 {
    let mut val = 0i32;
    let apply = |bb: u64, pst: &[i32; 64]| -> i32 {
        let mut v = 0; let mut b = bb;
        while b != 0 {
            let sq = b.trailing_zeros() as u8;
            b &= b - 1;
            v += pst_value(pst, sq, c);
        }
        v
    };
    val += apply(board.pieces(c, PieceKind::Pawn),   &PAWN_PST)   * W_PST;
    val += apply(board.pieces(c, PieceKind::Knight), &KNIGHT_PST) * W_PST;
    val += apply(board.pieces(c, PieceKind::Bishop), &BISHOP_PST) * W_PST;
    val += apply(board.pieces(c, PieceKind::Boat),   &BOAT_PST)   * W_PST;
    val += apply(board.pieces(c, PieceKind::King),   &KING_PST)   * W_PST;
    val
}
