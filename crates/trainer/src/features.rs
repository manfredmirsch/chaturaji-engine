//! Feature extraction: Board → f32-Vektor für das neuronale Netz.
//!
//! 56 kompakte Features:
//!   1.  Figuranzahl pro (Spieler, Typ)          4×5  = 20
//!   2.  Ø Bauern-Fortschritt pro Spieler              =  4
//!   3.  Zentrale Kontrolle pro Spieler                =  4
//!   4.  Mobilität (gewichtet) pro Spieler             =  4
//!   5.  Aktiv/Inaktiv pro Spieler                     =  4
//!   6.  Normalisierte Punkte pro Spieler              =  4
//!   7.  Wer ist am Zug (One-Hot)                      =  4
//!   8.  Spielphase (One-Hot früh/mittel/spät)         =  3
//!   9.  Halbzug-Normalisierung                        =  1
//!  10.  König-Distanz zur Mitte pro Spieler           =  4
//!  11.  Boot-Anzahl pro Spieler (Triumph-Potential)   =  4
//!                                                   ----
//!                                                     56

use chaturaji_core::board::{rank_of, Board};
use chaturaji_core::piece::{Color, PieceKind};

pub const INPUT_SIZE: usize = 56;

const MAX_SCORE: f32 = 200.0;

/// Extrahiert den Feature-Vektor für eine Stellung.
pub fn extract(board: &Board) -> Vec<f32> {
    let mut f = Vec::with_capacity(INPUT_SIZE);

    // 1. Figuranzahl pro (Spieler, Typ): 4 × 5 = 20
    for c in Color::ALL {
        for k in PieceKind::ALL {
            let count = board.bb[c.idx()][k.idx()].count_ones() as f32;
            let max_count: f32 = match k {
                PieceKind::Pawn   => 4.0,
                PieceKind::Knight => 1.0,
                PieceKind::Bishop => 1.0,
                PieceKind::Boat   => 1.0,
                PieceKind::King   => 1.0,
            };
            f.push(count / max_count.max(1.0));
        }
    }

    // 2. Ø Bauern-Fortschritt pro Spieler (+4 = 24)
    for c in Color::ALL {
        let pawns = board.bb[c.idx()][PieceKind::Pawn.idx()];
        let count = pawns.count_ones();
        if count == 0 {
            f.push(0.0);
        } else {
            let total_rank: u32 = (0u8..64)
                .filter(|&sq| pawns & (1u64 << sq) != 0)
                .map(|sq| {
                    let r = rank_of(sq) as u32;
                    match c {
                        Color::Red    => r,
                        Color::Blue   => 7 - r,
                        Color::Yellow => 7 - r,
                        Color::Green  => r,
                    }
                })
                .sum();
            f.push((total_rank as f32) / (count as f32) / 7.0);
        }
    }

    // 3. Zentrale Kontrolle (+4 = 28)
    const CENTER_MASK: u64 = {
        let mut m = 0u64;
        let squares = [18,19,20,21, 26,27,28,29, 34,35,36,37, 42,43,44,45];
        let mut i = 0;
        while i < squares.len() { m |= 1u64 << squares[i]; i += 1; }
        m
    };
    for c in Color::ALL {
        let occ = board.bb[c.idx()].iter().fold(0u64, |a, &b| a | b);
        f.push((occ & CENTER_MASK).count_ones() as f32 / 16.0);
    }

    // 4. Mobilität (Boot = Turm → avg. 7 Züge, wie Läufer) (+4 = 32)
    for c in Color::ALL {
        if !board.active[c.idx()] {
            f.push(0.0);
            continue;
        }
        let bishops = board.bb[c.idx()][PieceKind::Bishop.idx()].count_ones() as f32;
        let knights = board.bb[c.idx()][PieceKind::Knight.idx()].count_ones() as f32;
        let boats   = board.bb[c.idx()][PieceKind::Boat.idx()].count_ones() as f32;
        let pawns   = board.bb[c.idx()][PieceKind::Pawn.idx()].count_ones() as f32;
        let mobility = (bishops * 7.0 + knights * 4.0 + boats * 7.0 + pawns * 1.0) / 56.0;
        f.push(mobility.min(1.0));
    }

    // 5. Aktiv/Inaktiv (+4 = 36)
    for c in Color::ALL {
        f.push(if board.active[c.idx()] { 1.0 } else { 0.0 });
    }

    // 6. Normalisierte Punkte (+4 = 40)
    for c in Color::ALL {
        f.push((board.scores.get(c) as f32 / MAX_SCORE).clamp(-1.0, 1.0));
    }

    // 7. Wer ist am Zug (+4 = 44)
    for c in Color::ALL {
        f.push(if board.to_move == c { 1.0 } else { 0.0 });
    }

    // 8. Spielphase (+3 = 47)
    let total_pieces: u32 = Color::ALL.iter()
        .map(|&c| board.bb[c.idx()].iter().map(|bb| bb.count_ones()).sum::<u32>())
        .sum();
    let phase = if total_pieces >= 24 { 0 } else if total_pieces >= 12 { 1 } else { 2 };
    for i in 0..3 { f.push(if phase == i { 1.0 } else { 0.0 }); }

    // 9. Halbzug-Normalisierung (+1 = 48)
    f.push((board.half_moves as f32 / 200.0).min(1.0));

    // 10. König-Distanz zur Mitte (+4 = 52)
    for c in Color::ALL {
        let king_bb = board.bb[c.idx()][PieceKind::King.idx()];
        if king_bb == 0 {
            f.push(0.0);
            continue;
        }
        let sq = king_bb.trailing_zeros() as u8;
        let file = (sq % 8) as f32;
        let rank = (sq / 8) as f32;
        f.push(((file - 3.5).abs() + (rank - 3.5).abs()) / 7.0);
    }

    // 11. Boot-Anzahl (+4 = 56)
    for c in Color::ALL {
        f.push(board.bb[c.idx()][PieceKind::Boat.idx()].count_ones() as f32 / 4.0);
    }

    debug_assert_eq!(f.len(), INPUT_SIZE, "Feature-Vektor hat falsche Länge");
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::board::Board;

    #[test]
    fn feature_vector_correct_size() {
        let b = Board::default();
        assert_eq!(extract(&b).len(), INPUT_SIZE);
    }

    #[test]
    fn all_features_in_range() {
        let b = Board::default();
        for (i, &v) in extract(&b).iter().enumerate() {
            assert!(v >= -1.0 && v <= 1.0, "Feature {i} = {v} out of range");
        }
    }

    #[test]
    fn different_positions_different_features() {
        assert_ne!(extract(&Board::default()), extract(&Board::empty()));
    }
}
