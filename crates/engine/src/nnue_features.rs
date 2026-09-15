//! NNUE Input-Kodierung: Piece-Square (PS) via Bitboards **plus** Spielzustand.
//!
//! # Zwei Blöcke
//!
//! * **Binär, dünn besetzt** — die 1280 PS-Features. Das Brett liegt bereits als
//!   20 × u64 Bitboards vor (`bb[4][5]`, 4 Spieler × 5 Figurtypen):
//!
//!   ```text
//!   feature_idx = color * (5 * 64) + piece_kind * 64 + square
//!   ```
//!
//!   Sie werden NICHT als `Vec<usize>` extrahiert — der Forward-Pass iteriert
//!   direkt über die Bits. Das spart eine Heap-Allokation.
//!
//! * **Reellwertig, dicht** — 13 Features für den Spielzustand.
//!
//! # Warum der zweite Block nötig ist
//!
//! Das Netz soll die Endplatzierung vorhersagen, kannte aber bis zuletzt nur
//! die Figurenstellung. Damit sind zwei Stellungen mit identischen Figuren, aber
//! Punkteständen 20-5-5-5 und 5-5-5-20 für das Netz ununterscheidbar — obwohl
//! ihre Ausgänge nichts miteinander zu tun haben. Punktestand, Zugrecht,
//! Eliminierungen und Spielphase sind keine Feinheiten, sondern der Kern der
//! Zielgröße; ohne sie ist die Aufgabe schlicht nicht lösbar.

use chaturaji_core::board::Board;
use chaturaji_core::piece::{Color, PieceKind};

/// Binäre Piece-Square-Features.
pub const PIECE_FEATURES: usize = 4 * 5 * 64; // = 1280

/// Dichte Spielzustands-Features (siehe [`dense_features`]).
pub const DENSE_FEATURES: usize = 13;

/// Gesamte Eingabebreite von L1.
pub const INPUT_SIZE: usize = PIECE_FEATURES + DENSE_FEATURES; // = 1293

/// Punkteskala: typische Endstände liegen bei 15–25 Punkten, ein Wert von 20
/// bildet also ungefähr auf 1.0 ab.
const SCORE_SCALE: f32 = 20.0;

/// Schlagmaterial der Startstellung (4 Spieler × 20 Punkte) als Bezugsgröße
/// für die Spielphase.
const START_CAPTURE_VALUE: f32 = 80.0;

/// Berechnet den Feature-Index für eine einzelne Figur.
/// Wird für Tests und Debug-Ausgaben verwendet.
#[inline]
pub fn feature_index(color_idx: usize, kind_idx: usize, sq: usize) -> usize {
    color_idx * (5 * 64) + kind_idx * 64 + sq
}

/// Iteriert über alle aktiven binären Features in einem `bb[4][5]`-Array
/// und ruft `f(feature_idx)` für jede gesetzte Figur auf.
#[inline]
pub fn for_each_feature(bb: &[[u64; 5]; 4], mut f: impl FnMut(usize)) {
    for c in 0..4 {
        for k in 0..5 {
            let mut bits = bb[c][k];
            let base = c * (5 * 64) + k * 64;
            while bits != 0 {
                let sq = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                f(base + sq);
            }
        }
    }
}

/// Zählt die aktiven binären Features (= Anzahl Figuren auf dem Brett).
#[inline]
pub fn count_features(bb: &[[u64; 5]; 4]) -> usize {
    let mut n = 0;
    for c in 0..4 {
        for k in 0..5 {
            n += bb[c][k].count_ones() as usize;
        }
    }
    n
}

/// Dichte Spielzustands-Features, in dieser Reihenfolge:
///
/// | Index | Bedeutung                                             |
/// |-------|-------------------------------------------------------|
/// | 0–3   | gebuchte Punkte je Spieler / 20                       |
/// | 4–7   | Spieler noch im Spiel (1/0)                            |
/// | 8–11  | Zugrecht, One-Hot                                      |
/// | 12    | Spielphase: verbleibendes Schlagmaterial / 80          |
///
/// Die Indizes hier sind relativ zum dichten Block; im Netz liegen sie ab
/// [`PIECE_FEATURES`].
pub fn dense_features(board: &Board) -> [f32; DENSE_FEATURES] {
    let mut f = [0.0f32; DENSE_FEATURES];

    let mut remaining = 0i32;
    for c in Color::ALL {
        let ci = c.idx();
        f[ci] = board.scores.get(c) as f32 / SCORE_SCALE;
        f[4 + ci] = if board.active[ci] { 1.0 } else { 0.0 };
        if board.active[ci] {
            for k in PieceKind::ALL {
                remaining += board.pieces(c, k).count_ones() as i32 * k.capture_value();
            }
        }
    }

    f[8 + board.to_move.idx()] = 1.0;
    f[12] = remaining as f32 / START_CAPTURE_VALUE;

    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::board::Board;

    #[test]
    fn starting_position_has_32_features() {
        let b = Board::default();
        assert_eq!(count_features(&b.bb), 32);
    }

    #[test]
    fn feature_indices_in_range() {
        let b = Board::default();
        for_each_feature(&b.bb, |idx| {
            assert!(idx < PIECE_FEATURES, "Index {idx} außerhalb [0, {PIECE_FEATURES})");
        });
    }

    #[test]
    fn empty_board_has_no_features() {
        let b = Board::empty();
        assert_eq!(count_features(&b.bb), 0);
    }

    #[test]
    fn feature_indices_are_unique() {
        let b = Board::default();
        let mut seen = vec![false; PIECE_FEATURES];
        for_each_feature(&b.bb, |idx| {
            assert!(!seen[idx], "doppelter Feature-Index {idx}");
            seen[idx] = true;
        });
    }

    #[test]
    fn dense_features_encode_the_start_position() {
        let b = Board::default();
        let f = dense_features(&b);
        assert_eq!(&f[0..4], &[0.0; 4], "am Anfang hat niemand Punkte");
        assert_eq!(&f[4..8], &[1.0; 4], "am Anfang sind alle im Spiel");
        assert_eq!(f[8], 1.0, "Rot ist am Zug");
        assert_eq!(&f[9..12], &[0.0; 3]);
        assert!((f[12] - 1.0).abs() < 1e-6, "volle Spielphase, war {}", f[12]);
    }

    /// Der eigentliche Grund für den dichten Block: zwei Stellungen mit
    /// identischen Figuren, aber gespiegeltem Punktestand müssen sich in der
    /// Eingabe unterscheiden.
    #[test]
    fn dense_features_separate_mirrored_scores() {
        let mut a = Board::default();
        a.scores.add(Color::Red, 20);
        let mut b = Board::default();
        b.scores.add(Color::Green, 20);

        assert_eq!(a.bb, b.bb, "Testvoraussetzung: gleiche Figurenstellung");
        assert_ne!(dense_features(&a), dense_features(&b),
            "Punktestand muss die Eingabe unterscheidbar machen");
    }

    #[test]
    fn dense_features_track_elimination_and_phase() {
        let mut b = Board::default();
        let full_phase = dense_features(&b)[12];

        b.active[Color::Blue.idx()] = false;
        let f = dense_features(&b);
        assert_eq!(f[4 + Color::Blue.idx()], 0.0, "Blau ist raus");
        assert!(f[12] < full_phase,
            "Material eines eliminierten Spielers zählt nicht mehr zur Phase");
    }
}
