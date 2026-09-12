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

use serde::{Deserialize, Serialize};

use chaturaji_core::board::Board;
use chaturaji_core::piece::{Color, PieceKind};

/// Binäre Piece-Square-Features.
pub const PIECE_FEATURES: usize = 4 * 5 * 64; // = 1280

/// König-Bezugs-Merkmale: (Besitzverhältnis, Figurenart, Abstand, Geometrie).
pub const KING_FEATURES: usize = 4 * 5 * 3 * 4; // = 240

/// Dichte Spielzustands-Features (siehe [`dense_features`]).
pub const DENSE_FEATURES: usize = 13;

/// Eingabebreite des alten Merkmalssatzes.
pub const INPUT_SIZE: usize = PIECE_FEATURES + DENSE_FEATURES; // = 1293

/// Welche binären Merkmale ein Netz benutzt.
///
/// # Warum das ein Laufzeitwert ist und keine Konstante
///
/// Ein neuer Merkmalssatz ist nur dann eine Verbesserung, wenn er gegen den
/// alten gewinnt — und dafür müssen beide Netze **im selben Prozess** spielen
/// können. Als Compile-Zeit-Konstante wäre der Vergleich unmöglich; dieselbe
/// Lehre wie bei `BeamOrder` und dem Suchverfahren in der Arena.
///
/// Der Wert steht in der Gewichtsdatei. Alte Dateien haben das Feld nicht und
/// werden als [`FeatureSet::Legacy`] gelesen — deshalb ist das die Vorgabe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FeatureSet {
    /// 1280 Piece-Square + 13 dichte = 1293 Eingaben.
    #[default]
    #[serde(rename = "legacy")]
    Legacy,
    /// Zusätzlich 240 König-Bezugs-Merkmale = 1533 Eingaben.
    ///
    /// Der Grund für den Umbau: mit reinen Piece-Square-Merkmalen kann der
    /// Akkumulator „diese Figur bedroht jenen König" gar nicht darstellen. Zwei
    /// Stellungen, in denen dasselbe Boot einmal neben und einmal fern vom
    /// gegnerischen König steht, unterscheiden sich für das Netz in genau zwei
    /// von 1280 Bits, und nichts sagt ihm, dass diese Bits etwas miteinander zu
    /// tun haben. Genau das nennt die Roadmap von `Foobork/flutteraji` als
    /// Hauptursache ihres eigenen Plateaus.
    #[serde(rename = "king")]
    KingRelations,
}

impl FeatureSet {
    /// Aus einem CLI-Wort.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "legacy" | "alt" => Some(Self::Legacy),
            "king" | "koenig" => Some(Self::KingRelations),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self { Self::Legacy => "legacy", Self::KingRelations => "king" }
    }

    /// Zahl der binären Merkmale — zugleich der Versatz des dichten Blocks.
    #[inline]
    pub fn binary_features(self) -> usize {
        match self {
            Self::Legacy        => PIECE_FEATURES,
            Self::KingRelations => PIECE_FEATURES + KING_FEATURES,
        }
    }

    /// Gesamte Eingabebreite von L1.
    #[inline]
    pub fn input_size(self) -> usize {
        self.binary_features() + DENSE_FEATURES
    }

    /// Iteriert über alle aktiven binären Merkmale.
    #[inline]
    pub fn for_each(self, bb: &[[u64; 5]; 4], mut f: impl FnMut(usize)) {
        for_each_feature(bb, &mut f);
        if self == Self::KingRelations {
            for_each_king_feature(bb, &mut f);
        }
    }
}

/// Größter Chebyshev-Abstand, bis zu dem ein König-Bezug erfasst wird.
///
/// Weiter entfernte Figuren bleiben außen vor. Das ist kein Sparzwang, sondern
/// eine Entscheidung über Nutzen und Kosten: die Zahl der aktiven Merkmale
/// bestimmt direkt, was ein Forward-Pass kostet, und eine Figur sechs Felder
/// vom König entfernt bedroht ihn in aller Regel nicht. Wo sie doch zählt,
/// steht sie weiterhin im Piece-Square-Block.
pub const KING_MAX_DIST: i32 = 3;

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
pub fn for_each_feature(bb: &[[u64; 5]; 4], f: &mut impl FnMut(usize)) {
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

/// Index eines König-Bezugs-Merkmals, relativ zum Beginn des Blocks.
///
/// * `relation` — Besitzer der Figur minus Besitzer des Königs, modulo 4.
///   0 heißt „mein eigener König", 1..3 sind die Sitznachbarn. Die Kodierung
///   ist damit drehinvariant: dieselbe Konstellation zählt für Rot und für Gelb
///   auf denselben Index, und das Netz muss sie nur einmal lernen.
/// * `kind` — Figurenart 0..4.
/// * `bucket` — Chebyshev-Abstand 1..3, abgelegt als 0..2.
/// * `geometry` — auf welcher Linie die Figur zum König steht.
#[inline]
pub fn king_feature_index(relation: usize, kind: usize, bucket: usize, geometry: usize) -> usize {
    ((relation * 5 + kind) * 3 + bucket) * 4 + geometry
}

/// Geometrische Beziehung zweier Felder: 0 gerade Linie, 1 Diagonale,
/// 2 Springerabstand, 3 sonst.
///
/// Die ersten drei Klassen sind genau die Zugmuster des Spiels — Boot gerade,
/// Bishop diagonal, Springer im Sprung. Eine Figur, die auf einer dieser Linien
/// zum König steht, kann ihn bedrohen; „sonst" kann es nie.
#[inline]
fn geometry_class(dx: i32, dy: i32) -> usize {
    let (ax, ay) = (dx.abs(), dy.abs());
    if dx == 0 || dy == 0 { 0 }
    else if ax == ay      { 1 }
    else if (ax, ay) == (1, 2) || (ax, ay) == (2, 1) { 2 }
    else                  { 3 }
}

/// Iteriert über die König-Bezugs-Merkmale.
///
/// Für jede Figur und jeden **noch stehenden** König im Umkreis von
/// [`KING_MAX_DIST`] ein Merkmal. Ein geschlagener König hat kein Bit mehr im
/// Bitboard und fällt von selbst weg — `board.active` wird nicht gebraucht, was
/// wichtig ist, weil an dieser Stelle nur die Bitboards vorliegen.
#[inline]
pub fn for_each_king_feature(bb: &[[u64; 5]; 4], f: &mut impl FnMut(usize)) {
    const KING: usize = 4;   // PieceKind::King als Index

    // Königsfelder einmal einsammeln statt je Figur neu suchen.
    let mut kings: [(usize, i32, i32); 4] = [(0, 0, 0); 4];
    let mut n_kings = 0;
    for (c, bbc) in bb.iter().enumerate() {
        let bits = bbc[KING];
        if bits == 0 { continue; }
        let sq = bits.trailing_zeros() as i32;
        kings[n_kings] = (c, sq & 7, sq >> 3);
        n_kings += 1;
    }
    if n_kings == 0 { return; }

    for c in 0..4 {
        for k in 0..5 {
            let mut bits = bb[c][k];
            while bits != 0 {
                let sq = bits.trailing_zeros() as i32;
                bits &= bits - 1;
                let (px, py) = (sq & 7, sq >> 3);

                for &(kc, kx, ky) in &kings[..n_kings] {
                    let (dx, dy) = (px - kx, py - ky);
                    let cheb = dx.abs().max(dy.abs());
                    // Abstand 0 ist der König selbst, darüber hinaus zu weit.
                    if cheb == 0 || cheb > KING_MAX_DIST { continue; }
                    let relation = (c + 4 - kc) % 4;
                    f(PIECE_FEATURES + king_feature_index(
                        relation, k, (cheb - 1) as usize, geometry_class(dx, dy)));
                }
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
        for_each_feature(&b.bb, &mut |idx| {
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
        for_each_feature(&b.bb, &mut |idx| {
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

    // ─── König-Bezugs-Merkmale ────────────────────────────────────────────

    fn king_indices(b: &Board) -> Vec<usize> {
        let mut v = Vec::new();
        for_each_king_feature(&b.bb, &mut |i| v.push(i));
        v
    }

    /// Alle Indizes müssen im eigenen Block liegen. Ein Ausrutscher nach unten
    /// würde stillschweigend ein Piece-Square-Gewicht verändern, einer nach
    /// oben in den dichten Block greifen.
    #[test]
    fn king_feature_indices_stay_inside_their_block() {
        let b = Board::default();
        for i in king_indices(&b) {
            assert!((PIECE_FEATURES..PIECE_FEATURES + KING_FEATURES).contains(&i),
                    "Index {i} außerhalb des König-Blocks");
        }
    }

    /// Ohne König auf dem Brett darf kein einziges Merkmal entstehen — der
    /// Fall kommt im Self-Play vor, wenn alle Könige geschlagen sind.
    #[test]
    fn a_board_without_kings_has_no_king_features() {
        let mut b = Board::default();
        for c in 0..4 { b.bb[c][4] = 0; }
        assert!(king_indices(&b).is_empty());
    }

    /// Der eigentliche Zweck: dieselbe Figur nah und fern vom König muss
    /// verschiedene Merkmale ergeben. Genau das konnte der alte Satz nicht.
    #[test]
    fn distance_to_the_king_changes_the_encoding() {
        let mut nah = Board::empty();
        nah.bb[0][4] = 1u64 << 0;        // Roter König auf a1
        nah.bb[1][3] = 1u64 << 9;        // Blaues Boot auf b2 — Abstand 1
        let mut fern = Board::empty();
        fern.bb[0][4] = 1u64 << 0;
        fern.bb[1][3] = 1u64 << 45;      // b6 — weit weg

        let a = king_indices(&nah);
        let b = king_indices(&fern);
        assert!(!a.is_empty(), "nah am König muss ein Merkmal geben");
        assert!(b.is_empty(), "jenseits von KING_MAX_DIST darf keines entstehen");
        assert_ne!(a, b);
    }

    /// Die Kodierung ist drehinvariant: dieselbe Konstellation, um einen Sitz
    /// weitergedreht, muss denselben Index ergeben. Ohne das müsste das Netz
    /// jedes Muster viermal lernen.
    #[test]
    fn the_encoding_is_invariant_under_rotating_the_seats() {
        let mut a = Board::empty();
        a.bb[0][4] = 1u64 << 27;         // Roter König
        a.bb[1][3] = 1u64 << 28;         // Blaues Boot daneben
        let mut b = Board::empty();
        b.bb[1][4] = 1u64 << 27;         // Blauer König
        b.bb[2][3] = 1u64 << 28;         // Gelbes Boot daneben

        assert_eq!(king_indices(&a), king_indices(&b),
                   "gleiche Konstellation, anderer Sitz → gleicher Index");
    }

    /// Die Geometrieklassen müssen die Zugmuster des Spiels treffen.
    #[test]
    fn geometry_classes_match_the_movement_patterns() {
        assert_eq!(geometry_class(0, 3), 0, "gerade Linie (Boot)");
        assert_eq!(geometry_class(2, 0), 0);
        assert_eq!(geometry_class(2, 2), 1, "Diagonale (Bishop)");
        assert_eq!(geometry_class(1, 2), 2, "Springerabstand");
        assert_eq!(geometry_class(2, 1), 2);
        assert_eq!(geometry_class(1, 3), 3, "keines davon");
    }

    /// Der Merkmalssatz bestimmt Breite und Versatz; beides muss zusammenpassen,
    /// sonst schreibt der dichte Block in den König-Block.
    #[test]
    fn the_feature_set_reports_consistent_sizes() {
        assert_eq!(FeatureSet::Legacy.binary_features(), PIECE_FEATURES);
        assert_eq!(FeatureSet::Legacy.input_size(), INPUT_SIZE);
        assert_eq!(FeatureSet::KingRelations.binary_features(), PIECE_FEATURES + KING_FEATURES);
        assert_eq!(FeatureSet::KingRelations.input_size(), PIECE_FEATURES + KING_FEATURES + DENSE_FEATURES);
        assert!(FeatureSet::KingRelations.input_size() > FeatureSet::Legacy.input_size());
    }

    /// `for_each` muss beim alten Satz exakt das Alte liefern — sonst wären
    /// vorhandene Netze nicht mehr dasselbe Netz.
    #[test]
    fn the_legacy_set_is_unchanged() {
        let b = Board::default();
        let mut alt = Vec::new();
        for_each_feature(&b.bb, &mut |i| alt.push(i));
        let mut ueber_set = Vec::new();
        FeatureSet::Legacy.for_each(&b.bb, |i| ueber_set.push(i));
        assert_eq!(alt, ueber_set);
    }

    /// Wie viele Merkmale kostet der neue Satz zusätzlich? Die Zahl bestimmt
    /// direkt den Aufwand je Forward-Pass, deshalb steht sie hier fest.
    #[test]
    fn the_start_position_stays_affordable() {
        let b = Board::default();
        let mut n = 0;
        FeatureSet::KingRelations.for_each(&b.bb, |_| n += 1);
        assert_eq!(count_features(&b.bb), 32, "32 Figuren");
        assert!(n <= 96, "höchstens dreifache Kosten, waren {n}");
        assert!(n > 32, "der neue Satz muss mehr liefern als der alte, war {n}");
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
