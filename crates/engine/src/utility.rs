//! Rang-Nutzen-Transformation für die 4-Spieler-Bewertung.
//!
//! # Warum
//!
//! Chaturaji wird nach Punkten *platziert*: das `standings`-Feld der
//! chess.com-Exporte ist eine reine Punktordnung, und die Ratingänderung hängt
//! am Platz, nicht an der Punktdifferenz. Der Nutzen eines Spielers ist deshalb
//! nicht seine Punktzahl, sondern seine **erwartete Platzierung**.
//!
//! # Wie
//!
//! Aus dem Punktvektor `p` wird über paarweise Vergleiche die erwartete
//! Platzierung geschätzt und linear auf die Platzwertung abgebildet:
//!
//! ```text
//!     E[rang_i] = 1 + Σ_{j≠i} σ((p_j − p_i) / τ)
//!     u_i       = 1 − (2/3) · (E[rang_i] − 1)
//! ```
//!
//! `v = [1, 1/3, −1/3, −1]` ist die Platzwertung aus `standings_to_outcome`
//! (siehe `nnue::pgn_import`), linear im Rang interpoliert.
//!
//! # Zwei nützliche Eigenschaften
//!
//! * **Σ u_i ≡ 0, exakt.** Für jedes Paar gilt σ(x) + σ(−x) = 1, es gibt sechs
//!   Paare, also Σ E[rang] = 4 + 6 = 10 = 1+2+3+4. Damit hat die Max^n-Suche
//!   eine exakte Summenschranke für Shallow Pruning (Korf 1991) — vorher war
//!   die Summe der Rohbewertung unbeschränkt und die Schranke geraten.
//! * **Streng monoton im eigenen p.** Steigt `p_i` bei festen `p_j`, steigt
//!   `u_i` strikt. Die Suche verliert also keine Ordnungsinformation.
//!
//! Das Risikoverhalten fällt als Nebeneffekt korrekt aus: wer klar führt, sitzt
//! im flachen Teil der Kurve und bevorzugt sichere Züge; wer aussichtslos
//! Vierter ist, ebenfalls flach und sucht deshalb Varianz; wer auf Platz 2/3
//! einen Punkt Abstand hat, steht im steilen Teil und kämpft um jeden Bauern.

use chaturaji_core::board::Board;
use chaturaji_core::piece::{Color, PieceKind};

/// Ganzzahlige Skalierung: `u ∈ [−1, 1]` → `[−UTIL_SCALE, UTIL_SCALE]`.
/// Die Suche rechnet in dieser Einheit, nicht mehr in Centipawns.
pub const UTIL_SCALE: i32 = 10_000;

/// Temperatur-Untergrenze (in Punkten). Verhindert eine Stufenfunktion im
/// Endspiel, wenn kaum noch Material für Punktverschiebungen übrig ist.
const TAU_MIN: f32 = 0.75;

/// Temperatur-Steigung. τ soll die Standardabweichung der noch möglichen
/// Punktverschiebung sein. Das gesamte noch schlagbare Material ist eine
/// direkte Obergrenze dafür, und die Streuung einer Summe unabhängiger
/// Beiträge wächst mit deren Wurzel — daher `√(Restmaterial)`.
const TAU_K: f32 = 0.9;

#[inline]
fn sigma(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Punktvektor → Nutzenvektor mit Σ = 0 und Werten in `[−1, 1]`.
///
/// `tau` ist die Restunsicherheit in Punkten: klein = Punkte stehen fest
/// (harte Rangordnung), groß = alles noch offen (Nutzen ≈ 0 für alle).
pub fn rank_utility(p: [f32; 4], tau: f32) -> [f32; 4] {
    let tau = tau.max(0.05);
    std::array::from_fn(|i| {
        let expected_worse: f32 = (0..4)
            .filter(|&j| j != i)
            .map(|j| sigma((p[j] - p[i]) / tau))
            .sum();
        1.0 - (2.0 / 3.0) * expected_worse
    })
}

/// Gesamtes noch auf dem Brett stehendes Schlagmaterial (in Punkten).
/// Figuren eliminierter Spieler zählen nicht: sie bringen beim Schlagen
/// keine Punkte mehr (siehe `Board::apply_move`).
pub fn remaining_capture_value(board: &Board) -> i32 {
    let mut total = 0;
    for c in Color::ALL {
        if !board.active[c.idx()] {
            continue;
        }
        for k in PieceKind::ALL {
            total += board.pieces(c, k).count_ones() as i32 * k.capture_value();
        }
    }
    total
}

/// Restunsicherheit der Endpunkte, in Punkten.
///
/// Startstellung: 4 × 20 = 80 Punkte Material → τ ≈ 8.8. Ein Vorsprung von
/// 9 Punkten ist im ersten Zug also gerade mal „ein Sigma" wert. Im Endspiel
/// mit 10 Punkten Restmaterial fällt τ auf ≈ 3.6, die Rangordnung härtet aus.
pub fn tau_for(board: &Board) -> f32 {
    TAU_MIN + TAU_K * (remaining_capture_value(board).max(0) as f32).sqrt()
}

/// Rohbewertung (Centipunkte) → Nutzen (`UTIL_SCALE`-Einheiten).
///
/// Die Rohbewertung ist als *geschätzte Endpunktzahl × 100* kalibriert, also
/// teilt die Umrechnung durch 100.
pub fn to_utility(raw: [i32; 4], board: &Board) -> [i32; 4] {
    let points: [f32; 4] = std::array::from_fn(|i| raw[i] as f32 / 100.0);
    let u = rank_utility(points, tau_for(board));
    std::array::from_fn(|i| (u[i] * UTIL_SCALE as f32).round() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(u: [f32; 4]) -> f32 {
        u.iter().sum()
    }

    #[test]
    fn utility_sums_to_zero() {
        for p in [
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 5.0, 3.0, 1.0],
            [-4.0, 22.0, 7.5, 7.5],
            [100.0, 0.0, 0.0, -100.0],
        ] {
            for tau in [0.1, 1.0, 8.0, 50.0] {
                assert!(
                    sum(rank_utility(p, tau)).abs() < 1e-4,
                    "Σu ≠ 0 für p={p:?}, τ={tau}"
                );
            }
        }
    }

    #[test]
    fn equal_points_give_zero_utility() {
        for u in rank_utility([7.0; 4], 8.0) {
            assert!(u.abs() < 1e-5, "Gleichstand muss Nutzen 0 ergeben, war {u}");
        }
    }

    #[test]
    fn utility_is_bounded_by_place_values() {
        // Klarer Sieger → u ≈ 1, klarer Letzter → u ≈ −1.
        let u = rank_utility([50.0, 0.0, 0.0, -50.0], 1.0);
        assert!(u[0] > 0.99, "Sieger sollte ≈ 1 sein, war {}", u[0]);
        assert!(u[3] < -0.99, "Letzter sollte ≈ −1 sein, war {}", u[3]);
        for v in u {
            assert!((-1.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn utility_is_strictly_monotone_in_own_points() {
        // Kernvoraussetzung dafür, dass die Suche keine Ordnung verliert.
        let base = rank_utility([5.0, 5.0, 5.0, 5.0], 8.0)[0];
        let more = rank_utility([5.1, 5.0, 5.0, 5.0], 8.0)[0];
        assert!(more > base, "u muss in p_i streng wachsen: {base} → {more}");
    }

    #[test]
    fn second_place_gradient_exceeds_hopeless_gradient() {
        // Ein Punkt mehr muss im umkämpften Mittelfeld mehr wert sein als für
        // einen abgeschlagenen Vierten — genau das kann eine lineare
        // Bewertung nicht ausdrücken.
        let tau = 3.0;
        let contested = rank_utility([20.0, 10.0, 10.0, 2.0], tau)[1]
            - rank_utility([20.0, 9.0, 10.0, 2.0], tau)[1];
        let hopeless = rank_utility([20.0, 18.0, 17.0, 2.0], tau)[3]
            - rank_utility([20.0, 18.0, 17.0, 1.0], tau)[3];
        assert!(
            contested > hopeless,
            "umkämpfter Punkt ({contested}) muss mehr wiegen als aussichtsloser ({hopeless})"
        );
    }

    #[test]
    fn tau_shrinks_as_material_disappears() {
        use chaturaji_core::board::{bit, sq};
        let start = tau_for(&Board::default());

        let mut endgame = Board::empty();
        for c in Color::ALL {
            endgame.bb[c.idx()][PieceKind::King.idx()] = bit(sq(c.idx() as u8, 0));
        }
        let late = tau_for(&endgame);

        assert!(
            late < start,
            "τ muss mit schwindendem Material fallen: Start {start}, Endspiel {late}"
        );
    }
}
