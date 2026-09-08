//! Die eine Zielgröße des Netzes: **erwartete Platzierung**.
//!
//! # Warum es dieses Modul gibt
//!
//! Es gab vorher zwei unvereinbare Zielkodierungen nebeneinander:
//!
//! * `selfplay::final_targets` normalisierte die Endpunkte min-max auf [−1, 1];
//! * `pgn_import::standings_to_outcome` benutzte feste Platzwerte.
//!
//! Ein Netz, das auf dem einen trainiert und als das andere gelesen wird, ist
//! fehlkalibriert. Schlimmer noch: die Min-Max-Normalisierung zerstört die
//! Skala — ein Ausgang 23-19-15-15 und einer mit 8-7-6-6 landen auf demselben
//! Vektor, obwohl der eine deutlich klarer entschieden ist als der andere.
//!
//! Beide Wege benutzen jetzt [`place_values`]. Damit stimmt die Ausgabe des
//! Netzes mit der Einheit überein, in der die Suche rechnet
//! (`chaturaji_engine::utility`), und Netz- und Handbewertung sind an einem
//! Blattknoten direkt vergleichbar.

/// Platzwertung: Platz 1 → 1, Platz 2 → ½, Platz 3 → −½, Platz 4 → −1.
///
/// Die Abstände sind bewusst ungleich. Maßgeblich ist, was das Spiel
/// tatsächlich belohnt, und das ist die Glicko-Wertung: Platz 1 und 2 gewinnen
/// Rating, Platz 3 und 4 verlieren es. An allen 1.977 Partien aus `game_data/`
/// gemessen:
///
/// | Platz | Ø Rating-Änderung | Median | Anteil positiv |
/// |-------|-------------------|--------|----------------|
/// |   1   |            +6,42  | +11,0  |          66 %  |
/// |   2   |            +2,48  |  +5,8  |          65 %  |
/// |   3   |            −1,97  |  −5,4  |          38 %  |
/// |   4   |            −6,83  | −11,2  |          32 %  |
///
/// Auf ±1 normiert ergibt der Median [1, 0,52, −0,49, −1] — daher die Halben.
/// Der Mittelwert spräche für ⅓, aber Glicko-Änderungen hängen an Gegnerstärke
/// und Rating-Unsicherheit, und einzelne Ausreißer ziehen ihn nach außen; der
/// Median beschreibt den typischen Fall besser.
///
/// Die Summe bleibt 0, das Netz sagt also weiterhin eine Verteilung ohne
/// Ablage vorher.
pub const PLACE_VALUE: [f32; 4] = [1.0, 0.5, -0.5, -1.0];

/// Endpunkte → Platzwertung je Spieler.
///
/// Punktgleiche Spieler teilen sich den Mittelwert der Plätze, die sie
/// gemeinsam belegen. Das ist bewusst so: chess.com vergibt bei Gleichstand
/// zwar trotzdem verschiedene Plätze (in `100000255.json` etwa werden 15 und
/// 15 zu Platz 3 und 4), aber dieser Tiebreak ist keine Funktion der Stellung.
/// Würde man ihn als Ziel vorgeben, lernte das Netz, Rauschen vorherzusagen.
///
/// Die Summe bleibt in jedem Fall 0.
pub fn place_values(points: [i32; 4]) -> [f32; 4] {
    std::array::from_fn(|i| {
        let better = (0..4).filter(|&j| points[j] > points[i]).count();
        let tied   = (0..4).filter(|&j| points[j] == points[i]).count();
        // Belegt die 0-basierten Plätze `better .. better + tied`.
        (better..better + tied).map(|r| PLACE_VALUE[r]).sum::<f32>() / tied as f32
    })
}

/// Wie [`place_values`], aber aus einer bereits fertigen Platzierung
/// (1-basiert, wie das `standings`-Feld der chess.com-Exporte).
///
/// Nur als Notnagel gedacht, wenn keine Punkte vorliegen — der Punkteweg ist
/// vorzuziehen, weil er Gleichstände korrekt behandelt.
pub fn place_values_from_standings(standings: [u8; 4]) -> [f32; 4] {
    std::array::from_fn(|i| match standings[i] {
        1..=4 => PLACE_VALUE[standings[i] as usize - 1],
        _     => 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(v: [f32; 4]) -> f32 { v.iter().sum() }

    #[test]
    fn distinct_points_map_to_the_four_places() {
        // 23 / 19 / 15 / 5 → Plätze 1 / 2 / 3 / 4.
        let v = place_values([23, 19, 15, 5]);
        assert_eq!(v, PLACE_VALUE);
    }

    #[test]
    fn ties_share_the_average_place() {
        // Der echte Ausgang aus game_data/100000255.json: 15 / 23 / 19 / 15.
        // Spieler 0 und 3 sind punktgleich und teilen Platz 3 und 4.
        let v = place_values([15, 23, 19, 15]);
        let shared = (PLACE_VALUE[2] + PLACE_VALUE[3]) / 2.0;
        assert!((v[0] - shared).abs() < 1e-6, "erwartet {shared}, war {}", v[0]);
        assert!((v[3] - shared).abs() < 1e-6, "erwartet {shared}, war {}", v[3]);
        assert!((v[1] - PLACE_VALUE[0]).abs() < 1e-6, "Spieler 1 ist Erster");
        assert!((v[2] - PLACE_VALUE[1]).abs() < 1e-6, "Spieler 2 ist Zweiter");
    }

    /// Platz 1 und 2 müssen positiv sein, Platz 3 und 4 negativ — das ist die
    /// Aussage der Glicko-Wertung und der Grund für die Vorzeichen.
    #[test]
    fn top_two_are_rewarded() {
        let v = place_values([20, 19, 18, 15]);
        assert!(v[0] > 0.0 && v[1] > 0.0, "Platz 1 und 2 gewinnen: {v:?}");
        assert!(v[2] < 0.0 && v[3] < 0.0, "Platz 3 und 4 verlieren: {v:?}");
        assert!(v[0] > v[1], "der erste Platz ist mehr wert als der zweite");
    }

    #[test]
    fn sum_is_always_zero() {
        for p in [
            [23, 19, 15, 5],
            [15, 23, 19, 15],
            [7, 7, 7, 7],
            [0, 0, 1, 1],
            [-3, 40, 0, 0],
        ] {
            assert!(sum(place_values(p)).abs() < 1e-5, "Σ ≠ 0 für {p:?}");
        }
    }

    #[test]
    fn a_four_way_tie_is_neutral() {
        for v in place_values([9, 9, 9, 9]) {
            assert!(v.abs() < 1e-6, "Gleichstand muss 0 ergeben, war {v}");
        }
    }

    /// Anders als die alte Min-Max-Normalisierung darf die Kodierung nicht
    /// davon abhängen, wie weit die Punkte auseinanderliegen — nur von der
    /// Reihenfolge. Zwei Partien mit gleicher Rangfolge sind dasselbe Ergebnis.
    #[test]
    fn only_the_ordering_matters_not_the_spread() {
        assert_eq!(place_values([23, 19, 15, 5]), place_values([8, 7, 6, 1]));
    }

    #[test]
    fn standings_path_matches_the_points_path() {
        // Ohne Gleichstände müssen beide Wege dasselbe liefern.
        // points [15, 23, 19, 5] → standings [3, 1, 2, 4].
        assert_eq!(
            place_values([15, 23, 19, 5]),
            place_values_from_standings([3, 1, 2, 4]),
        );
    }
}
