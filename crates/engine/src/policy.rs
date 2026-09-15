//! Policy-Kopf: ein kleines Netz, das Züge bewertet.
//!
//! # Warum überhaupt
//!
//! Die Baumsuche verteilt ihre Besuche bei den Budgets, die im Browser
//! realistisch sind, überwiegend nach dem **Prior** — gemessen: bei 800
//! Simulationen auf rund 30 Züge bekommt jeder Zug etwa 27 Besuche, und ein
//! kleineres Erkundungsgewicht macht das Spiel schlechter, nicht besser
//! (`c_puct` 0,25 gegen 1,5: −0,156 und −0,146 Platzwert). Der Prior ist damit
//! die Größe, an der die Spielstärke tatsächlich hängt.
//!
//! Heute liefert ihn [`crate::move_features::MoveModel`]: eine konditionale
//! Logit-Regression über 16 handgebaute Merkmale, 27,2 % Top-1 und 75,5 % im
//! Beam-6. Das ist die Komponente, die AlphaZero als gelerntes Netz hat und
//! dieses Projekt bisher nicht.
//!
//! # Die Form bleibt
//!
//! Bewertet wird weiter **je Zug einzeln**, und der Softmax läuft über die
//! legale Zugmenge. Das ist bewusst nicht die AlphaZero-Form (ein fester
//! Ausgabevektor über alle 64×64 Feldpaare): die hätte hier 1,05 Millionen
//! Gewichte allein in der letzten Schicht und wäre als JSON zweistellig in
//! Megabyte — für ein Frontend, das die Datei bei jedem Besuch lädt, keine
//! Option. Die Zug-für-Zug-Form kostet stattdessen einen Vorwärtslauf je Zug,
//! bleibt aber winzig.
//!
//! # Was hier geprüft wird
//!
//! Zuerst die billigste Frage: Holt ein nichtlineares Modell über **denselben**
//! Merkmalen etwas heraus? Wenn ja, lohnt der Ausbau des Merkmalssatzes; wenn
//! nein, ist die Linearität nicht die Grenze, sondern die Information.

use serde::{Deserialize, Serialize};

/// Was die Baumsuche von einem Prior braucht: ein Logit je Zug.
///
/// Zwei Umsetzungen — die lineare Logit-Regression
/// ([`crate::move_features::MoveModel`]) und das Netz hier. Über einen Trait,
/// damit die Arena beide in derselben Partie gegeneinander stellen kann; ohne
/// das ließe sich der Wechsel nicht messen, sondern nur behaupten.
pub trait MovePrior: Sync {
    fn logit(&self, f: &[f32]) -> f32;
}

impl MovePrior for PolicyNet {
    fn logit(&self, f: &[f32]) -> f32 { self.score(f) }
}

/// Breite der verborgenen Schicht.
///
/// Klein gehalten: das Netz läuft je Zug einmal, bei ~30 Zügen je Knoten und
/// tausenden Knoten je Suche. 32 Einheiten über 16 Eingaben sind 576 Gewichte —
/// gegenüber dem Bewertungsnetz mit 348.000 nicht der Rede wert.
pub const POLICY_HIDDEN: usize = 32;

/// Ein Zwei-Schicht-Netz mit einem Skalar als Ausgabe.
///
/// Die Ausgabe ist ein **Logit**, kein Wahrscheinlichkeitswert: normiert wird
/// erst über die legale Zugmenge, genau wie beim linearen Modell. Deshalb gibt
/// es auf der letzten Schicht keine Aktivierung und keinen Bias — eine
/// Konstante auf allen Zügen einer Stellung kürzt sich im Softmax ohnehin weg.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyNet {
    /// [hidden][eingang]
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    /// [hidden]
    pub w2: Vec<f32>,
    #[serde(default)]
    pub note: String,
}

impl PolicyNet {
    /// Netz mit He-Initialisierung.
    ///
    /// Deterministisch aus einem Seed, damit zwei Läufe vergleichbar sind —
    /// ein Trainingsergebnis, das von der Zufallsinitialisierung abhängt, ist
    /// als Messung wertlos.
    pub fn new(inputs: usize, seed: u64) -> Self {
        let std = (2.0 / inputs as f32).sqrt();
        let mut zustand = seed | 1;
        let mut zufall = move || {
            // Xorshift und Box-Muller — kein rand nötig, und über Plattformen
            // hinweg dasselbe Ergebnis.
            let mut u = || {
                zustand ^= zustand << 13;
                zustand ^= zustand >> 7;
                zustand ^= zustand << 17;
                (zustand >> 11) as f32 / (1u64 << 53) as f32
            };
            let (u1, u2) = (u().max(1e-7), u());
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };
        Self {
            w1: (0..POLICY_HIDDEN)
                .map(|_| (0..inputs).map(|_| zufall() * std).collect())
                .collect(),
            b1: vec![0.0; POLICY_HIDDEN],
            w2: (0..POLICY_HIDDEN).map(|_| zufall() * (2.0 / POLICY_HIDDEN as f32).sqrt()).collect(),
            note: String::new(),
        }
    }

    pub fn inputs(&self) -> usize {
        self.w1.first().map_or(0, |r| r.len())
    }

    pub fn param_count(&self) -> usize {
        self.inputs() * POLICY_HIDDEN + POLICY_HIDDEN + POLICY_HIDDEN
    }

    /// Logit für einen Zug.
    pub fn score(&self, f: &[f32]) -> f32 {
        let mut summe = 0.0;
        for (h, (zeile, &bias)) in self.w1.iter().zip(&self.b1).enumerate() {
            let a = (bias + zeile.iter().zip(f).map(|(&w, &x)| w * x).sum::<f32>()).max(0.0);
            if a > 0.0 { summe += a * self.w2[h]; }
        }
        summe
    }

    /// Wie [`score`], gibt aber die verborgenen Aktivierungen mit zurück — für
    /// die Rückwärtsrechnung im Training.
    pub fn forward(&self, f: &[f32]) -> (f32, Vec<f32>) {
        let mut h = vec![0.0f32; POLICY_HIDDEN];
        let mut summe = 0.0;
        for (i, (zeile, &bias)) in self.w1.iter().zip(&self.b1).enumerate() {
            let a = (bias + zeile.iter().zip(f).map(|(&w, &x)| w * x).sum::<f32>()).max(0.0);
            h[i] = a;
            summe += a * self.w2[i];
        }
        (summe, h)
    }

    pub fn validate(&self, inputs: usize) -> Result<(), String> {
        if self.w1.len() != POLICY_HIDDEN {
            return Err(format!("w1: {POLICY_HIDDEN} Zeilen erwartet, {} geladen", self.w1.len()));
        }
        if self.b1.len() != POLICY_HIDDEN || self.w2.len() != POLICY_HIDDEN {
            return Err("b1/w2 passen nicht zur verborgenen Schicht".into());
        }
        if self.w1.iter().any(|r| r.len() != inputs) {
            return Err(format!("w1: Zeilenbreite {inputs} erwartet"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zwei_netze_aus_demselben_seed_sind_gleich() {
        let a = PolicyNet::new(16, 42);
        let b = PolicyNet::new(16, 42);
        assert_eq!(a.w1, b.w1, "Initialisierung muss reproduzierbar sein");
        assert_eq!(a.w2, b.w2);
    }

    #[test]
    fn verschiedene_seeds_geben_verschiedene_netze() {
        let a = PolicyNet::new(16, 1);
        let b = PolicyNet::new(16, 2);
        assert_ne!(a.w1, b.w1);
    }

    /// Ein frisches Netz darf nicht überall dasselbe liefern — sonst wäre der
    /// Softmax darüber die Gleichverteilung und das Training startete blind.
    #[test]
    fn ein_frisches_netz_unterscheidet_eingaben() {
        let netz = PolicyNet::new(4, 7);
        let a = netz.score(&[1.0, 0.0, 0.0, 0.0]);
        let b = netz.score(&[0.0, 1.0, 0.0, 0.0]);
        assert!((a - b).abs() > 1e-6, "Ausgaben identisch: {a} und {b}");
    }

    #[test]
    fn forward_und_score_stimmen_ueberein() {
        let netz = PolicyNet::new(5, 99);
        let f = [0.3, -0.2, 1.0, 0.5, -0.8];
        let (s, h) = netz.forward(&f);
        assert!((s - netz.score(&f)).abs() < 1e-6);
        assert_eq!(h.len(), POLICY_HIDDEN);
        assert!(h.iter().all(|&x| x >= 0.0), "ReLU darf nicht negativ werden");
    }

    #[test]
    fn validate_erkennt_falsche_form() {
        let netz = PolicyNet::new(16, 1);
        assert!(netz.validate(16).is_ok());
        assert!(netz.validate(20).is_err());
    }
}
