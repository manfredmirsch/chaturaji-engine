//! Evaluate-Schnittstelle für die NNUE-Engine.
//!
//! `evaluate(board)` ist kompatibel zur Signatur in `crates/engine/src/eval.rs`
//! und kann daher als Drop-in-Ersatz verwendet werden.

use std::sync::OnceLock;
use chaturaji_core::board::Board;
use crate::network::NnueNetwork;

/// Skalierungsfaktor: die tanh-Ausgabe [-1, 1] des Netzes ist bereits eine
/// erwartete Platzierung (Platz 1 → 1, Platz 4 → −1), also dieselbe Größe, die
/// `chaturaji_engine::eval::evaluate` liefert. Deshalb wird hier exakt mit
/// `UTIL_SCALE` skaliert — Netz- und Handbewertung sind sonst nicht
/// vergleichbar, und genau das war vorher der Fall (5000 gegen 10 000).
const EVAL_SCALE: f32 = chaturaji_engine::utility::UTIL_SCALE as f32;

static GLOBAL_NET: OnceLock<NnueNetwork> = OnceLock::new();

/// Lädt Gewichte aus einer JSON-Datei in die globale Netz-Instanz.
/// Muss vor dem ersten `evaluate()`-Aufruf aufgerufen werden.
pub fn load_weights(path: &str) -> Result<(), String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("Datei '{}' nicht lesbar: {}", path, e))?;
    let mut net: NnueNetwork = serde_json::from_str(&json)
        .map_err(|e| format!("JSON-Deserialisierung fehlgeschlagen: {}", e))?;
    // Ältere Exporte kennen den dichten Eingabeblock noch nicht; ohne das
    // Auffüllen liefe der Forward-Pass in einen Index-Überlauf.
    net.ensure_input_size();
    GLOBAL_NET.set(net).map_err(|_| "Gewichte wurden bereits geladen".to_string())?;
    Ok(())
}

/// Gibt `true` zurück wenn bereits Gewichte geladen sind.
pub fn weights_loaded() -> bool {
    GLOBAL_NET.get().is_some()
}

/// Bewertet eine Stellung mit dem NNUE und gibt einen Score-Vektor zurück.
/// Rückgabe: `[i32; 4]` in Centipawns, ein Wert pro Spieler.
///
/// Ohne geladene Gewichte wird ein untrained-Forward-Pass verwendet (Nullwerte
/// oder zufällige He-Initialisierung). Typischerweise wird zuerst `load_weights`
/// aufgerufen.
pub fn evaluate(board: &Board) -> [i32; 4] {
    let v = match GLOBAL_NET.get() {
        Some(net) => net.forward(board),
        None => [0.0; 4],
    };
    [
        (v[0] * EVAL_SCALE) as i32,
        (v[1] * EVAL_SCALE) as i32,
        (v[2] * EVAL_SCALE) as i32,
        (v[3] * EVAL_SCALE) as i32,
    ]
}

/// Bewertet eine Stellung mit einem explizit übergebenen Netz (für Training).
pub fn evaluate_with(net: &NnueNetwork, board: &Board) -> [i32; 4] {
    let v = net.forward(board);
    [
        (v[0] * EVAL_SCALE) as i32,
        (v[1] * EVAL_SCALE) as i32,
        (v[2] * EVAL_SCALE) as i32,
        (v[3] * EVAL_SCALE) as i32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_returns_four_values() {
        let b = Board::default();
        let scores = evaluate(&b);
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn evaluate_with_net_in_range() {
        let net = NnueNetwork::new(0.001, 0.9);
        let b = Board::default();
        let scores = evaluate_with(&net, &b);
        for &s in &scores {
            let max = (EVAL_SCALE * 1.01) as i32;
            assert!(s.abs() <= max, "Score {s} außerhalb erwarteter Bandbreite");
        }
    }
}
