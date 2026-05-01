//! Supervised Pre-Training aus chess.com PGN-Exporten.
//!
//! Das Netz lernt, das normalisierte Spielergebnis ([-1,1] pro Spieler)
//! direkt aus den Stellungsfeatures vorherzusagen.
//!
//! Im Gegensatz zu TD(λ) gibt es hier:
//!   • Kein Self-Play, kein Epsilon-Greedy
//!   • Einfaches MSE-Loss: L = Σ(V(s) - outcome)² / 4
//!   • Standard-Backprop (λ=0, keine Eligibility Traces)

use crate::network::{Network, Traces};
use crate::pgn_import::ParsedGame;

/// Trainiert das Netz supervised auf den gegebenen Partien.
pub fn run_supervised(net: &mut Network, games: &[ParsedGame], log_every: usize) {
    let total_pos: usize = games.iter().map(|g| g.positions.len()).sum();
    println!("=== Supervised Pre-Training ===");
    println!("Partien   : {}", games.len());
    println!("Stellungen: {}", total_pos);
    println!("{}", "-".repeat(60));

    let mut acc_loss  = 0.0f32;
    let mut acc_count = 0usize;

    for (game_idx, game) in games.iter().enumerate() {
        for features in &game.positions {
            acc_loss  += train_one(net, features, &game.outcome);
            acc_count += 1;
        }

        if (game_idx + 1) % log_every == 0 || game_idx + 1 == games.len() {
            println!(
                "Partie {:>5}/{} | ∅Loss {:>8.5} | Schritte {}",
                game_idx + 1,
                games.len(),
                if acc_count > 0 { acc_loss / acc_count as f32 } else { 0.0 },
                net.steps,
            );
            acc_loss  = 0.0;
            acc_count = 0;
        }
    }

    println!("{}", "-".repeat(60));
    println!("Supervised Training abgeschlossen. Schritte: {}", net.steps);
}

/// Trainiert einen einzelnen (Stellung, Ergebnis)-Sample mit MSE-Loss + Adam.
/// Gibt den Loss zurück.
fn train_one(net: &mut Network, features: &[f32], target: &[f32; 4]) -> f32 {
    let cache = net.forward_full(features);

    // Fehlersignal: target - prediction (positiv = Netz soll V erhöhen)
    let error: [f32; 4] = std::array::from_fn(|i| target[i] - cache.a3[i]);
    let loss = error.iter().map(|e| e * e).sum::<f32>() / 4.0;

    // λ=0: keine Akkumulation, reine Einzel-Schritt-Gradienten
    let mut traces = Traces::new();
    net.backward_into_traces(&cache, &mut traces, 0.0);
    net.apply_td_update(&traces, &error);

    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supervised_update_reduces_loss() {
        let mut net = Network::new(0.01, 0.9);
        net.init_momentum();
        let features = vec![0.5f32; crate::features::INPUT_SIZE];
        let target   = [1.0f32, -1.0, 0.5, -0.5];

        let loss_before = {
            let pred = net.forward(&features);
            let e: f32 = (0..4).map(|i| (pred[i] - target[i]).powi(2)).sum();
            e / 4.0
        };

        // Ein paar Trainingsschritte
        for _ in 0..20 {
            train_one(&mut net, &features, &target);
        }

        let loss_after = {
            let pred = net.forward(&features);
            let e: f32 = (0..4).map(|i| (pred[i] - target[i]).powi(2)).sum();
            e / 4.0
        };

        assert!(loss_after < loss_before, "Loss muss sinken: {loss_before:.4} → {loss_after:.4}");
    }
}
