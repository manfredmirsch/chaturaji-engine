//! Supervised Pre-Training für NNUE aus echten Partien.
//!
//! Das Netz lernt, das normalisierte Spielergebnis ([-1,1] pro Spieler)
//! direkt aus den Bitboards vorherzusagen.
//!
//! MSE-Loss: L = Σ(V(s) - outcome)² / 4
//! Kein Self-Play, kein Epsilon — reines Supervised Learning.

use chaturaji_core::board::Board;

use crate::network::NnueNetwork;
use crate::pgn_import::ParsedGame;

/// Trainiert das NNUE supervised auf den gegebenen Partien.
pub fn run_supervised(net: &mut NnueNetwork, games: &[ParsedGame], log_every: usize) {
    let total_pos: usize = games.iter().map(|g| g.positions.len()).sum();
    println!("=== NNUE Supervised Pre-Training ===");
    println!("Partien   : {}", games.len());
    println!("Stellungen: {total_pos}");
    println!("{}", "-".repeat(60));

    let mut acc_loss  = 0.0f32;
    let mut acc_count = 0usize;

    for (game_idx, game) in games.iter().enumerate() {
        for position in &game.positions {
            acc_loss  += train_one(net, position, &game.outcome);
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
    println!("Tipp: Jetzt TD Self-Play als Fine-Tuning starten:");
    println!("  cargo run --release -p chaturaji-nnue --bin train_nnue -- --games 5000 --depth 3");
}

/// Trainiert einen einzelnen (Stellung, Ergebnis)-Sample. Gibt den Loss zurück.
fn train_one(net: &mut NnueNetwork, board: &Board, target: &[f32; 4]) -> f32 {
    let cache = net.forward_full(board);
    let error: [f32; 4] = std::array::from_fn(|i| target[i] - cache.a3[i]);
    net.apply_supervised_gradient(&cache, &error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supervised_update_reduces_loss() {
        // Die Lernrate des echten Trainings (`train_nnue --lr`, Standard 0.001).
        // Höhere Werte treiben tanh bei wiederholtem Training auf demselben
        // einzelnen Sample in die Sättigung: dort ist tanh' ≈ 0 und das Netz
        // kommt nicht mehr zurück. Das ist eine Eigenschaft des normalisierten
        // Adam-Schritts, keine Aussage über den Gradienten.
        let mut net = NnueNetwork::new(0.001, 0.9);
        net.init_momentum();
        let board  = Board::default();
        let target = [1.0f32, -1.0, 0.5, -0.5];

        let loss_before = {
            let pred = net.forward(&board);
            (0..4).map(|i| (pred[i] - target[i]).powi(2)).sum::<f32>() / 4.0
        };

        for _ in 0..20 {
            train_one(&mut net, &board, &target);
        }

        let loss_after = {
            let pred = net.forward(&board);
            (0..4).map(|i| (pred[i] - target[i]).powi(2)).sum::<f32>() / 4.0
        };

        assert!(loss_after < loss_before,
            "Loss muss sinken: {loss_before:.4} → {loss_after:.4}");
    }
}
