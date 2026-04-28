//! TD(λ) Trainings-Loop.
//!
//! Ablauf pro Partie:
//!   1. Self-Play: Partie spielen, alle Stellungen + Netz-Bewertungen speichern
//!   2. TD-Update rückwärts durch die Partie:
//!      δ_t = V(s_{t+1}) - V(s_t)   (TD-Fehler)
//!      e_t = λ·e_{t-1} + ∇V(s_t)   (Eligibility Trace)
//!      Δw  = α · δ_t · e_t
//!   3. Letzter Schritt: Target = normalisiertes Endergebnis
//!   4. Checkpoint alle N Partien in SQLite speichern

use rand::SeedableRng;
use crate::db::{self, GameRecord};
use crate::network::{Network, Traces};
use crate::selfplay::{final_targets, play_game, SelfPlayConfig};

/// Konfiguration des Trainings.
pub struct TrainConfig {
    /// Pfad zur SQLite-Datenbank
    pub db_path: String,
    /// Gesamtzahl Trainingspartien
    pub total_games: u32,
    /// Checkpoint alle N Partien speichern
    pub save_every: u32,
    /// Statistiken alle N Partien ausgeben
    pub log_every: u32,
    /// TD(λ) Lambda-Parameter (0 = TD(0), 1 = Monte-Carlo)
    pub lambda: f32,
    /// Lernrate
    pub lr: f32,
    /// Momentum
    pub momentum: f32,
    /// Self-Play-Konfiguration
    pub selfplay: SelfPlayConfig,
    /// Lernraten-Abfall pro Checkpoint-Intervall
    pub lr_decay: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            db_path:     "chaturaji.db".to_string(),
            total_games: 10_000,
            save_every:  200,
            log_every:   50,
            lambda:      0.7,
            lr:          0.001,
            momentum:    0.9,
            selfplay:    SelfPlayConfig::default(),
            lr_decay:    0.99,
        }
    }
}

/// Startet den Trainings-Loop.
pub fn run(cfg: TrainConfig) {
    println!("=== Chaturaji TD(\u{03bb}) Trainer ===");
    println!("Datenbank : {}", cfg.db_path);
    println!("Partien   : {}", cfg.total_games);
    println!("\u{03bb}         : {}", cfg.lambda);
    println!("Lernrate  : {}", cfg.lr);
    println!("{}", "-".repeat(60));

    let conn = db::open(&cfg.db_path).expect("Datenbank konnte nicht geöffnet werden");

    // Netz laden oder neu initialisieren
    let mut net = match db::load_latest_network(&conn) {
        Ok(Some(mut n)) => {
            n.init_momentum();
            println!("Vorhandene Gewichte geladen: {} Trainingsschritte", n.steps);
            n
        }
        Ok(None) => {
            let n = Network::new(cfg.lr, cfg.momentum);
            println!("Neues Netz initialisiert ({} Parameter)", n.param_count());
            n
        }
        Err(e) => {
            eprintln!("Ladefehler: {e}. Neues Netz wird verwendet.");
            Network::new(cfg.lr, cfg.momentum)
        }
    };
    net.lr       = cfg.lr;
    net.momentum = cfg.momentum;

    let mut rng     = rand::rngs::SmallRng::seed_from_u64(42);
    let mut epsilon = cfg.selfplay.epsilon_start;

    // Akkumulatoren für Logging
    let mut acc_loss  = 0.0f32;
    let mut acc_moves = 0u32;
    let mut acc_games = 0u32;

    for game_idx in 1..=cfg.total_games {
        // 1. Self-Play-Partie
        let result = play_game(&net, &cfg.selfplay, epsilon, &mut rng);
        let final_target = final_targets(&result.final_board);
        let n_steps = result.steps.len();

        // 2. TD(λ)-Update
        let mut traces    = Traces::new();
        let mut game_loss = 0.0f32;

        for t in (0..n_steps).rev() {
            let step = &result.steps[t];
            let v_next: [f32; 4] = if t + 1 < n_steps {
                result.steps[t + 1].value
            } else {
                final_target
            };
            let td_error: [f32; 4] = std::array::from_fn(|i| v_next[i] - step.value[i]);
            game_loss += td_error.iter().map(|e| e * e).sum::<f32>() / 4.0;

            let cache = net.forward_full(&step.features);
            net.backward_into_traces(&cache, &mut traces, cfg.lambda);
            net.apply_td_update(&traces, &td_error);
        }

        let avg_loss = if n_steps > 0 { game_loss / n_steps as f32 } else { 0.0 };
        acc_loss  += avg_loss;
        acc_moves += n_steps as u32;
        acc_games += 1;

        // 3. Partie in DB speichern
        let fb = &result.final_board;
        let _  = db::save_game(&conn, &GameRecord {
            moves:         n_steps as i32,
            winner:        result.winner.map(|c| c.name().to_string()),
            score_red:     fb.scores.get(chaturaji_core::piece::Color::Red),
            score_blue:    fb.scores.get(chaturaji_core::piece::Color::Blue),
            score_yellow:  fb.scores.get(chaturaji_core::piece::Color::Yellow),
            score_green:   fb.scores.get(chaturaji_core::piece::Color::Green),
            pgn:           result.move_log.join(" "),
            network_steps: net.steps,
        });

        // 4. Epsilon-Decay
        epsilon = (epsilon * cfg.selfplay.epsilon_decay).max(cfg.selfplay.epsilon_end);

        // 5. Logging
        if game_idx % cfg.log_every == 0 {
            println!(
                "Partie {:>6}/{} | \u{2205}Loss {:>8.5} | \u{2205}Züge {:>5.1} | \u{03b5} {:.3} | Schritte {}",
                game_idx, cfg.total_games,
                acc_loss / acc_games as f32,
                acc_moves as f32 / acc_games as f32,
                epsilon,
                net.steps
            );
            acc_loss = 0.0; acc_moves = 0; acc_games = 0;
        }

        // 6. Checkpoint
        if game_idx % cfg.save_every == 0 {
            let avg = acc_loss / acc_games.max(1) as f32;
            match db::save_network(&conn, &net, Some(avg)) {
                Ok(id) => println!("  \u{2192} Checkpoint #{id} (Schritte: {}, lr: {:.6})", net.steps, net.lr),
                Err(e) => eprintln!("  \u{2192} Checkpoint fehlgeschlagen: {e}"),
            }
            let epoch = game_idx / cfg.save_every;
            let _ = db::save_stats(&conn, epoch, cfg.save_every,
                avg, acc_moves as f32 / acc_games.max(1) as f32);
            net.lr = (net.lr * cfg.lr_decay).max(1e-6);
        }
    }

    // Abschließender Checkpoint
    if let Ok(id) = db::save_network(&conn, &net, None) {
        println!("\nFinaler Checkpoint #{id} gespeichert.");
    }
    println!("{}", "-".repeat(60));
    println!("Training abgeschlossen. Netz-Schritte: {}", net.steps);
    println!("Datenbank: {}", cfg.db_path);
    println!("\nGewichte für Browser exportieren:");
    println!("  cargo run --bin train -- --export weights.json");
}

/// Exportiert die neuesten Gewichte als JSON-Datei (für den Browser).
pub fn export_weights(db_path: &str, out_path: &str) {
    let conn = db::open(db_path).expect("DB nicht gefunden");
    let net = db::load_latest_network(&conn)
        .expect("DB-Fehler")
        .expect("Keine Gewichte in der Datenbank. Bitte zuerst trainieren.");

    let json = serde_json::to_string(&net).expect("Serialisierung fehlgeschlagen");
    std::fs::write(out_path, &json).expect("Datei konnte nicht geschrieben werden");
    println!("Gewichte exportiert: {out_path}");
    println!("  Schritte  : {}", net.steps);
    println!("  Parameter : {}", net.param_count());
    println!("  Dateigröße: {} KB", json.len() / 1024);
    println!("\nIn der Web-UI: 'Neuronales Netz' > 'Gewichte laden' > {out_path} auswählen.");
}

/// Zeigt Trainingsstatistiken aus der Datenbank an.
pub fn show_stats(db_path: &str, n: usize) {
    let conn = db::open(db_path).expect("DB nicht gefunden");

    println!("=== Trainingsstatistiken: {db_path} ===\n");

    let checkpoints = db::list_checkpoints(&conn).unwrap_or_default();
    println!("Checkpoints gesamt: {}", checkpoints.len());
    for cp in checkpoints.iter().take(8) {
        println!(
            "  #{:<4} | {:>8} Schritte | Loss {:>8} | {}",
            cp.id, cp.steps,
            cp.avg_loss.map(|v| format!("{v:.5}")).unwrap_or("–".into()),
            cp.saved_at
        );
    }

    let stats = db::load_stats(&conn, n).unwrap_or_default();
    if !stats.is_empty() {
        println!("\nLetzte {} Epochen:", stats.len());
        println!("{:>7}  {:>8}  {:>10}  {:>9}", "Epoche", "Partien", "Ø Loss", "Ø Züge");
        println!("{}", "-".repeat(40));
        for s in &stats {
            println!("{:>7}  {:>8}  {:>10.5}  {:>9.1}", s.epoch, s.games, s.avg_loss, s.avg_game_len);
        }
    }

    println!("\nGesamt gespeicherte Partien: {}", db::game_count(&conn).unwrap_or(0));

    let recent = db::load_recent_games(&conn, 8).unwrap_or_default();
    if !recent.is_empty() {
        println!("\nLetzte Partien:");
        for g in &recent {
            println!(
                "  {:>4} Züge | Sieger: {:>6} | R:{:>3} B:{:>3} Y:{:>3} G:{:>3}",
                g.moves,
                g.winner.as_deref().unwrap_or("–"),
                g.score_red, g.score_blue, g.score_yellow, g.score_green
            );
        }
    }
}
