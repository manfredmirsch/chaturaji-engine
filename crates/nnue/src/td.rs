//! TD(λ)-Trainings-Loop für NNUE.
//! Analog zu `crates/trainer/src/td.rs`.

use rand::SeedableRng;
use chaturaji_core::piece::Color;
use chaturaji_core::zobrist::ZobristKeys;
use chaturaji_engine::book::OpeningBook;
use crate::db::{self, GameRecord};
use crate::network::{NnueNetwork, Traces};
use crate::selfplay::{final_targets, play_game, SelfPlayConfig};

pub struct TrainConfig {
    pub db_path:     String,
    pub total_games: u32,
    pub save_every:  u32,
    pub log_every:   u32,
    pub lambda:      f32,
    pub lr:          f32,
    pub momentum:    f32,
    pub selfplay:    SelfPlayConfig,
    pub lr_decay:    f32,
    pub book:        Option<OpeningBook>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            db_path:     "nnue.db".to_string(),
            total_games: 10_000,
            save_every:  200,
            log_every:   50,
            lambda:      0.7,
            lr:          0.001,
            momentum:    0.9,
            selfplay:    SelfPlayConfig::default(),
            lr_decay:    0.99,
            book:        None,
        }
    }
}

pub fn run(cfg: TrainConfig) {
    println!("=== Chaturaji NNUE TD(λ) Trainer ===");
    println!("Datenbank : {}", cfg.db_path);
    println!("Partien   : {}", cfg.total_games);
    println!("λ         : {}", cfg.lambda);
    println!("Lernrate  : {}", cfg.lr);
    match &cfg.book {
        Some(b) => println!("Buch      : {} Stellungen, max {} Halbzüge",
                            b.len(), cfg.selfplay.book_max_plies),
        None    => println!("Buch      : (kein Eröffnungsbuch)"),
    }
    println!("{}", "-".repeat(60));

    let conn = db::open(&cfg.db_path).expect("Datenbank konnte nicht geöffnet werden");

    let mut net = match db::load_latest_network(&conn) {
        Ok(Some(mut n)) => {
            n.init_momentum();
            println!("Vorhandene Gewichte geladen: {} Trainingsschritte", n.steps);
            n
        }
        Ok(None) => {
            let n = NnueNetwork::new(cfg.lr, cfg.momentum);
            println!("Neues NNUE initialisiert ({} Parameter)", n.param_count());
            n
        }
        Err(e) => {
            eprintln!("Ladefehler: {e}. Neues Netz wird verwendet.");
            NnueNetwork::new(cfg.lr, cfg.momentum)
        }
    };
    net.lr = cfg.lr;

    let mut rng     = rand::rngs::SmallRng::seed_from_u64(42);
    let mut epsilon = cfg.selfplay.epsilon_start;
    let zkeys       = ZobristKeys::new();

    let mut acc_loss  = 0.0f32;
    let mut acc_moves = 0u32;
    let mut acc_games = 0u32;

    for game_idx in 1..=cfg.total_games {
        let result = play_game(&net, &cfg.selfplay, epsilon, &mut rng,
                               cfg.book.as_ref(), &zkeys);
        let final_target = final_targets(&result.final_board);
        let n_steps = result.steps.len();

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

            let cache = net.forward_full(&step.board);
            net.backward_into_traces(&cache, &mut traces, cfg.lambda);
            net.apply_td_update(&traces, &td_error);
        }

        let avg_loss = if n_steps > 0 { game_loss / n_steps as f32 } else { 0.0 };
        acc_loss  += avg_loss;
        acc_moves += n_steps as u32;
        acc_games += 1;

        let fb = &result.final_board;
        let _ = db::save_game(&conn, &GameRecord {
            moves:         n_steps as i32,
            winner:        result.winner.map(|c| c.name().to_string()),
            score_red:     fb.scores.get(Color::Red),
            score_blue:    fb.scores.get(Color::Blue),
            score_yellow:  fb.scores.get(Color::Yellow),
            score_green:   fb.scores.get(Color::Green),
            pgn:           result.move_log.join(" "),
            network_steps: net.steps,
        });

        epsilon = (epsilon * cfg.selfplay.epsilon_decay).max(cfg.selfplay.epsilon_end);

        if game_idx % cfg.log_every == 0 {
            println!(
                "Partie {:>6}/{} | ∅Loss {:>8.5} | ∅Züge {:>5.1} | ε {:.3} | Schritte {}",
                game_idx, cfg.total_games,
                acc_loss / acc_games as f32,
                acc_moves as f32 / acc_games as f32,
                epsilon,
                net.steps,
            );
            acc_loss = 0.0; acc_moves = 0; acc_games = 0;
        }

        if game_idx % cfg.save_every == 0 {
            let avg = acc_loss / acc_games.max(1) as f32;
            match db::save_network(&conn, &net, Some(avg)) {
                Ok(id) => println!("  → Checkpoint #{id} (Schritte: {}, lr: {:.6})", net.steps, net.lr),
                Err(e) => eprintln!("  → Checkpoint fehlgeschlagen: {e}"),
            }
            let epoch = game_idx / cfg.save_every;
            let _ = db::save_stats(&conn, epoch, cfg.save_every,
                avg, acc_moves as f32 / acc_games.max(1) as f32);
            net.lr = (net.lr * cfg.lr_decay).max(1e-6);
        }
    }

    if let Ok(id) = db::save_network(&conn, &net, None) {
        println!("\nFinaler Checkpoint #{id} gespeichert.");
    }
    println!("{}", "-".repeat(60));
    println!("Training abgeschlossen. Netz-Schritte: {}", net.steps);
    println!("Datenbank: {}", cfg.db_path);
    println!("\nGewichte exportieren:");
    println!("  cargo run --release -p chaturaji-nnue --bin train_nnue -- --export nnue_weights.json");
}

pub fn export_weights(db_path: &str, out_path: &str) {
    let conn = db::open(db_path).expect("DB nicht gefunden");
    let net = db::load_latest_network(&conn)
        .expect("DB-Fehler")
        .expect("Keine Gewichte. Bitte zuerst trainieren.");
    let json = serde_json::to_string(&net).expect("Serialisierung fehlgeschlagen");
    std::fs::write(out_path, &json).expect("Datei konnte nicht geschrieben werden");
    println!("Gewichte exportiert: {out_path}");
    println!("  Schritte  : {}", net.steps);
    println!("  Parameter : {}", net.param_count());
    println!("  Dateigröße: {} KB", json.len() / 1024);
}

pub fn show_stats(db_path: &str, n: usize) {
    let conn = db::open(db_path).expect("DB nicht gefunden");
    println!("=== NNUE Trainingsstatistiken: {db_path} ===\n");

    let checkpoints = db::list_checkpoints(&conn).unwrap_or_default();
    println!("Checkpoints gesamt: {}", checkpoints.len());
    for cp in checkpoints.iter().take(8) {
        println!(
            "  #{:<4} | {:>8} Schritte | Loss {:>8} | {}",
            cp.id, cp.steps,
            cp.avg_loss.map(|v| format!("{v:.5}")).unwrap_or("–".into()),
            cp.saved_at,
        );
    }

    let stats = db::load_stats(&conn, n).unwrap_or_default();
    if !stats.is_empty() {
        println!("\nLetzte {} Epochen:", stats.len());
        println!("{:>7}  {:>8}  {:>10}  {:>9}", "Epoche", "Partien", "∅ Loss", "∅ Züge");
        println!("{}", "-".repeat(40));
        for s in &stats {
            println!("{:>7}  {:>8}  {:>10.5}  {:>9.1}", s.epoch, s.games, s.avg_loss, s.avg_game_len);
        }
    }

    println!("\nGesamt gespeicherte Partien: {}", db::game_count(&conn).unwrap_or(0));
}
