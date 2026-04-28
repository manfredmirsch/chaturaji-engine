//! Chaturaji Trainer – Kommandozeilen-Interface
//!
//! Verwendung:
//!   cargo run --bin train                              # Training mit Standardwerten
//!   cargo run --bin train -- --games 5000             # 5000 Partien
//!   cargo run --bin train -- --db meine.db            # eigene Datenbank
//!   cargo run --bin train -- --lambda 0.8 --lr 0.0005
//!   cargo run --bin train -- --stats                  # Statistiken anzeigen
//!   cargo run --bin train -- --export weights.json    # Gewichte exportieren

use chaturaji_trainer::selfplay::SelfPlayConfig;
use chaturaji_trainer::td::{export_weights, run, show_stats, TrainConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut db_path     = "chaturaji.db".to_string();
    let mut total_games = 10_000u32;
    let mut lambda      = 0.7f32;
    let mut lr          = 0.001f32;
    let mut depth       = 2u8;
    let mut save_every  = 200u32;
    let mut log_every   = 50u32;
    let mut mode        = Mode::Train;
    let mut export_path = "weights.json".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db"         => { i += 1; if i < args.len() { db_path = args[i].clone(); } }
            "--games"      => { i += 1; if i < args.len() { total_games = args[i].parse().unwrap_or(total_games); } }
            "--lambda"     => { i += 1; if i < args.len() { lambda = args[i].parse().unwrap_or(lambda); } }
            "--lr"         => { i += 1; if i < args.len() { lr = args[i].parse().unwrap_or(lr); } }
            "--depth"      => { i += 1; if i < args.len() { depth = args[i].parse().unwrap_or(depth); } }
            "--save-every" => { i += 1; if i < args.len() { save_every = args[i].parse().unwrap_or(save_every); } }
            "--log-every"  => { i += 1; if i < args.len() { log_every = args[i].parse().unwrap_or(log_every); } }
            "--export"     => {
                mode = Mode::Export;
                i += 1;
                if i < args.len() && !args[i].starts_with('-') {
                    export_path = args[i].clone();
                }
            }
            "--stats"      => { mode = Mode::Stats; }
            "--help" | "-h"=> { print_help(); return; }
            _ => {}
        }
        i += 1;
    }

    match mode {
        Mode::Stats  => show_stats(&db_path, 20),
        Mode::Export => export_weights(&db_path, &export_path),
        Mode::Train  => {
            let cfg = TrainConfig {
                db_path,
                total_games,
                save_every,
                log_every,
                lambda,
                lr,
                momentum: 0.9,
                selfplay: SelfPlayConfig {
                    engine_depth: depth,
                    ..SelfPlayConfig::default()
                },
                lr_decay: 0.99,
            };
            run(cfg);
        }
    }
}

enum Mode { Train, Stats, Export }

fn print_help() {
    println!("Chaturaji TD(\u{03bb}) Trainer\n");
    println!("VERWENDUNG:");
    println!("  cargo run --bin train -- [OPTIONEN]\n");
    println!("OPTIONEN:");
    println!("  --db <pfad>          SQLite-Datenbankpfad     [Standard: chaturaji.db]");
    println!("  --games <n>          Trainingspartien         [Standard: 10000]");
    println!("  --lambda <0..1>      TD-Lambda                [Standard: 0.7]");
    println!("  --lr <rate>          Lernrate                 [Standard: 0.001]");
    println!("  --depth <n>          Engine-Suchtiefe         [Standard: 2]");
    println!("  --save-every <n>     Checkpoint-Intervall     [Standard: 200]");
    println!("  --log-every <n>      Log-Intervall            [Standard: 50]");
    println!("  --stats              Trainingsstatistiken anzeigen");
    println!("  --export [pfad]      Gewichte als JSON exportieren [Standard: weights.json]");
    println!("  --help               Diese Hilfe\n");
    println!("BEISPIELE:");
    println!("  cargo run --bin train -- --games 1000 --log-every 10");
    println!("  cargo run --bin train -- --stats");
    println!("  cargo run --bin train -- --export meine_gewichte.json");
}
