//! Chaturaji Trainer – Kommandozeilen-Interface
//!
//! Verwendung:
//!   cargo run --bin train                                  # TD-Training (Standard)
//!   cargo run --bin train -- --games 5000                 # 5000 Partien
//!   cargo run --bin train -- --db meine.db                # eigene Datenbank
//!   cargo run --bin train -- --depth 3 --lambda 0.8
//!   cargo run --bin train -- --pgn-dir ./meine_spiele     # Supervised Pre-Training
//!   cargo run --bin train -- --stats                      # Statistiken anzeigen
//!   cargo run --bin train -- --export weights.json        # Gewichte exportieren

use chaturaji_trainer::selfplay::SelfPlayConfig;
use chaturaji_trainer::td::{export_weights, run, show_stats, TrainConfig};
use chaturaji_trainer::pgn_import::load_games_from_dir;
use chaturaji_trainer::supervised::run_supervised;
use chaturaji_trainer::db;
use chaturaji_trainer::network::Network;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut db_path     = "chaturaji.db".to_string();
    let mut total_games = 10_000u32;
    let mut lambda      = 0.7f32;
    let mut lr          = 0.001f32;
    let mut depth       = 3u8;
    let mut save_every  = 200u32;
    let mut log_every   = 50u32;
    let mut mode        = Mode::Train;
    let mut export_path = "weights.json".to_string();
    let mut pgn_dir     = String::new();

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
            "--pgn-dir"    => {
                mode = Mode::Supervised;
                i += 1;
                if i < args.len() { pgn_dir = args[i].clone(); }
            }
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
        Mode::Supervised => {
            if pgn_dir.is_empty() {
                eprintln!("Fehler: --pgn-dir <verzeichnis> erforderlich.");
                return;
            }
            let games = load_games_from_dir(&pgn_dir);
            if games.is_empty() {
                eprintln!("Keine Partien gefunden in '{}'.", pgn_dir);
                return;
            }
            let conn = db::open(&db_path).expect("Datenbank konnte nicht geöffnet werden");
            let mut net = match db::load_latest_network(&conn) {
                Ok(Some(mut n)) => { n.init_momentum(); println!("Vorhandene Gewichte geladen."); n }
                _               => { println!("Neues Netz initialisiert."); Network::new(lr, 0.9) }
            };
            net.lr = lr;
            run_supervised(&mut net, &games, log_every as usize);
            db::save_network(&conn, &net, None).ok();
            println!("Gewichte gespeichert in '{}'.", db_path);
            println!("Tipp: Jetzt TD Self-Play als Fine-Tuning starten:");
            println!("  cargo run --release --bin train -- --games 5000 --depth 3");
        }
        Mode::Train => {
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

enum Mode { Train, Stats, Export, Supervised }

fn print_help() {
    println!("Chaturaji TD(\u{03bb}) Trainer\n");
    println!("VERWENDUNG:");
    println!("  cargo run --bin train -- [OPTIONEN]\n");
    println!("OPTIONEN:");
    println!("  --db <pfad>          SQLite-Datenbankpfad        [Standard: chaturaji.db]");
    println!("  --games <n>          TD-Trainingspartien         [Standard: 10000]");
    println!("  --lambda <0..1>      TD-Lambda                   [Standard: 0.7]");
    println!("  --lr <rate>          Lernrate                    [Standard: 0.001]");
    println!("  --depth <n>          Engine-Suchtiefe            [Standard: 3]");
    println!("  --save-every <n>     Checkpoint-Intervall        [Standard: 200]");
    println!("  --log-every <n>      Log-Intervall               [Standard: 50]");
    println!("  --pgn-dir <pfad>     Supervised Training aus PGN-Verzeichnis");
    println!("  --stats              Trainingsstatistiken anzeigen");
    println!("  --export [pfad]      Gewichte als JSON exportieren [Standard: weights.json]");
    println!("  --help               Diese Hilfe\n");
    println!("WORKFLOW (empfohlen):");
    println!("  1. Supervised Pre-Training mit chess.com Exporten:");
    println!("     cargo run --release --bin train -- --pgn-dir ./pgn_spiele");
    println!("  2. TD Self-Play als Fine-Tuning:");
    println!("     cargo run --release --bin train -- --games 10000 --depth 3");
}
