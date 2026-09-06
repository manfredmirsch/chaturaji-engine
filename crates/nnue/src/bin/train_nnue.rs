//! NNUE-Trainer CLI
//!
//! Verwendung:
//!   cargo run --release -p chaturaji-nnue --bin train_nnue
//!   cargo run --release -p chaturaji-nnue --bin train_nnue -- --games 5000
//!   cargo run --release -p chaturaji-nnue --bin train_nnue -- --json-dir ./game_data
//!   cargo run --release -p chaturaji-nnue --bin train_nnue -- --pgn-dir ./pgn_spiele
//!   cargo run --release -p chaturaji-nnue --bin train_nnue -- --export nnue_weights.json
//!   cargo run --release -p chaturaji-nnue --bin train_nnue -- --stats

use chaturaji_nnue::db;
use chaturaji_nnue::network::NnueNetwork;
use chaturaji_nnue::pgn_import::{load_games_from_dir, load_games_from_json_dir};
use chaturaji_nnue::supervised::run_supervised;
use chaturaji_nnue::td::{export_weights, run, show_stats, TrainConfig};
use chaturaji_nnue::selfplay::SelfPlayConfig;
use chaturaji_engine::book::OpeningBook;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut db_path      = "nnue.db".to_string();
    let mut total_games  = 10_000u32;
    let mut lambda       = 0.7f32;
    let mut lr           = 0.001f32;
    let mut depth        = 1u8;
    let mut beam_width   = 0usize;
    let mut max_moves    = SelfPlayConfig::default().max_moves;
    let mut save_every   = 200u32;
    let mut log_every    = 50u32;
    let mut mode         = Mode::Train;
    let mut export_path  = "nnue_weights.json".to_string();
    let mut book_path    = "/home/manfred/chaturaji/opening_book.json".to_string();
    let mut use_book     = true;
    let mut book_plies   = 16usize;
    let mut book_min     = 2u32;
    let mut pgn_dir      = String::new();
    let mut json_dir     = String::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db"         => { i += 1; if i < args.len() { db_path = args[i].clone(); } }
            "--games"      => { i += 1; if i < args.len() { total_games = args[i].parse().unwrap_or(total_games); } }
            "--lambda"     => { i += 1; if i < args.len() { lambda = args[i].parse().unwrap_or(lambda); } }
            "--lr"         => { i += 1; if i < args.len() { lr = args[i].parse().unwrap_or(lr); } }
            "--depth"       => { i += 1; if i < args.len() { depth = args[i].parse().unwrap_or(depth); } }
            "--beam-width"  => { i += 1; if i < args.len() { beam_width = args[i].parse().unwrap_or(beam_width); } }
            "--max-moves"   => { i += 1; if i < args.len() { max_moves = args[i].parse().unwrap_or(max_moves); } }
            "--save-every" => { i += 1; if i < args.len() { save_every = args[i].parse().unwrap_or(save_every); } }
            "--log-every"  => { i += 1; if i < args.len() { log_every = args[i].parse().unwrap_or(log_every); } }
            "--pgn-dir"    => {
                mode = Mode::Supervised;
                i += 1;
                if i < args.len() { pgn_dir = args[i].clone(); }
            }
            "--json-dir"   => {
                mode = Mode::Supervised;
                i += 1;
                if i < args.len() { json_dir = args[i].clone(); }
            }
            "--export"     => {
                mode = Mode::Export;
                i += 1;
                if i < args.len() && !args[i].starts_with('-') {
                    export_path = args[i].clone();
                }
            }
            "--stats"      => { mode = Mode::Stats; }
            "--book"       => { i += 1; if i < args.len() { book_path = args[i].clone(); } }
            "--no-book"    => { use_book = false; }
            "--book-plies" => { i += 1; if i < args.len() { book_plies = args[i].parse().unwrap_or(book_plies); } }
            "--book-min"   => { i += 1; if i < args.len() { book_min = args[i].parse().unwrap_or(book_min); } }
            "--help" | "-h" => { print_help(); return; }
            _ => {}
        }
        i += 1;
    }

    match mode {
        Mode::Stats  => show_stats(&db_path, 20),
        Mode::Export => export_weights(&db_path, &export_path),
        Mode::Supervised => {
            if pgn_dir.is_empty() && json_dir.is_empty() {
                eprintln!("Fehler: --pgn-dir oder --json-dir erforderlich.");
                return;
            }
            let mut games = Vec::new();
            if !pgn_dir.is_empty()  { games.extend(load_games_from_dir(&pgn_dir)); }
            if !json_dir.is_empty() { games.extend(load_games_from_json_dir(&json_dir)); }
            if games.is_empty() {
                eprintln!("Keine Partien gefunden.");
                return;
            }
            let conn = db::open(&db_path).expect("Datenbank konnte nicht geöffnet werden");
            let mut net = match db::load_latest_network(&conn) {
                Ok(Some(mut n)) => { n.init_momentum(); println!("Vorhandene Gewichte geladen."); n }
                _               => { println!("Neues NNUE initialisiert."); NnueNetwork::new(lr, 0.9) }
            };
            net.lr = lr;
            run_supervised(&mut net, &games, log_every as usize);
            db::save_network(&conn, &net, None).ok();
            println!("Gewichte gespeichert in '{db_path}'.");
        }
        Mode::Train  => {
            let book = if use_book {
                match OpeningBook::load(&book_path) {
                    Ok(b) => {
                        println!("Eröffnungsbuch geladen: {} ({} Stellungen)", book_path, b.len());
                        Some(b)
                    }
                    Err(e) => {
                        eprintln!("Warnung: Buch '{}' nicht ladbar ({e}). Training ohne Buch.", book_path);
                        None
                    }
                }
            } else { None };

            let cfg = TrainConfig {
                db_path,
                total_games,
                save_every,
                log_every,
                lambda,
                lr,
                momentum: 0.9,
                selfplay: SelfPlayConfig {
                    engine_depth:   depth,
                    beam_width,
                    book_max_plies: book_plies,
                    book_min_count: book_min,
                    max_moves,
                    ..SelfPlayConfig::default()
                },
                lr_decay: 0.99,
                book,
            };
            run(cfg);
        }
    }
}

enum Mode { Train, Stats, Export, Supervised }

fn print_help() {
    println!("Chaturaji NNUE TD(λ) Trainer\n");
    println!("Architektur: 1280 (PS-Features) → 256 → 64 → 4\n");
    println!("VERWENDUNG:");
    println!("  cargo run --release -p chaturaji-nnue --bin train_nnue -- [OPTIONEN]\n");
    println!("OPTIONEN:");
    println!("  --db <pfad>          SQLite-Datenbankpfad        [Standard: nnue.db]");
    println!("  --games <n>          TD-Trainingspartien         [Standard: 10000]");
    println!("  --lambda <0..1>      TD-Lambda                   [Standard: 0.7]");
    println!("  --lr <rate>          Lernrate                    [Standard: 0.001]");
    println!("  --depth <n>          NNUE-Suchtiefe (1=greedy, 4=stark)     [Standard: 1]");
    println!("  --beam-width <n>     Beam-Breite interne Knoten (0=alle)    [Standard: 0]");
    println!("                       Empfehlung: depth 4 → --beam-width 6");
    println!("  --max-moves <n>      Halbzüge je Selbstspiel-Partie         [Standard: 150]");
    println!("                       Echte Partien: Median 94, 90 % unter 152");
    println!("  --save-every <n>     Checkpoint-Intervall        [Standard: 200]");
    println!("  --log-every <n>      Log-Intervall               [Standard: 50]");
    println!("  --stats              Trainingsstatistiken anzeigen");
    println!("  --export [pfad]      Gewichte als JSON exportieren [Standard: nnue_weights.json]");
    println!("  --book <pfad>        Eröffnungsbuch");
    println!("  --no-book            Training ohne Eröffnungsbuch");
    println!("  --book-plies <n>     Halbzüge mit Buch            [Standard: 16]");
    println!("  --book-min <n>       Mindestbeobachtungen je Buchzug [Standard: 2]");
    println!("  --pgn-dir <pfad>     Supervised Training aus PGN-Verzeichnis");
    println!("  --json-dir <pfad>    Supervised Training aus JSON-Verzeichnis (chess.com Export)");
    println!("  --help               Diese Hilfe\n");
    println!("EMPFOHLENER WORKFLOW:");
    println!("  1. Supervised Pre-Training mit echten Top-Partien:");
    println!("     cargo run --release -p chaturaji-nnue --bin train_nnue -- --json-dir ./game_data");
    println!("  2. TD Self-Play als Fine-Tuning:");
    println!("     cargo run --release -p chaturaji-nnue --bin train_nnue -- --games 5000 --depth 1\n");
    println!("VERGLEICH MIT BESTEHENDEM TRAINER:");
    println!("  Aktuelles MLP:  56 handgefertigte Features → 128 → 64 → 4  (~15.800 Parameter)");
    println!("  NNUE:         1280 PS-Features (sparse) → 256 → 64 → 4  (~344.000 Parameter)");
}
