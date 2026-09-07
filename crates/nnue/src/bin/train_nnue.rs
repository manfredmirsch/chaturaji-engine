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
use chaturaji_nnue::dist::{self, GenerateConfig, LearnConfig, PretrainConfig, Progress};
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
    let mut lr_decay     = 0.99f32;

    // Verteiltes Training
    let mut weights_path = "weights.json".to_string();
    let mut weights_out  = String::new();
    let mut opt_state    = "opt_state.bin".to_string();
    let mut progress_path = "progress.json".to_string();
    let mut games_dir    = String::new();
    let mut out_path     = String::new();
    let mut shards       = 1u64;
    let mut shard        = 0u64;
    let mut max_seconds  = 0u64;
    let mut threads      = 0usize;
    let mut run_seed     = 42u64;
    let mut games_done   = 0u64;
    let mut epochs       = 1u32;

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
            "--generate"   => {
                mode = Mode::Generate;
                i += 1;
                if i < args.len() { out_path = args[i].clone(); }
            }
            "--learn"      => {
                mode = Mode::Learn;
                i += 1;
                if i < args.len() { games_dir = args[i].clone(); }
            }
            "--pretrain"   => {
                mode = Mode::Pretrain;
                i += 1;
                if i < args.len() { games_dir = args[i].clone(); }
            }
            "--epochs"      => { i += 1; if i < args.len() { epochs = args[i].parse().unwrap_or(epochs); } }
            "--init-state" => { mode = Mode::InitState; }
            "--weights"     => { i += 1; if i < args.len() { weights_path  = args[i].clone(); } }
            "--weights-out" => { i += 1; if i < args.len() { weights_out   = args[i].clone(); } }
            "--opt-state"   => { i += 1; if i < args.len() { opt_state     = args[i].clone(); } }
            "--progress"    => { i += 1; if i < args.len() { progress_path = args[i].clone(); } }
            "--shards"      => { i += 1; if i < args.len() { shards      = args[i].parse().unwrap_or(shards); } }
            "--shard"       => { i += 1; if i < args.len() { shard       = args[i].parse().unwrap_or(shard); } }
            "--max-seconds" => { i += 1; if i < args.len() { max_seconds = args[i].parse().unwrap_or(max_seconds); } }
            "--threads"     => { i += 1; if i < args.len() { threads     = args[i].parse().unwrap_or(threads); } }
            "--seed"        => { i += 1; if i < args.len() { run_seed    = args[i].parse().unwrap_or(run_seed); } }
            "--games-done"  => { i += 1; if i < args.len() { games_done  = args[i].parse().unwrap_or(games_done); } }
            "--lr-decay"    => { i += 1; if i < args.len() { lr_decay    = args[i].parse().unwrap_or(lr_decay); } }
            "--book"       => { i += 1; if i < args.len() { book_path = args[i].clone(); } }
            "--no-book"    => { use_book = false; }
            "--book-plies" => { i += 1; if i < args.len() { book_plies = args[i].parse().unwrap_or(book_plies); } }
            "--book-min"   => { i += 1; if i < args.len() { book_min = args[i].parse().unwrap_or(book_min); } }
            "--help" | "-h" => { print_help(); return; }
            _ => {}
        }
        i += 1;
    }

    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("Thread-Pool konnte nicht gesetzt werden");
    }

    let load_book = |use_book: bool, book_path: &str| -> Option<OpeningBook> {
        if !use_book { return None; }
        match OpeningBook::load(book_path) {
            Ok(b) => {
                println!("Eröffnungsbuch geladen: {book_path} ({} Stellungen)", b.len());
                Some(b)
            }
            Err(e) => {
                eprintln!("Warnung: Buch '{book_path}' nicht ladbar ({e}). Training ohne Buch.");
                None
            }
        }
    };

    match mode {
        Mode::Stats  => show_stats(&db_path, 20),

        Mode::InitState => {
            // Startzustand für einen verteilten Lauf: entweder aus der lokalen
            // Trainings-DB (dann geht der bisherige Lauf weiter) oder frisch.
            let out = if weights_out.is_empty() { weights_path.clone() } else { weights_out.clone() };
            let net = match db::open(&db_path).ok().and_then(|c| db::load_latest_network(&c).ok().flatten()) {
                Some(mut n) => {
                    n.init_momentum();
                    println!("Gewichte aus '{db_path}' übernommen ({} Schritte).", n.steps);
                    n
                }
                None => {
                    println!("Keine DB gefunden — neues NNUE.");
                    let mut n = NnueNetwork::new(lr, 0.9);
                    n.init_momentum();
                    n
                }
            };
            dist::save_weights(&out, &net).expect("Gewichte nicht schreibbar");
            let progress = Progress {
                round: 0,
                games_done,
                run_seed,
                last_avg_loss:  None,
                last_avg_plies: None,
            };
            progress.save(&progress_path).expect("Fortschritt nicht schreibbar");
            println!("Startzustand geschrieben: {out}, {progress_path}");
            println!("  Seed {run_seed} | bereits gespielt: {games_done} Partien");
        }

        Mode::Pretrain => {
            if games_dir.is_empty() {
                eprintln!("Fehler: --pretrain braucht ein Verzeichnis mit .json- oder .pgn-Partien.");
                return;
            }
            let cfg = PretrainConfig {
                weights_path: weights_path.clone(),
                weights_out:  if weights_out.is_empty() { weights_path } else { weights_out },
                opt_state_path: opt_state,
                data_dir:  games_dir,
                lr,
                epochs,
                log_every: log_every as usize,
            };
            match dist::pretrain(cfg) {
                Ok(s) => println!(
                    "\n{} Partien, {} Stellungen, {} Epoche(n) in {:.1} min | Schritte {}",
                    s.games, s.positions, epochs, s.seconds / 60.0, s.steps,
                ),
                Err(e) => {
                    eprintln!("Pre-Training fehlgeschlagen: {e}");
                    std::process::exit(1);
                }
            }
        }

        Mode::Generate => {
            if out_path.is_empty() {
                eprintln!("Fehler: --generate braucht einen Ausgabepfad.");
                return;
            }
            let cfg = GenerateConfig {
                weights_path,
                progress_path,
                out_path:    out_path.clone(),
                shards,
                shard,
                games_total: total_games as u64,
                selfplay: SelfPlayConfig {
                    engine_depth:   depth,
                    beam_width,
                    book_max_plies: book_plies,
                    book_min_count: book_min,
                    max_moves,
                    ..SelfPlayConfig::default()
                },
                book: load_book(use_book, &book_path),
                max_seconds,
            };
            match dist::generate(cfg) {
                Ok(s) => {
                    println!(
                        "\n{} Partien in {:.1} min | ∅Züge {:.1} | {}",
                        s.games, s.seconds / 60.0, s.avg_plies,
                        if s.hit_budget { "Zeitbudget erreicht" } else { "vollständig" },
                    );
                    println!("Geschrieben: {out_path}");
                }
                Err(e) => {
                    eprintln!("Erzeugen fehlgeschlagen: {e}");
                    std::process::exit(1);
                }
            }
        }

        Mode::Learn => {
            if games_dir.is_empty() {
                eprintln!("Fehler: --learn braucht ein Verzeichnis mit .jsonl-Dateien.");
                return;
            }
            let cfg = LearnConfig {
                weights_path: weights_path.clone(),
                weights_out:  if weights_out.is_empty() { weights_path } else { weights_out },
                opt_state_path: opt_state,
                progress_path,
                games_dir,
                lambda,
                lr0: lr,
                lr_decay,
                save_every,
            };
            match dist::learn(cfg) {
                Ok(s) => {
                    if s.skipped > 0 {
                        eprintln!("{} Partien übersprungen.", s.skipped);
                    }
                    if s.games == 0 {
                        eprintln!("Keine Partie gelernt — Abbruch, damit die Runde nicht als erledigt gilt.");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Lernen fehlgeschlagen: {e}");
                    std::process::exit(1);
                }
            }
        }

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
            let book = load_book(use_book, &book_path);

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
                lr_decay,
                book,
            };
            run(cfg);
        }
    }
}

enum Mode { Train, Stats, Export, Supervised, Generate, Learn, InitState, Pretrain }

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
    println!("  --lr-decay <f>       Lernraten-Zerfall je save-every       [Standard: 0.99]");
    println!("  --threads <n>        Worker-Threads (0 = alle Kerne)       [Standard: 0]");
    println!("  --help               Diese Hilfe\n");
    println!("VERTEILTES TRAINING (mehrere Runner, rundenweise):");
    println!("  --init-state         Startzustand aus --db (oder frisch) schreiben");
    println!("  --generate <datei>   Partien gegen eingefrorene Gewichte spielen → JSONL");
    println!("  --learn <verz>       JSONL-Partien nachspielen und TD-Updates anwenden");
    println!("  --pretrain <verz>    Supervised Pre-Training aus echten Partien (dateibasiert)");
    println!("  --epochs <n>         Durchläufe über den Datensatz    [Standard: 1]");
    println!("  --weights <datei>    Gewichte (Ein-/Ausgabe)               [Standard: weights.json]");
    println!("  --weights-out <d>    Ziel für neue Gewichte               [Standard: wie --weights]");
    println!("  --opt-state <datei>  Adam-Momente über Runden hinweg      [Standard: opt_state.bin]");
    println!("  --progress <datei>   Runde, Partienzahl, Seed             [Standard: progress.json]");
    println!("  --shards <n>         Anzahl paralleler Erzeuger dieser Runde");
    println!("  --shard <i>          Nummer dieses Erzeugers (0-basiert)");
    println!("  --games <n>          Partien der *ganzen* Runde (über alle Shards)");
    println!("  --seed <n>           Lauf-Seed (nur bei --init-state)      [Standard: 42]");
    println!("  --games-done <n>     Bisher gespielte Partien (nur --init-state)");
    println!("  --max-seconds <n>    Zeitbudget je Erzeuger (0 = keins)\n");
    println!("EMPFOHLENER WORKFLOW:");
    println!("  1. Supervised Pre-Training mit echten Top-Partien:");
    println!("     cargo run --release -p chaturaji-nnue --bin train_nnue -- --json-dir ./game_data");
    println!("  2. TD Self-Play als Fine-Tuning:");
    println!("     cargo run --release -p chaturaji-nnue --bin train_nnue -- --games 5000 --depth 1\n");
    println!("VERGLEICH MIT BESTEHENDEM TRAINER:");
    println!("  Aktuelles MLP:  56 handgefertigte Features → 128 → 64 → 4  (~15.800 Parameter)");
    println!("  NNUE:         1280 PS-Features (sparse) → 256 → 64 → 4  (~344.000 Parameter)");
}
