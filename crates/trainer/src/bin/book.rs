//! Eröffnungsbuch aus chess.com-Spielen bauen / anzeigen.
//!
//! Verwendung:
//!   cargo run --release --bin book -- build  --in <dir>  --out <file>  [--plies N]
//!   cargo run --release --bin book -- report --in <book.json>  [--top N] [--moves M]

use chaturaji_trainer::opening_book::{
    build_book_from_dir, load, print_report, save,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        return;
    }
    match args[1].as_str() {
        "build"  => cmd_build(&args[2..]),
        "report" => cmd_report(&args[2..]),
        _ => usage(),
    }
}

fn cmd_build(args: &[String]) {
    let mut in_dir   = "game_data".to_string();
    let mut out_path = "opening_book.json".to_string();
    let mut plies    = 16usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in"    => { i += 1; if i < args.len() { in_dir   = args[i].clone(); } }
            "--out"   => { i += 1; if i < args.len() { out_path = args[i].clone(); } }
            "--plies" => { i += 1; if i < args.len() { plies    = args[i].parse().unwrap_or(plies); } }
            _ => {}
        }
        i += 1;
    }

    let book = build_book_from_dir(&in_dir, plies);
    if let Err(e) = save(&book, &out_path) {
        eprintln!("Konnte nicht speichern: {}", e);
        return;
    }
    println!("Buch gespeichert in '{}'.", out_path);
}

fn cmd_report(args: &[String]) {
    let mut in_path = "opening_book.json".to_string();
    let mut top     = 10usize;
    let mut moves   = 6usize;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in"    => { i += 1; if i < args.len() { in_path = args[i].clone(); } }
            "--top"   => { i += 1; if i < args.len() { top     = args[i].parse().unwrap_or(top); } }
            "--moves" => { i += 1; if i < args.len() { moves   = args[i].parse().unwrap_or(moves); } }
            _ => {}
        }
        i += 1;
    }

    let book = match load(&in_path) {
        Ok(b)  => b,
        Err(e) => { eprintln!("Konnte '{}' nicht laden: {}", in_path, e); return; }
    };
    print_report(&book, top, moves);
}

fn usage() {
    println!("Eröffnungsbuch-Tool\n");
    println!("VERWENDUNG:");
    println!("  cargo run --release --bin book -- build  --in <dir>  --out <file> [--plies N]");
    println!("  cargo run --release --bin book -- report --in <book.json>         [--top N] [--moves M]");
    println!();
    println!("BEISPIELE:");
    println!("  cargo run --release --bin book -- build  --in /home/manfred/chaturaji/game_data --out opening_book.json --plies 12");
    println!("  cargo run --release --bin book -- report --in opening_book.json --top 5 --moves 5");
}
