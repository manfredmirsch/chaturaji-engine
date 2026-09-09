//! Was die Engine von den Buchzügen hält: jeden Zug der Stellung anwenden und
//! die Folgestellung mit BRS durchrechnen. Ausgegeben wird der Wert aus Sicht
//! des Ziehenden.
//!
//!   cargo run --release --bin rate_moves -- --depth 4 [d2d3 b6c6 ...]

use chaturaji_core::board::Board;
use chaturaji_core::notation::{move_to_str, parse_move};
use chaturaji_core::rules::Rules;
use chaturaji_engine::search::Engine;

fn main() {
    let mut depth = 4u8;
    let mut line: Vec<String> = Vec::new();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--depth" { depth = args.get(i+1).and_then(|v| v.parse().ok()).unwrap_or(depth); i += 2; }
        else { line.push(args[i].clone()); i += 1; }
    }

    let mut board = Board::default();
    for tok in &line {
        match parse_move(&board, tok) {
            Ok(mv) => board = Rules::apply_with_effects(&board, mv),
            Err(e) => { eprintln!("Zug '{tok}' nicht spielbar: {e}"); std::process::exit(1); }
        }
    }
    let seat = board.to_move.idx();
    println!("Linie: {}   am Zug: {}   Tiefe {}", line.join(" "), board.to_move.name(), depth);

    let mut eng = Engine::new(64);
    let mut rows: Vec<(String, i32)> = Rules::legal_moves(&board).into_iter().map(|mv| {
        let child = Rules::apply_with_effects(&board, mv);
        let r = eng.search_brs(&child, depth.saturating_sub(1), None);
        (move_to_str(&mv), r.scores[seat])
    }).collect();
    rows.sort_by_key(|(_, s)| -s);
    for (mv, s) in rows { println!("  {mv:6} {s:+6}"); }
}
