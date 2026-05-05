//! Debug-Tool: zeigt für ein paar Spiele, ab welchem Zug das Replay scheitert.
use std::fs;
use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};
use chaturaji_trainer::pgn_import::parse_move_token;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "game_data".into());
    let n   = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(5usize);

    let keys = ZobristKeys::new();
    let mut total = 0usize;
    let mut failed_at: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut failed_examples = Vec::new();
    let mut hashes_seen: std::collections::HashSet<u64> = Default::default();
    let mut sample_trace: Vec<(usize, u64, String)> = Vec::new();

    for entry in fs::read_dir(&dir).unwrap().flatten().take(n * 50) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
        let text = match fs::read_to_string(&path) { Ok(t) => t, Err(_) => continue };
        let json: serde_json::Value = match serde_json::from_str(&text) { Ok(j) => j, Err(_) => continue };
        let pgn4 = match json.get("pgn4").and_then(|v| v.as_str()) { Some(s) => s, None => continue };

        let mut move_text = String::new();
        for line in pgn4.lines() {
            let line = line.trim();
            if line.starts_with('[') || line.is_empty() { continue; }
            move_text.push(' ');
            move_text.push_str(&strip_braces(line));
        }
        let tokens: Vec<&str> = move_text.split_whitespace()
            .filter(|t| !t.ends_with('.'))
            .filter(|t| *t != ".." && *t != "*")
            .collect();

        let mut board = Board::default();
        let mut ply = 0usize;
        let trace_this = total < 1; // dump full trace for the first game
        for token in tokens.iter().take(16) {
            if Rules::is_game_over(&board) { break; }
            let h = hash_board(&board, &keys);
            hashes_seen.insert(h);
            if trace_this {
                sample_trace.push((ply, h, token.to_string()));
            }
            let parsed = parse_move_token(token);
            let (from_sq, to_sq) = match parsed { Some(x) => x, None => continue };
            let legal = Rules::legal_moves(&board);
            match legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
                Some(m) => { board = Rules::apply_with_effects(&board, *m); ply += 1; }
                None    => {
                    *failed_at.entry(ply).or_default() += 1;
                    if failed_examples.len() < n {
                        let game_nr = json.get("gameNr").and_then(|v| v.as_u64()).unwrap_or(0);
                        failed_examples.push((game_nr, ply, token.to_string(), from_sq, to_sq, board.to_move));
                    }
                    break;
                }
            }
        }
        total += 1;
    }
    println!("Eindeutige Hashes über alle Spiele: {}", hashes_seen.len());
    println!("Trace erstes Spiel:");
    for (ply, h, tok) in &sample_trace {
        println!("  ply {:>2}: hash=0x{:016x}  token={}", ply, h, tok);
    }

    println!("Spiele inspiziert: {}", total);
    println!("Fehlerverteilung nach ply:");
    for (p, c) in &failed_at {
        println!("  ply {:>2}: {} mal", p, c);
    }
    println!("\nBeispiele für gescheiterte Replays:");
    for (gnr, ply, tok, fr, to, mover) in &failed_examples {
        println!("  Spiel {}: ply {} → Token '{}' → from={} to={} mover={:?}",
            gnr, ply, tok, fr, to, mover);
    }
}

fn strip_braces(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut depth = 0i32;
    for ch in line.chars() {
        match ch {
            '{' => depth += 1,
            '}' => if depth > 0 { depth -= 1; },
            _   => if depth == 0 { out.push(ch); },
        }
    }
    out
}
