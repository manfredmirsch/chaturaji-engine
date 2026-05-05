//! Eröffnungsbuch aus chess.com Chaturaji-Spielen (`game_data/*.json`).
//!
//! Pro Stellung (Zobrist-Hash) wird für jeden gespielten Zug erfasst:
//!   • wie oft er aus dieser Stellung gespielt wurde
//!   • welche Punktzahl der ziehende Spieler im Spiel insgesamt erreichte
//!   • Platz (1-4) und Rating-Differenz dieses Spielers
//!
//! Daraus lässt sich ableiten, ob ein Eröffnungszug im Mittel zu einem
//! besseren Abschneiden geführt hat.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};

// Datenstrukturen leben in der Engine, damit die zur Laufzeit das Buch ohne
// Trainer-Abhängigkeit lesen kann. Hier wird gebaut, dort konsultiert.
pub use chaturaji_engine::book::{MoveStats, OpeningBook};

use crate::pgn_import::parse_move_token;

#[derive(Debug)]
struct GameMeta {
    points:       [i32; 4],
    rating_diffs: [f64; 4],
    ratings:      [f64; 4],
    standings:    [u32; 4],
}

// ─── Loader ───────────────────────────────────────────────────────────────────

/// Liest alle `*.json`-Spiele aus `dir` und baut das Eröffnungsbuch über die
/// ersten `max_plies` Halbzüge.
pub fn build_book_from_dir(dir: &str, max_plies: usize) -> OpeningBook {
    let keys = ZobristKeys::new();
    let mut book = OpeningBook::default();
    let mut ok = 0u32;
    let mut skipped = 0u32;

    let entries = match fs::read_dir(Path::new(dir)) {
        Ok(e)  => e,
        Err(e) => { eprintln!("Verzeichnis '{}' nicht lesbar: {}", dir, e); return book; }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }

        let text = match fs::read_to_string(&path) {
            Ok(t) => t, Err(_) => { skipped += 1; continue; }
        };
        let json: serde_json::Value = match serde_json::from_str(&text) {
            Ok(j) => j, Err(_) => { skipped += 1; continue; }
        };
        let pgn4 = match json.get("pgn4").and_then(|v| v.as_str()) {
            Some(s) => s, None => { skipped += 1; continue; }
        };
        let meta = match parse_meta(&json) {
            Some(m) => m, None => { skipped += 1; continue; }
        };

        if record_game(pgn4, &meta, max_plies, &keys, &mut book) {
            ok += 1;
        } else {
            skipped += 1;
        }
    }

    println!(
        "Opening book: {} games processed, {} skipped, {} unique positions.",
        ok, skipped, book.positions.len()
    );
    book
}

fn parse_meta(json: &serde_json::Value) -> Option<GameMeta> {
    let pi = |k: &str| json.get(k).and_then(|v| v.as_i64()).map(|v| v as i32);
    let pf = |k: &str| json.get(k).and_then(|v| v.as_f64());

    let points = [pi("points1")?, pi("points2")?, pi("points3")?, pi("points4")?];
    let rating_diffs = [
        pf("ratingDiff1").unwrap_or(0.0),
        pf("ratingDiff2").unwrap_or(0.0),
        pf("ratingDiff3").unwrap_or(0.0),
        pf("ratingDiff4").unwrap_or(0.0),
    ];
    let ratings = [
        pf("rating1").unwrap_or(0.0),
        pf("rating2").unwrap_or(0.0),
        pf("rating3").unwrap_or(0.0),
        pf("rating4").unwrap_or(0.0),
    ];
    let st = json.get("standings")?.as_array()?;
    if st.len() != 4 { return None; }
    let standings = [
        st[0].as_u64()? as u32,
        st[1].as_u64()? as u32,
        st[2].as_u64()? as u32,
        st[3].as_u64()? as u32,
    ];
    Some(GameMeta { points, rating_diffs, ratings, standings })
}

fn record_game(
    pgn4: &str,
    meta: &GameMeta,
    max_plies: usize,
    keys: &ZobristKeys,
    book: &mut OpeningBook,
) -> bool {
    // Move-Text einsammeln. Headers sind oft mehrzeilig (z.B. der StartFen4
    // erstreckt sich über alle 14 Brett-Reihen), deshalb tracken wir explizit,
    // ob wir noch in einem [Tag "..."]-Block sind, bis wir das schließende ']'
    // sehen — sonst rutschen die FEN-Zeilen als „Züge" durch.
    let mut move_text = String::new();
    let mut in_header = false;
    for line in pgn4.lines() {
        let line = line.trim();
        if in_header {
            if line.ends_with(']') { in_header = false; }
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']') { in_header = true; }
            continue;
        }
        if line.is_empty() { continue; }
        move_text.push(' ');
        move_text.push_str(&strip_braces(line));
    }

    let tokens: Vec<&str> = move_text.split_whitespace()
        .filter(|t| !t.ends_with('.'))
        .filter(|t| *t != ".." && *t != "*")
        .collect();

    let mut board   = Board::default();
    let mut applied = 0usize;
    for token in tokens.iter().take(max_plies) {
        if Rules::is_game_over(&board) { break; }
        let (from_sq, to_sq) = match parse_move_token(token) {
            Some(x) => x, None => continue, // unbekannte Tokens überspringen
        };
        let legal = Rules::legal_moves(&board);
        let mv = match legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
            Some(m) => *m,
            None    => return applied > 0, // Stellungsmismatch — abbrechen
        };

        let mover_idx = board.to_move.idx();
        let hash      = hash_board(&board, keys);
        let key       = format!("{}-{}", from_sq, to_sq);

        let stats = book
            .positions.entry(hash).or_default()
            .entry(key).or_default();
        stats.count           += 1;
        stats.sum_points      += meta.points[mover_idx] as i64;
        stats.sum_rank        += meta.standings[mover_idx];
        stats.sum_rating_diff += meta.rating_diffs[mover_idx];
        stats.sum_rating      += meta.ratings[mover_idx];

        board   = Rules::apply_with_effects(&board, mv);
        applied += 1;
    }
    applied > 0
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

// ─── Persistenz ───────────────────────────────────────────────────────────────
//
// Save/Load liegen direkt auf `OpeningBook` (Engine-Crate). Die freien
// Funktionen hier sind nur Aliase fürs CLI.

pub fn save(book: &OpeningBook, path: &str) -> std::io::Result<()> { book.save(path) }
pub fn load(path: &str) -> std::io::Result<OpeningBook> { OpeningBook::load(path) }

// ─── Reporting ────────────────────────────────────────────────────────────────

/// Druckt für die ersten `top_positions` Stellungen (sortiert nach Anzahl
/// Spiele) die häufigsten Züge inkl. Stats. Stellung wird über die
/// abgespielten Halbzüge identifiziert (wir laufen nochmal die Hauptlinien ab).
pub fn print_report(book: &OpeningBook, top_positions: usize, top_moves: usize) {
    // Stellungen nach Gesamtfrequenz sortieren.
    let mut positions: Vec<(&u64, &HashMap<String, MoveStats>)> =
        book.positions.iter().collect();
    positions.sort_by(|a, b| {
        let total_a: u32 = a.1.values().map(|s| s.count).sum();
        let total_b: u32 = b.1.values().map(|s| s.count).sum();
        total_b.cmp(&total_a)
    });

    println!("\n=== Top-{} Eröffnungsstellungen ===", top_positions);
    for (i, (hash, moves)) in positions.iter().take(top_positions).enumerate() {
        let total: u32 = moves.values().map(|s| s.count).sum();
        println!("\n[{}] hash=0x{:016x}  ({} Spiele)", i + 1, hash, total);

        let mut sorted: Vec<(&String, &MoveStats)> = moves.iter().collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        for (mv, s) in sorted.iter().take(top_moves) {
            let avg_rank = s.sum_rank as f64 / s.count as f64;
            let avg_pts  = s.sum_points as f64 / s.count as f64;
            let avg_rd   = s.sum_rating_diff / s.count as f64;
            let avg_rt   = s.sum_rating / s.count as f64;
            let from_to  = format_move(mv);
            println!(
                "    {:<12} ×{:<4}  Ø-Platz {:.2}  Ø-Punkte {:.1}  Ø-RatingΔ {:+.2}  Ø-Rating {:.0}",
                from_to, s.count, avg_rank, avg_pts, avg_rd, avg_rt
            );
        }
    }
}

/// Wandelt "12-20" in "c2 → c3" (mit a-h, 1-8 — interne Notation).
fn format_move(key: &str) -> String {
    let parts: Vec<&str> = key.split('-').collect();
    if parts.len() != 2 { return key.to_string(); }
    let a: u8 = parts[0].parse().unwrap_or(0);
    let b: u8 = parts[1].parse().unwrap_or(0);
    format!("{} → {}", sq_name(a), sq_name(b))
}

fn sq_name(s: u8) -> String {
    let f = (b'a' + (s & 7)) as char;
    let r = (s >> 3) + 1;
    format!("{}{}", f, r)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_braces_removes_clock_annotations() {
        let s = "1. f5-f6 { date=2026-04-04T22:48:07 clock=63451 }  .. e9-f9";
        let out = strip_braces(s);
        assert!(out.contains("f5-f6"));
        assert!(out.contains("e9-f9"));
        assert!(!out.contains("clock"));
        assert!(!out.contains('{'));
    }

    #[test]
    fn sq_name_edges() {
        assert_eq!(sq_name(0), "a1");
        assert_eq!(sq_name(7), "h1");
        assert_eq!(sq_name(56), "a8");
        assert_eq!(sq_name(63), "h8");
    }
}
