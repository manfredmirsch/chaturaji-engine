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
pub(crate) struct GameMeta {
    pub(crate) points:       [i32; 4],
    pub(crate) rating_diffs: [f64; 4],
    pub(crate) ratings:      [f64; 4],
    /// Platz je Sitz (1 = bester), Index = Sitz in der Reihenfolge Rot, Blau,
    /// Gelb, Grün. **Nicht** das rohe `standings`-Feld: das listet die
    /// umgekehrte Zuordnung, siehe `parse_meta`.
    pub(crate) ranks:        [u32; 4],
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

pub(crate) fn parse_meta(json: &serde_json::Value) -> Option<GameMeta> {
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
    // `standings` ist die Ergebnisliste, nicht die Platzliste: an Position r
    // steht die **Spielernummer** (1-basiert), die Platz r+1 belegt hat.
    // Gebraucht wird die Umkehrung — der Platz je Sitz.
    //
    // Nachgeprüft an allen 4.921 Partien aus `game_data/` mit vier
    // verschiedenen Punktzahlen: die so gewonnene Rangfolge stimmt ausnahmslos
    // mit der Rangfolge nach Punkten überein. Bei Gleichstand vergibt
    // chess.com trotzdem verschiedene Plätze; dieser Tiebreak wird hier
    // übernommen, weil er im Datensatz der einzige verfügbare ist.
    let st = json.get("standings")?.as_array()?;
    if st.len() != 4 { return None; }
    let mut ranks = [0u32; 4];
    for (place, seat) in st.iter().enumerate() {
        let seat = seat.as_u64()? as usize;
        if seat < 1 || seat > 4 { return None; }
        ranks[seat - 1] = place as u32 + 1;
    }
    if ranks.contains(&0) { return None; }   // keine Permutation → Partie verwerfen
    Some(GameMeta { points, rating_diffs, ratings, ranks })
}

/// Zerlegt den `pgn4`-Text in Zug-Tokens.
///
/// Headers sind oft mehrzeilig — der `StartFen4` erstreckt sich über alle 14
/// Brett-Reihen —, deshalb wird explizit verfolgt, ob noch ein `[Tag "..."]`
/// offen ist, bis das schließende `]` kommt. Ohne das rutschen die FEN-Zeilen
/// als „Züge" durch.
///
/// Wird auch vom Zugmodell benutzt ([`crate::move_model`]), deshalb liegt sie
/// hier und nicht in `record_game`.
pub(crate) fn move_tokens(pgn4: &str) -> Vec<String> {
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

    move_text.split_whitespace()
        .filter(|t| !t.ends_with('.'))
        .filter(|t| *t != ".." && *t != "*")
        .map(str::to_string)
        .collect()
}

fn record_game(
    pgn4: &str,
    meta: &GameMeta,
    max_plies: usize,
    keys: &ZobristKeys,
    book: &mut OpeningBook,
) -> bool {
    let tokens = move_tokens(pgn4);

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
        stats.sum_rank        += meta.ranks[mover_idx];
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

    /// `standings` ist die Ergebnisliste, nicht die Platzliste. Der Test hält
    /// die Lesart fest, die an allen 4.921 eindeutigen Partien aus
    /// `game_data/` gegen die Punkte geprüft wurde.
    #[test]
    fn standings_are_read_as_a_finish_order() {
        let json: serde_json::Value = serde_json::from_str(r#"{
            "points1": 15, "points2": 23, "points3": 19, "points4": 5,
            "rating1": 2400, "rating2": 2500, "rating3": 2450, "rating4": 2350,
            "ratingDiff1": -3.0, "ratingDiff2": 8.0, "ratingDiff3": 2.0, "ratingDiff4": -7.0,
            "standings": [2, 3, 1, 4]
        }"#).unwrap();

        let meta = parse_meta(&json).expect("Metadaten müssen lesbar sein");
        // Blau (23 Punkte) wurde Erster, Gelb (19) Zweiter, Rot (15) Dritter,
        // Grün (5) Vierter — der Platz steht am Sitz, nicht am Listenplatz.
        assert_eq!(meta.ranks, [3, 1, 2, 4]);

        // Gegenprobe: die Rangfolge muss der nach Punkten entsprechen.
        let mut by_points: Vec<usize> = (0..4).collect();
        by_points.sort_by_key(|&i| -meta.points[i]);
        for (place, seat) in by_points.into_iter().enumerate() {
            assert_eq!(meta.ranks[seat], place as u32 + 1);
        }
    }

    #[test]
    fn standings_that_are_no_permutation_are_rejected() {
        let json: serde_json::Value = serde_json::from_str(r#"{
            "points1": 1, "points2": 2, "points3": 3, "points4": 4,
            "standings": [1, 1, 2, 3]
        }"#).unwrap();
        assert!(parse_meta(&json).is_none());
    }

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
