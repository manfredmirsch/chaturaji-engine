//! Chess.com Chaturaji PGN/JSON-Importer für NNUE-Training.
//!
//! Analog zu `crates/trainer/src/pgn_import.rs`, aber mit Bitboard-Darstellung:
//! Statt `Vec<f32>` (56 handgefertigte Features) werden `[[u64; 5]; 4]`
//! (rohe Bitboards) als Trainings-Input gespeichert.
//!
//! Koordinaten-Mapping (chess.com → intern):
//!   Dateien d-k  →  0-7
//!   Ränge  4-11  →  0-7

use std::path::{Path, PathBuf};
use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;

use crate::outcome::{place_values, place_values_from_standings};

/// Dateien eines Verzeichnisses mit der gesuchten Endung, nach Namen sortiert.
///
/// `read_dir` liefert die Reihenfolge des Dateisystems, und die ist auf einem
/// CI-Runner eine andere als lokal. Beim Supervised Training ist das keine
/// Kosmetik: SGD läuft die Partien in genau dieser Reihenfolge ab, derselbe
/// Datensatz ergäbe sonst je Maschine ein anderes Netz.
fn sorted_files(dir: &str, ext: &str) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(Path::new(dir)) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(ext))
            .collect(),
        Err(e) => {
            eprintln!("Verzeichnis '{dir}' nicht lesbar: {e}");
            Vec::new()
        }
    };
    paths.sort();
    paths
}

pub struct ParsedGame {
    /// Vollständige Stellungen. Früher nur Bitboards — das reichte nicht, weil
    /// das Netz auch Punktestand, Zugrecht und Spielphase als Eingabe bekommt.
    pub positions: Vec<Board>,
    pub outcome:   [f32; 4],
}

/// Lädt alle `.pgn`-Dateien aus `dir`.
pub fn load_games_from_dir(dir: &str) -> Vec<ParsedGame> {
    let mut games   = Vec::new();
    let mut ok      = 0usize;
    let mut skipped = 0usize;

    for path in sorted_files(dir, "pgn") {
        let text = match std::fs::read_to_string(&path) {
            Ok(t)  => t,
            Err(_) => { skipped += 1; continue; }
        };

        for game_text in split_games(&text) {
            match parse_game(game_text) {
                Some(g) => { games.push(g); ok += 1; }
                None    => { skipped += 1; }
            }
        }
    }

    println!("PGN: {} Partien geladen, {} übersprungen.", ok, skipped);
    games
}

/// Lädt alle `.json`-Dateien aus `dir` (chess.com Export-Format mit `pgn4` + `standings`).
pub fn load_games_from_json_dir(dir: &str) -> Vec<ParsedGame> {
    let mut games   = Vec::new();
    let mut ok      = 0usize;
    let mut skipped = 0usize;

    for path in sorted_files(dir, "json") {
        let text = match std::fs::read_to_string(&path) {
            Ok(t)  => t,
            Err(_) => { skipped += 1; continue; }
        };

        let json: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v)  => v,
            Err(_) => { skipped += 1; continue; }
        };

        let pgn4 = match json["pgn4"].as_str() {
            Some(s) => s,
            None    => { skipped += 1; continue; }
        };

        // Punkte sind die bessere Quelle als `standings`: sie machen
        // Gleichstände sichtbar, die `place_values` dann mitteln kann, statt
        // den willkürlichen chess.com-Tiebreak als Ziel zu lernen.
        let points: Option<[i32; 4]> = (|| Some([
            json["points1"].as_i64()? as i32,
            json["points2"].as_i64()? as i32,
            json["points3"].as_i64()? as i32,
            json["points4"].as_i64()? as i32,
        ]))();

        let outcome = match points {
            Some(p) => place_values(p),
            None => {
                let sa = &json["standings"];
                let standings: Option<[u8; 4]> = (|| Some([
                    sa[0].as_u64()? as u8,
                    sa[1].as_u64()? as u8,
                    sa[2].as_u64()? as u8,
                    sa[3].as_u64()? as u8,
                ]))();
                match standings {
                    Some(v) => place_values_from_standings(v),
                    None    => { skipped += 1; continue; }
                }
            }
        };

        let positions = match parse_positions_from_pgn(pgn4) {
            Some(p) if !p.is_empty() => p,
            _                        => { skipped += 1; continue; }
        };

        games.push(ParsedGame { positions, outcome });
        ok += 1;
    }

    println!("JSON: {} Partien geladen, {} übersprungen.", ok, skipped);
    games
}

// ─── Internes Parsing ─────────────────────────────────────────────────────────

fn split_games(text: &str) -> Vec<&str> {
    let mut starts: Vec<usize> = text.match_indices("[GameNr").map(|(i, _)| i).collect();
    if starts.is_empty() { return vec![text]; }
    starts.push(text.len());
    starts.windows(2).map(|w| &text[w[0]..w[1]]).collect()
}

fn parse_game(text: &str) -> Option<ParsedGame> {
    let mut outcome_opt: Option<[f32; 4]> = None;
    let mut in_header = false;

    for line in text.lines() {
        let line = line.trim();
        if in_header {
            if line.ends_with(']') { in_header = false; }
            continue;
        }
        if line.starts_with('[') {
            if line.starts_with("[Result ") {
                if let Some(val) = extract_tag_value(line) {
                    outcome_opt = parse_result_tag(&val);
                }
            }
            if !line.ends_with(']') { in_header = true; }
            continue;
        }
    }

    let outcome   = outcome_opt?;
    let positions = parse_positions_from_pgn(text)?;
    if positions.is_empty() { return None; }
    Some(ParsedGame { positions, outcome })
}

fn parse_positions_from_pgn(text: &str) -> Option<Vec<Board>> {
    let mut move_text = String::new();
    let mut in_header = false;

    for line in text.lines() {
        let line = line.trim();
        if in_header {
            if line.ends_with(']') { in_header = false; }
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']') { in_header = true; }
            continue;
        }
        if !line.is_empty() {
            move_text.push(' ');
            move_text.push_str(line);
        }
    }

    let mut board     = Board::default();
    let mut positions = Vec::new();

    for token in tokenize_moves(&move_text) {
        if Rules::is_game_over(&board) { break; }

        // `R` (Aufgabe) und `T` (Zeitüberschreitung) belegen einen Zugslot.
        // Sie hier nur zu überspringen — wie es jedes andere unverstandene
        // Token erfährt — verschiebt das Zugrecht um einen Spieler, und ab da
        // passt kein Zug mehr. Genau daran scheiterte ein Viertel aller
        // Partien: 242 von 1000, und in allen 242 stand ein R oder T davor.
        if token == "R" || token == "T" {
            board = Rules::resign(&board);
            continue;
        }

        let (from_sq, to_sq) = match parse_move_token(token) {
            Some(sq) => sq,
            None     => continue,
        };

        let legal = Rules::legal_moves(&board);
        let mv = match legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
            Some(m) => *m,
            None    => return None,
        };

        positions.push(board.clone());
        board = Rules::apply_with_effects(&board, mv);
    }

    Some(positions)
}

pub fn to_internal_sq(file: char, rank: u8) -> Option<u8> {
    if rank < 4 || rank > 11 { return None; }
    let f: u8 = match file {
        'd' => 0, 'e' => 1, 'f' => 2, 'g' => 3,
        'h' => 4, 'i' => 5, 'j' => 6, 'k' => 7,
        _   => return None,
    };
    Some((rank - 4) * 8 + f)
}

pub fn parse_move_token(s: &str) -> Option<(u8, u8)> {
    let s = s.trim_end_matches(['#', '+']);
    let s = if let Some(p) = s.rfind('=') { &s[..p] } else { s };
    let b = s.as_bytes();
    let mut i = 0;

    if i < b.len() && b[i].is_ascii_uppercase() { i += 1; }
    if i >= b.len() || !b[i].is_ascii_lowercase() { return None; }
    let from_file = b[i] as char; i += 1;

    let rs = i;
    while i < b.len() && b[i].is_ascii_digit() { i += 1; }
    if i == rs { return None; }
    let from_rank: u8 = s[rs..i].parse().ok()?;

    if i >= b.len() || (b[i] != b'-' && b[i] != b'x') { return None; }
    i += 1;

    if i < b.len() && b[i].is_ascii_uppercase() { i += 1; }
    if i >= b.len() || !b[i].is_ascii_lowercase() { return None; }
    let to_file = b[i] as char; i += 1;

    let rs = i;
    while i < b.len() && b[i].is_ascii_digit() { i += 1; }
    if i == rs { return None; }
    let to_rank: u8 = s[rs..i].parse().ok()?;

    Some((to_internal_sq(from_file, from_rank)?, to_internal_sq(to_file, to_rank)?))
}

/// Parst einen `Result`-Tag der Form `"NameA: 15 - NameB: 23 - …"` und macht
/// daraus die Platzwertung.
fn parse_result_tag(s: &str) -> Option<[f32; 4]> {
    let points: Vec<i32> = s.split(" - ")
        .filter_map(|part| part.split(": ").nth(1)?.trim().parse::<i32>().ok())
        .collect();
    if points.len() != 4 { return None; }
    Some(place_values([points[0], points[1], points[2], points[3]]))
}

fn extract_tag_value(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let end   = line.rfind('"')?;
    if end <= start { return None; }
    Some(line[start..end].to_string())
}

fn tokenize_moves(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .filter(|t| !t.ends_with('.'))
        .filter(|t| *t != "..")
        .filter(|t| *t != "*")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_sq_mapping() {
        assert_eq!(to_internal_sq('d', 4),  Some(0));
        assert_eq!(to_internal_sq('g', 4),  Some(3));
        assert_eq!(to_internal_sq('d', 11), Some(56));
        assert_eq!(to_internal_sq('k', 4),  Some(7));
        assert_eq!(to_internal_sq('a', 4),  None);
    }

    #[test]
    fn parse_quiet_pawn_move() {
        let (f, t) = parse_move_token("g5-g6").unwrap();
        assert_eq!(f, to_internal_sq('g', 5).unwrap());
        assert_eq!(t, to_internal_sq('g', 6).unwrap());
    }

    #[test]
    fn parse_capture_with_label() {
        let (f, t) = parse_move_token("Kh9xBg10#").unwrap();
        assert_eq!(f, to_internal_sq('h', 9).unwrap());
        assert_eq!(t, to_internal_sq('g', 10).unwrap());
    }

    #[test]
    fn parse_promotion() {
        let (f, t) = parse_move_token("e4-d4=R").unwrap();
        assert_eq!(f, to_internal_sq('e', 4).unwrap());
        assert_eq!(t, to_internal_sq('d', 4).unwrap());
    }

    #[test]
    fn positions_are_bitboards() {
        let pgn = "\
[GameNr \"1\"]
[Result \"A: 10 - B: 12 - C: 8 - D: 11\"]

1. f5-f6 .. e9-f9 .. h10-h9 .. j6-i6
";
        let g = parse_game(pgn).expect("Partie muss parsierbar sein");
        assert_eq!(g.positions.len(), 4);
        // Startstellung hat 32 Figuren
        let total_pieces: u32 = g.positions[0].bb.iter()
            .flat_map(|row| row.iter())
            .map(|bb| bb.count_ones())
            .sum();
        assert_eq!(total_pieces, 32);
    }
}
