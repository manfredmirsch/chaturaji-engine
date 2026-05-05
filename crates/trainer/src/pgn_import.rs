//! Chess.com Chaturaji PGN-Importer.
//!
//! Koordinaten-Mapping (externe → interne Darstellung):
//!   Externe Dateien d-k  →  interne Dateien 0-7 (a-h)
//!   Externe Ränge  4-11  →  interne Ränge   0-7 (1-8)
//!
//! Zugformat (chess.com):
//!   "g5-g6"       – ruhiger Bauernzug
//!   "Bf4-g5+"     – Läufer mit Schach
//!   "Nf6xe8"      – Springerschlag
//!   "Kh9xBg10#"   – König schlägt Läufer (mit Figurkennung des Geschlagenen)
//!   "e4-d4=R"     – Bauernumwandlung zum Boot

use std::path::Path;
use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use crate::features::extract;

/// Eine geparste Partie: Stellungen + normalisiertes Endergebnis.
pub struct ParsedGame {
    pub positions: Vec<Vec<f32>>,
    pub outcome:   [f32; 4],
}

/// Lädt alle `.pgn`-Dateien aus `dir` und gibt geparste Partien zurück.
pub fn load_games_from_dir(dir: &str) -> Vec<ParsedGame> {
    let mut games   = Vec::new();
    let mut ok      = 0usize;
    let mut skipped = 0usize;

    let entries = match std::fs::read_dir(Path::new(dir)) {
        Ok(e)  => e,
        Err(e) => { eprintln!("Verzeichnis '{}' nicht lesbar: {}", dir, e); return games; }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("pgn") { continue; }

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

// ─── Internes Parsing ─────────────────────────────────────────────────────────

/// Teilt einen Text mit mehreren Partien (getrennt durch Leerzeilen + '['-Tags).
fn split_games(text: &str) -> Vec<&str> {
    // Einfach: jede Datei enthält eine Partie.
    // Mehrere Partien pro Datei werden durch Suche nach "[GameNr" getrennt.
    let mut starts: Vec<usize> = text.match_indices("[GameNr").map(|(i, _)| i).collect();
    if starts.is_empty() {
        return vec![text];
    }
    starts.push(text.len());
    starts.windows(2).map(|w| &text[w[0]..w[1]]).collect()
}

fn parse_game(text: &str) -> Option<ParsedGame> {
    let mut outcome_opt: Option<[f32; 4]> = None;
    let mut move_text = String::new();

    // chess.com-Headers können mehrzeilig sein (z.B. `[StartFen4 "..."]` läuft
    // über alle 14 Brett-Reihen). Wir tracken explizit, ob wir noch in einem
    // Tag-Block sind, sonst rutschen die FEN-Zeilen als „Züge" durch.
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
        if !line.is_empty() {
            move_text.push(' ');
            move_text.push_str(line);
        }
    }

    let outcome = outcome_opt?;

    let mut board     = Board::default();
    let mut positions = Vec::new();

    for token in tokenize_moves(&move_text) {
        if Rules::is_game_over(&board) { break; }

        let (from_sq, to_sq) = match parse_move_token(token) {
            Some(sq) => sq,
            None     => continue,  // Nicht parsierbar (z.B. "R" am Ende) → überspringen
        };

        let legal = Rules::legal_moves(&board);
        let mv = match legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
            Some(m) => *m,
            None    => return None,  // Stellungsmismatch → Partie verwerfen
        };

        positions.push(extract(&board));
        board = Rules::apply_with_effects(&board, mv);
    }

    if positions.is_empty() { return None; }
    Some(ParsedGame { positions, outcome })
}

/// Tokenisiert den Zugtext: Zugnummern ("1.", "16.") und Separatoren ("..") entfernen.
fn tokenize_moves(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .filter(|t| !t.ends_with('.'))  // Zugnummern wie "1.", "16."
        .filter(|t| *t != "..")          // Spieler-Separatoren
        .filter(|t| *t != "*")           // Ergebnismarker
        .collect()
}

/// Wandelt externe chess.com-Koordinaten in einen internen Feldindex um.
/// Externe Datei d-k → interne 0-7; externer Rang 4-11 → interne 0-7.
pub fn to_internal_sq(file: char, rank: u8) -> Option<u8> {
    if rank < 4 || rank > 11 { return None; }
    let f: u8 = match file {
        'd' => 0, 'e' => 1, 'f' => 2, 'g' => 3,
        'h' => 4, 'i' => 5, 'j' => 6, 'k' => 7,
        _   => return None,
    };
    let r = rank - 4;
    Some(r * 8 + f)
}

/// Parst einen einzelnen Zugtoken und gibt (from_sq, to_sq) in internen Koordinaten zurück.
///
/// Unterstützte Formate:
///   "g5-g6", "Bf4-g5+", "Nf6xe8", "Kh9xBg10#", "e4-d4=R"
pub fn parse_move_token(s: &str) -> Option<(u8, u8)> {
    // Schach/Matt-Suffix und Umwandlung abschneiden
    let s = s.trim_end_matches(['#', '+']);
    let s = if let Some(p) = s.rfind('=') { &s[..p] } else { s };

    let b = s.as_bytes();
    let mut i = 0;

    // Optionaler Figurenbuchstabe am Anfang (Großbuchstabe)
    if i < b.len() && b[i].is_ascii_uppercase() { i += 1; }

    // Ausgangsfeld: Datei (Kleinbuchstabe d-k)
    if i >= b.len() || !b[i].is_ascii_lowercase() { return None; }
    let from_file = b[i] as char; i += 1;

    // Ausgangsfeld: Rang (1-2 Ziffern)
    let rs = i;
    while i < b.len() && b[i].is_ascii_digit() { i += 1; }
    if i == rs { return None; }
    let from_rank: u8 = s[rs..i].parse().ok()?;

    // Trennzeichen: '-' oder 'x'
    if i >= b.len() || (b[i] != b'-' && b[i] != b'x') { return None; }
    i += 1;

    // Optionaler Figurenbuchstabe der geschlagenen Figur (Großbuchstabe)
    if i < b.len() && b[i].is_ascii_uppercase() { i += 1; }

    // Zielfeld: Datei
    if i >= b.len() || !b[i].is_ascii_lowercase() { return None; }
    let to_file = b[i] as char; i += 1;

    // Zielfeld: Rang
    let rs = i;
    while i < b.len() && b[i].is_ascii_digit() { i += 1; }
    if i == rs { return None; }
    let to_rank: u8 = s[rs..i].parse().ok()?;

    let from_sq = to_internal_sq(from_file, from_rank)?;
    let to_sq   = to_internal_sq(to_file,   to_rank)?;
    Some((from_sq, to_sq))
}

/// Parst den `[Result "..."]`-Tag und normalisiert die 4 Punktestände auf [-1, 1].
/// Erwartet Format: "Name1: 6 - Name2: 23 - Name3: 16 - Name4: 14"
/// Reihenfolge entspricht Rot, Blau, Gelb, Grün.
fn parse_result_tag(s: &str) -> Option<[f32; 4]> {
    let scores: Vec<f32> = s.split(" - ")
        .filter_map(|part| part.split(": ").nth(1)?.trim().parse::<f32>().ok())
        .collect();

    if scores.len() != 4 { return None; }

    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min = scores.iter().cloned().fold(f32::INFINITY,     f32::min);
    let range = (max - min).max(1.0);

    Some(std::array::from_fn(|i| (2.0 * (scores[i] - min) / range) - 1.0))
}

fn extract_tag_value(line: &str) -> Option<String> {
    let start = line.find('"')? + 1;
    let end   = line.rfind('"')?;
    if end <= start { return None; }
    Some(line[start..end].to_string())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_sq_mapping() {
        assert_eq!(to_internal_sq('d', 4), Some(0));   // a1
        assert_eq!(to_internal_sq('g', 4), Some(3));   // d1 (Red King start)
        assert_eq!(to_internal_sq('d', 11), Some(56)); // a8 (Blue Boat start)
        assert_eq!(to_internal_sq('k', 4), Some(7));   // h1 (Green Boat start)
        assert_eq!(to_internal_sq('h', 11), Some(60)); // e8 (Yellow King start)
        assert_eq!(to_internal_sq('a', 4), None);      // außerhalb
        assert_eq!(to_internal_sq('d', 3), None);      // außerhalb
    }

    #[test]
    fn parse_quiet_pawn_move() {
        let (from, to) = parse_move_token("g5-g6").unwrap();
        assert_eq!(from, to_internal_sq('g', 5).unwrap()); // d2
        assert_eq!(to,   to_internal_sq('g', 6).unwrap()); // d3
    }

    #[test]
    fn parse_piece_move_with_check() {
        let (from, to) = parse_move_token("Bf4-g5+").unwrap();
        assert_eq!(from, to_internal_sq('f', 4).unwrap());
        assert_eq!(to,   to_internal_sq('g', 5).unwrap());
    }

    #[test]
    fn parse_capture_with_captured_piece_label() {
        let (from, to) = parse_move_token("Kh9xBg10#").unwrap();
        assert_eq!(from, to_internal_sq('h', 9).unwrap());
        assert_eq!(to,   to_internal_sq('g', 10).unwrap());
    }

    #[test]
    fn parse_promotion() {
        let (from, to) = parse_move_token("e4-d4=R").unwrap();
        assert_eq!(from, to_internal_sq('e', 4).unwrap());
        assert_eq!(to,   to_internal_sq('d', 4).unwrap());
    }

    #[test]
    fn parse_result_tag_normalizes() {
        let t = parse_result_tag("A: 6 - B: 23 - C: 16 - D: 14").unwrap();
        assert!((t[0] - (-1.0)).abs() < 1e-4); // min → -1
        assert!((t[1] -   1.0 ).abs() < 1e-4); // max → +1
        for v in t { assert!(v >= -1.0 && v <= 1.0); }
    }

    #[test]
    fn garbage_token_returns_none() {
        assert!(parse_move_token("R").is_none());
        assert!(parse_move_token("..").is_none());
        assert!(parse_move_token("*").is_none());
    }

    /// Regression: chess.com-PGN hat einen mehrzeiligen `[StartFen4 "..."]`-
    /// Header. Wenn `parse_game` nur die erste Zeile als Header erkennt,
    /// rutschen die folgenden 14 FEN-Reihen als „Züge" durch und verbrauchen
    /// die Token-Quote, bevor die echten Halbzüge geparst werden.
    #[test]
    fn parse_game_handles_multiline_startfen_header() {
        let pgn = "\
[GameNr \"1\"]
[Variant \"FFA\"]
[StartFen4 \"R-0,0,0,0-0,0,0,0-0,0,0,0-0,0,0,0-0-{'dim':'8x8','boxOffset':1}-
x,x,x,x,x,x,x,x,x,x,x,x,x,x/
x,x,x,x,x,x,x,x,x,x,x,x,x,x/
x,x,x,bR,bP,2,yK,yB,yN,yR,x,x,x/
x,x,x,bN,bP,2,yP,yP,yP,yP,x,x,x/
x,x,x,bB,bP,6/
x,x,x,bK,bP,6/
x,x,x,6,gP,gK,x,x,x/
x,x,x,6,gP,gB,x,x,x/
x,x,x,rP,rP,rP,rP,2,gP,gN,x,x,x/
x,x,x,rR,rN,rB,rK,2,gP,gR,x,x,x/
x,x,x,x,x,x,x,x,x,x,x,x,x,x/
x,x,x,x,x,x,x,x,x,x,x,x,x,x\"]
[Result \"A: 10 - B: 12 - C: 8 - D: 11\"]

1. f5-f6 .. e9-f9 .. h10-h9 .. j6-i6
2. e5-e6 .. e10-f10 .. h9-h8 .. j5-i5
";
        let g = parse_game(pgn).expect("multi-line header must not break parse_game");
        // 8 Halbzüge im Text → 8 Stellungen aufgezeichnet
        assert_eq!(g.positions.len(), 8,
            "got {} positions, expected 8 — Header-Filter greift nicht durch?",
            g.positions.len());
    }
}
