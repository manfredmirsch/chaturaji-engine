//! Notation support: square names, move serialization, basic PGN parsing.
//!
//! Square naming follows standard algebraic notation: a1..h8.
//! Move format: `<from><to>[p]`  e.g. `a2a3`, `c1e3p` (promotion).

use crate::board::{file_of, rank_of, sq, Board, Move};
use crate::movegen::MoveGen;
use crate::piece::Color;

// ─── Square helpers ───────────────────────────────────────────────────────────

/// `"a1"` → `0`, `"h8"` → `63`.
pub fn parse_square(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() < 2 { return None; }
    let f = bytes[0].wrapping_sub(b'a');
    let r = bytes[1].wrapping_sub(b'1');
    if f > 7 || r > 7 { return None; }
    Some(sq(f, r))
}

/// `0` → `"a1"`, `63` → `"h8"`.
pub fn square_name(s: u8) -> String {
    let f = (b'a' + file_of(s)) as char;
    let r = (b'1' + rank_of(s)) as char;
    format!("{}{}", f, r)
}

// ─── Move serialization ───────────────────────────────────────────────────────

/// Serialize a move: `"a2a3"` or `"c2c8p"` (promotion suffix).
pub fn move_to_str(mv: &Move) -> String {
    let mut s = format!("{}{}", square_name(mv.from), square_name(mv.to));
    if mv.promoted { s.push('p'); }
    s
}

/// Parse a move string and match it against legal moves on `board`.
/// Returns `Err` with a message if the move is not found.
pub fn parse_move(board: &Board, s: &str) -> Result<Move, String> {
    let s = s.trim();
    if s.len() < 4 { return Err(format!("move too short: {s}")); }

    let from = parse_square(&s[0..2]).ok_or_else(|| format!("bad from square: {}", &s[0..2]))?;
    let to   = parse_square(&s[2..4]).ok_or_else(|| format!("bad to square: {}",   &s[2..4]))?;
    let prom = s.len() > 4 && s.as_bytes()[4] == b'p';

    MoveGen::generate(board)
        .into_iter()
        .find(|mv| mv.from == from && mv.to == to && mv.promoted == prom)
        .ok_or_else(|| format!("illegal move: {s}"))
}

// ─── PGN-like game record ─────────────────────────────────────────────────────

/// A minimal game record (move list + headers).
#[derive(Default)]
pub struct GameRecord {
    pub event:  String,
    pub site:   String,
    pub date:   String,
    pub moves:  Vec<String>,   // serialized move strings
    pub result: String,
}

impl GameRecord {
    /// Serialize to a simple PGN-like text.
    pub fn to_pgn(&self) -> String {
        let mut out = String::new();
        if !self.event.is_empty()  { out += &format!("[Event \"{}\"]\n", self.event); }
        if !self.site.is_empty()   { out += &format!("[Site \"{}\"]\n",  self.site); }
        if !self.date.is_empty()   { out += &format!("[Date \"{}\"]\n",  self.date); }
        if !self.result.is_empty() { out += &format!("[Result \"{}\"]\n",self.result); }
        out.push('\n');

        let players = [Color::Red, Color::Blue, Color::Yellow, Color::Green];
        let mut board = Board::default();
        let mut ply = 0usize;

        for mv_str in &self.moves {
            let player_idx = ply % 4;
            let _player = players[player_idx];
            if player_idx == 0 {
                let full = ply / 4 + 1;
                out += &format!("{}.", full);
            }
            out += &format!(" {}", mv_str);

            // Advance board so next iteration knows who moves
            if let Ok(mv) = parse_move(&board, mv_str) {
                board = crate::rules::Rules::apply_with_effects(&board, mv);
            }
            ply += 1;
        }

        if !self.result.is_empty() { out += &format!(" {}", self.result); }
        out.push('\n');
        out
    }

    /// Parse a PGN-like string into a `GameRecord`.
    /// Only parses the move list; headers are extracted but not validated.
    pub fn from_pgn(pgn: &str) -> Result<Self, String> {
        let mut rec = GameRecord::default();
        let mut in_moves = false;

        for line in pgn.lines() {
            let line = line.trim();
            if line.starts_with('[') {
                // Parse header tag
                if let Some(rest) = line.strip_prefix('[') {
                    let rest = rest.trim_end_matches(']');
                    let mut parts = rest.splitn(2, ' ');
                    let tag = parts.next().unwrap_or("").trim();
                    let val = parts.next().unwrap_or("")
                        .trim()
                        .trim_matches('"')
                        .to_string();
                    match tag {
                        "Event"  => rec.event  = val,
                        "Site"   => rec.site   = val,
                        "Date"   => rec.date   = val,
                        "Result" => rec.result = val,
                        _        => {}
                    }
                }
            } else if !line.is_empty() {
                in_moves = true;
                // Tokenize: strip move numbers (e.g. "1." "2."), collect move tokens
                for token in line.split_whitespace() {
                    if token.ends_with('.') { continue; }      // move number
                    if token == rec.result  { continue; }      // result token
                    if token.len() >= 4 { rec.moves.push(token.to_string()); }
                }
            }
        }

        if !in_moves && rec.moves.is_empty() {
            return Err("no moves found in PGN".to_string());
        }
        Ok(rec)
    }

    /// Replay all moves from a PGN string, returning the final board.
    pub fn replay(pgn: &str) -> Result<Board, String> {
        let rec = Self::from_pgn(pgn)?;
        let mut board = Board::default();
        for mv_str in &rec.moves {
            let mv = parse_move(&board, mv_str)?;
            board = crate::rules::Rules::apply_with_effects(&board, mv);
        }
        Ok(board)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_roundtrip() {
        for s in 0u8..64 {
            let name = square_name(s);
            assert_eq!(parse_square(&name), Some(s), "roundtrip failed for sq {s}");
        }
    }

    #[test]
    fn parse_legal_pawn_move() {
        let b = Board::default();
        // Red pawn on a2 can go to a3
        let mv = parse_move(&b, "a2a3").expect("a2a3 should be legal");
        assert_eq!(mv.from, sq(0,1));
        assert_eq!(mv.to,   sq(0,2));
        assert!(!mv.promoted);
    }

    #[test]
    fn parse_illegal_move_err() {
        let b = Board::default();
        assert!(parse_move(&b, "a1a5").is_err()); // king can't jump
    }

    #[test]
    fn pgn_roundtrip() {
        let pgn = "[Event \"Test\"]\n[Result \"*\"]\n\n1. a2a3 g1g2 h7h6 b5b4 *\n";
        let rec = GameRecord::from_pgn(pgn).expect("parse failed");
        assert_eq!(rec.event, "Test");
        assert_eq!(rec.moves.len(), 4);
    }
}
