//! Eröffnungsbuch — vom Trainer aus echten Spielen gebaut, von der Engine
//! konsumiert. Schlüssel ist der Zobrist-Hash der Stellung; pro Hash sind
//! die historisch gespielten Züge mit ihren Outcome-Stats abgelegt.
//!
//! Dieselbe `ZobristKeys::new()`-Instanz erzeugt deterministisch identische
//! Hashes — Trainer und Engine sehen also dasselbe Buch.

use std::collections::HashMap;
use std::fs;

use serde::{Deserialize, Serialize};

use chaturaji_core::board::{Board, Move};
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MoveStats {
    /// Wie oft wurde dieser Zug aus der Stellung gespielt.
    pub count: u32,
    /// Summe der Spielpunkte des ziehenden Spielers (am Spielende).
    pub sum_points: i64,
    /// Summe der Plätze (1 = bester, 4 = schlechtester).
    pub sum_rank: u32,
    /// Summe der Rating-Differenzen.
    pub sum_rating_diff: f64,
    /// Summe der Pre-Game-Ratings (für später optionale Gewichtung).
    pub sum_rating: f64,
}

impl MoveStats {
    pub fn avg_rank(&self)        -> f64 { self.sum_rank        as f64 / self.count as f64 }
    pub fn avg_points(&self)      -> f64 { self.sum_points      as f64 / self.count as f64 }
    pub fn avg_rating_diff(&self) -> f64 { self.sum_rating_diff        / self.count as f64 }
    pub fn avg_rating(&self)      -> f64 { self.sum_rating              / self.count as f64 }
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct OpeningBook {
    /// Stellung-Hash → { Zug-String "from-to" → Stats }.
    pub positions: HashMap<u64, HashMap<String, MoveStats>>,
}

impl OpeningBook {
    /// Lädt ein zuvor vom Trainer geschriebenes Buch.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let book: OpeningBook = serde_json::from_str(&text)?;
        Ok(book)
    }

    /// Schreibt das Buch als JSON.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        fs::write(path, serde_json::to_string(self)?)
    }

    /// Anzahl der erfassten Stellungen.
    pub fn len(&self) -> usize { self.positions.len() }
    pub fn is_empty(&self) -> bool { self.positions.is_empty() }

    /// Suche den besten Buchzug für `board`. Liefert nur Züge, die
    /// mindestens `min_count` Mal historisch gespielt wurden — alles darunter
    /// ist statistisches Rauschen. Bewertung: niedrigerer Ø-Platz ist besser
    /// (1 = bester), Ø-Punkte als Tiebreaker.
    pub fn probe(&self, board: &Board, keys: &ZobristKeys, min_count: u32) -> Option<Move> {
        let hash  = hash_board(board, keys);
        let stats = self.positions.get(&hash)?;
        let key   = pick_best(stats, min_count)?;
        decode_move(board, key)
    }

    /// Liefert alle Buchzüge für `board` als (from, to, count)-Tupel.
    /// Nur Züge mit `count >= min_count`. Für gewichtetes Sampling im Trainer.
    pub fn entries(&self, board: &Board, keys: &ZobristKeys, min_count: u32)
        -> Option<Vec<(u8, u8, u32)>>
    {
        let hash  = hash_board(board, keys);
        let stats = self.positions.get(&hash)?;
        let mut out = Vec::with_capacity(stats.len());
        for (key, s) in stats {
            if s.count < min_count { continue; }
            let mut parts = key.splitn(2, '-');
            let from: u8 = parts.next()?.parse().ok()?;
            let to:   u8 = parts.next()?.parse().ok()?;
            out.push((from, to, s.count));
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

fn pick_best<'a>(stats: &'a HashMap<String, MoveStats>, min_count: u32) -> Option<&'a str> {
    stats.iter()
        .filter(|(_, s)| s.count >= min_count)
        .max_by(|a, b| {
            score(a.1).partial_cmp(&score(b.1)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(k, _)| k.as_str())
}

/// Score-Funktion: niedriger Ø-Platz besser (negativ rein), Ø-Punkte als
/// Tiebreaker mit kleinem Gewicht.
fn score(s: &MoveStats) -> f64 {
    -s.avg_rank() + s.avg_points() * 0.01
}

/// Wandelt den "from-to"-String in einen tatsächlichen `Move` aus den legal
/// moves der Stellung um (so dass `mover`, `captured` und `promoted` korrekt
/// gesetzt sind).
fn decode_move(board: &Board, key: &str) -> Option<Move> {
    let mut parts = key.splitn(2, '-');
    let from: u8  = parts.next()?.parse().ok()?;
    let to: u8    = parts.next()?.parse().ok()?;
    Rules::legal_moves(board)
        .into_iter()
        .find(|m| m.from == from && m.to == to)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::piece::Color;

    fn dummy_stats(count: u32, sum_rank: u32) -> MoveStats {
        MoveStats { count, sum_rank, sum_points: 0, sum_rating_diff: 0.0, sum_rating: 0.0 }
    }

    #[test]
    fn pick_best_filters_min_count() {
        let mut s = HashMap::new();
        // Kleine Stichprobe mit super Score → soll ignoriert werden.
        s.insert("0-8".to_string(),  dummy_stats(2, 2));
        // Größere Stichprobe mit etwas schlechterem Score → soll gewinnen.
        s.insert("1-9".to_string(),  dummy_stats(50, 110));
        let best = pick_best(&s, 5).unwrap();
        assert_eq!(best, "1-9");
    }

    #[test]
    fn probe_unknown_position_returns_none() {
        let book = OpeningBook::default();
        let keys = ZobristKeys::new();
        let b    = Board::default();
        assert!(book.probe(&b, &keys, 1).is_none());
    }

    #[test]
    fn probe_returns_legal_move_when_position_known() {
        // Künstliches Buch mit nur einer Stellung (die Startposition) und
        // einem legalen Zug: Bauer d2 (sq 11) → d3 (sq 19).
        let keys = ZobristKeys::new();
        let b    = Board::default();
        let hash = hash_board(&b, &keys);

        let mut moves = HashMap::new();
        moves.insert(
            format!("{}-{}", 11u8, 19u8),
            MoveStats { count: 100, sum_rank: 200, sum_points: 1500,
                        sum_rating_diff: 0.0, sum_rating: 0.0 },
        );
        let mut book = OpeningBook::default();
        book.positions.insert(hash, moves);

        let mv = book.probe(&b, &keys, 5)
            .expect("probe must return a move for the start position");
        assert_eq!(mv.from, 11);
        assert_eq!(mv.to,   19);
        assert_eq!(mv.mover, Color::Red);
    }
}
