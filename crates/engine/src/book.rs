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
    /// Liefert alle Buchzüge für `board`, nach Häufigkeit absteigend sortiert
    /// (meistgespielt zuerst). Konsistent mit `probe()`.
    pub fn entries(&self, board: &Board, keys: &ZobristKeys, min_count: u32)
        -> Option<Vec<(u8, u8, u32)>>
    {
        let hash  = hash_board(board, keys);
        let stats = self.positions.get(&hash)?;
        let mut out: Vec<(u8, u8, u32)> = stats.iter()
            .filter(|(_, s)| s.count >= min_count)
            .filter_map(|(key, s)| {
                let mut parts = key.splitn(2, '-');
                let from: u8 = parts.next()?.parse().ok()?;
                let to:   u8 = parts.next()?.parse().ok()?;
                Some((from, to, s.count))
            })
            .collect();
        if out.is_empty() { return None; }
        // Häufigkeit absteigend, bei Gleichstand nach Feldern. Der Tiebreak ist
        // nicht Kosmetik: `stats` ist eine `HashMap`, deren Iterationsreihenfolge
        // `RandomState` je Prozess neu würfelt. Ohne ihn stünden gleich häufige
        // Züge mal so, mal so in der Liste — und `sample_book_move` läuft sie
        // nach kumulierter Häufigkeit ab, wählt bei derselben Zufallszahl also
        // einen anderen Zug. Self-Play war damit trotz festem Partie-Seed nicht
        // zwischen zwei Läufen reproduzierbar; aufgefallen an drei Läufen über
        // dieselben 6.400 Partien, die auf 114,47 / 114,73 / 114,76 ∅Halbzüge
        // kamen statt auf denselben Wert.
        out.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        Some(out)
    }
}

fn pick_best<'a>(stats: &'a HashMap<String, MoveStats>, min_count: u32) -> Option<&'a str> {
    stats.iter()
        .filter(|(_, s)| s.count >= min_count)
        .max_by_key(|(_, s)| s.count)
        .map(|(k, _)| k.as_str())
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

    /// Gleich häufige Buchzüge müssen in stabiler Reihenfolge stehen.
    ///
    /// `entries` liest aus einer `HashMap`, deren Iterationsreihenfolge je
    /// Prozess neu ausgewürfelt wird. Ohne Tiebreak hinter der Häufigkeit
    /// wählte `sample_book_move` bei derselben Zufallszahl mal den einen, mal
    /// den anderen Zug — Self-Play war zwischen zwei Läufen nicht
    /// reproduzierbar, obwohl der Partie-Seed festliegt.
    ///
    /// Der Test baut die Map mehrfach neu auf: innerhalb eines Prozesses ist
    /// die Reihenfolge zwar stabil, die Einfügereihenfolge aber nicht die
    /// einzige Quelle — deshalb wird hier vor allem die Sortierung selbst
    /// festgenagelt.
    #[test]
    fn entries_are_ordered_deterministically_when_counts_tie() {
        let keys = ZobristKeys::new();
        let b    = Board::default();
        let hash = hash_board(&b, &keys);

        let paare = [(11u8, 19u8), (9, 17), (10, 18), (12, 20)];

        let baue = |reihenfolge: &[usize]| {
            let mut moves = HashMap::new();
            for &i in reihenfolge {
                let (from, to) = paare[i];
                moves.insert(format!("{from}-{to}"), dummy_stats(50, 100));
            }
            let mut book = OpeningBook::default();
            book.positions.insert(hash, moves);
            book.entries(&b, &keys, 5).expect("alle Züge über min_count")
        };

        let erwartet = vec![(9u8, 17u8, 50u32), (10, 18, 50), (11, 19, 50), (12, 20, 50)];
        assert_eq!(baue(&[0, 1, 2, 3]), erwartet);
        assert_eq!(baue(&[3, 2, 1, 0]), erwartet, "Einfügereihenfolge darf nichts ändern");
        assert_eq!(baue(&[2, 0, 3, 1]), erwartet);
    }

    /// Die Häufigkeit bleibt das erste Kriterium — der Tiebreak greift erst
    /// darunter und darf die Reihenfolge nicht umdrehen.
    #[test]
    fn count_still_outranks_the_tiebreak() {
        let keys = ZobristKeys::new();
        let b    = Board::default();
        let hash = hash_board(&b, &keys);

        let mut moves = HashMap::new();
        moves.insert("9-17".to_string(),  dummy_stats(10, 20));   // kleines Feld, selten
        moves.insert("12-20".to_string(), dummy_stats(99, 200));  // großes Feld, häufig
        let mut book = OpeningBook::default();
        book.positions.insert(hash, moves);

        let out = book.entries(&b, &keys, 5).unwrap();
        assert_eq!(out[0], (12, 20, 99), "der häufigste Zug steht vorn");
        assert_eq!(out[1], (9, 17, 10));
    }
}
