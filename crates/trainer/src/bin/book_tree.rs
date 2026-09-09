//! Das Eröffnungsbuch von der Startstellung aus begehbar machen.
//!
//! Das Buch ist nach Zobrist-Hash abgelegt — von außen sagt es also nicht,
//! *welche* Stellung ein Eintrag meint. Dieses Werkzeug läuft von
//! `Board::default()` los, spielt die Buchzüge nach und schreibt den so
//! erreichbaren Anfang als JSON-Baum heraus: je Zug Häufigkeit, Ø-Platz,
//! Ø-Punkte, Ø-Rating-Differenz und Ø-Rating des Ziehenden.
//!
//! Verwendung:
//!   cargo run --release --bin book_tree -- \
//!     --in opening_book.json --out book_tree.json [--max-ply 8] \
//!     [--min-count 1] [--expand-min 20]

use std::collections::HashMap;

use chaturaji_core::board::{Board, Move};
use chaturaji_core::notation::move_to_str;
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};
use chaturaji_engine::book::{MoveStats, OpeningBook};

fn main() {
    let mut in_path    = "opening_book.json".to_string();
    let mut out_path   = "book_tree.json".to_string();
    let mut max_ply    = 8usize;
    let mut min_count  = 1u32;
    let mut expand_min = 20u32;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let next = |i: usize| args.get(i + 1).cloned();
        match args[i].as_str() {
            "--in"         => { if let Some(v) = next(i) { in_path  = v; } i += 1; }
            "--out"        => { if let Some(v) = next(i) { out_path = v; } i += 1; }
            "--max-ply"    => { if let Some(v) = next(i) { max_ply    = v.parse().unwrap_or(max_ply); }    i += 1; }
            "--min-count"  => { if let Some(v) = next(i) { min_count  = v.parse().unwrap_or(min_count); }  i += 1; }
            "--expand-min" => { if let Some(v) = next(i) { expand_min = v.parse().unwrap_or(expand_min); } i += 1; }
            other => eprintln!("unbekanntes Argument: {other}"),
        }
        i += 1;
    }

    let book = match chaturaji_trainer::opening_book::load(&in_path) {
        Ok(b) => b,
        Err(e) => { eprintln!("'{in_path}' nicht lesbar: {e}"); std::process::exit(1); }
    };
    let keys = ZobristKeys::new();

    eprintln!("Buch: {} Stellungen", book.len());

    let mut json = String::from("{\n");
    let root = Board::default();
    json.push_str(&format!("  \"positions_in_book\": {},\n", book.len()));
    json.push_str("  \"root\": ");
    json.push_str(&node_json(&book, &keys, &root, 0, max_ply, min_count, expand_min, 1));
    json.push_str("\n}\n");

    if let Err(e) = std::fs::write(&out_path, json) {
        eprintln!("Schreiben von '{out_path}' fehlgeschlagen: {e}");
        std::process::exit(1);
    }
    eprintln!("geschrieben: {out_path}");
}

/// Ein Knoten: die Stellung nach `ply` Halbzügen samt aller Buchzüge daraus.
#[allow(clippy::too_many_arguments)]
fn node_json(
    book:       &OpeningBook,
    keys:       &ZobristKeys,
    board:      &Board,
    ply:        usize,
    max_ply:    usize,
    min_count:  u32,
    expand_min: u32,
    indent:     usize,
) -> String {
    let pad  = "  ".repeat(indent);
    let pad2 = "  ".repeat(indent + 1);

    let entries: Vec<(Move, &MoveStats)> = match book.positions.get(&hash_board(board, keys)) {
        Some(stats) => collect(board, stats, min_count),
        None        => Vec::new(),
    };
    let total: u32 = entries.iter().map(|(_, s)| s.count).sum();

    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("{pad2}\"ply\": {ply},\n"));
    s.push_str(&format!("{pad2}\"to_move\": \"{}\",\n", board.to_move.name()));
    s.push_str(&format!("{pad2}\"games\": {total},\n"));
    s.push_str(&format!("{pad2}\"moves\": ["));

    for (n, (mv, st)) in entries.iter().enumerate() {
        if n > 0 { s.push(','); }
        s.push('\n');
        let child = Rules::apply_with_effects(board, *mv);
        let pad3 = "  ".repeat(indent + 2);
        s.push_str(&format!("{pad3}{{"));
        s.push_str(&format!("\"move\": \"{}\", ", move_to_str(mv)));
        s.push_str(&format!("\"from\": {}, \"to\": {}, ", mv.from, mv.to));
        let kind = board.piece_at(mv.from).map(|p| format!("{:?}", p.kind))
                        .unwrap_or_else(|| "?".to_string());
        s.push_str(&format!("\"piece\": \"{kind}\", "));
        s.push_str(&format!("\"capture\": {}, ", mv.captured.is_some()));
        s.push_str(&format!("\"count\": {}, ", st.count));
        s.push_str(&format!("\"avg_rank\": {:.4}, ",        st.avg_rank()));
        s.push_str(&format!("\"avg_points\": {:.3}, ",      st.avg_points()));
        s.push_str(&format!("\"avg_rating_diff\": {:.3}, ", st.avg_rating_diff()));
        s.push_str(&format!("\"avg_rating\": {:.1}",        st.avg_rating()));
        if ply + 1 < max_ply && st.count >= expand_min {
            s.push_str(", \"next\": ");
            s.push_str(&node_json(book, keys, &child, ply + 1, max_ply,
                                  min_count, expand_min, indent + 3));
        }
        s.push('}');
    }
    if !entries.is_empty() { s.push('\n'); s.push_str(&pad2); }
    s.push_str("]\n");
    s.push_str(&format!("{pad}}}"));
    s
}

/// Buchzüge dieser Stellung, auf echte `Move`s abgebildet und nach Häufigkeit
/// sortiert. Schlüssel, die hier nicht legal sind, fallen weg — das wäre ein
/// Zeichen dafür, dass Buch und Regeln auseinanderlaufen.
fn collect<'a>(
    board:     &Board,
    stats:     &'a HashMap<String, MoveStats>,
    min_count: u32,
) -> Vec<(Move, &'a MoveStats)> {
    let legal = Rules::legal_moves(board);
    let mut out: Vec<(Move, &MoveStats)> = stats.iter()
        .filter(|(_, s)| s.count >= min_count)
        .filter_map(|(key, s)| {
            let mut parts = key.splitn(2, '-');
            let from: u8 = parts.next()?.parse().ok()?;
            let to:   u8 = parts.next()?.parse().ok()?;
            let mv = legal.iter().find(|m| m.from == from && m.to == to)?;
            Some((*mv, s))
        })
        .collect();
    // Vollständige Ordnung, nicht nur nach Häufigkeit: bei gleichem `count` und
    // gleichem Ausgangsfeld entschiede sonst die HashMap-Reihenfolge, und
    // dieselbe Datei ergäbe je Lauf einen anders sortierten Baum.
    out.sort_by(|a, b| b.1.count.cmp(&a.1.count)
        .then(a.0.from.cmp(&b.0.from))
        .then(a.0.to.cmp(&b.0.to)));
    out
}
