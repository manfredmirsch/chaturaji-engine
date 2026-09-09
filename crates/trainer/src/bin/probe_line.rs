//! Eine Eröffnungslinie abspielen und für den Ziehenden jeden Zug danach
//! befragen: Wie viel Beweglichkeit gewinnt er, und was steht danach für die
//! drei anderen zum Schlagen bereit?
//!
//!   cargo run --release --bin probe_line -- d2d3 [b6c6 ...]

use chaturaji_core::board::Board;
use chaturaji_core::notation::{move_to_str, parse_move, square_name};
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;

fn mobility(board: &Board, seat: Color) -> usize {
    let mut b = board.clone();
    b.to_move = seat;
    Rules::legal_moves(&b).len()
}

/// Was der Sitz `by` in dieser Stellung unmittelbar schlagen könnte.
fn captures_for(board: &Board, by: Color) -> Vec<(PieceKind, u8, PieceKind)> {
    let mut b = board.clone();
    b.to_move = by;
    Rules::legal_moves(&b).into_iter()
        .filter_map(|mv| {
            let victim = mv.captured?;
            let attacker = b.piece_at(mv.from)?;
            Some((attacker.kind, mv.to, victim.kind))
        })
        .collect()
}

fn main() {
    let line: Vec<String> = std::env::args().skip(1).collect();
    let mut board = Board::default();
    for tok in &line {
        match parse_move(&board, tok) {
            Ok(mv) => board = Rules::apply_with_effects(&board, mv),
            Err(e) => { eprintln!("Zug '{tok}' nicht spielbar: {e}"); std::process::exit(1); }
        }
    }
    let seat = board.to_move;
    println!("Linie: {}   am Zug: {}", line.join(" "), seat.name());

    let before = mobility(&board, seat);
    let mut rows: Vec<(String, i64, String)> = Vec::new();
    for mv in Rules::legal_moves(&board) {
        let after = Rules::apply_with_effects(&board, mv);
        let dm = mobility(&after, seat) as i64 - before as i64;

        // Wer kann danach was von uns schlagen?
        let mut threats: Vec<String> = Vec::new();
        for other in Color::ALL {
            if other == seat { continue; }
            for (att, to, victim) in captures_for(&after, other) {
                if let Some(p) = after.piece_at(to) {
                    if p.color == seat {
                        threats.push(format!("{} {:?}x{:?}@{}", other.name(), att, victim, square_name(to)));
                    }
                }
            }
        }
        rows.push((move_to_str(&mv), dm, threats.join(", ")));
    }
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    for (mv, dm, threats) in rows {
        println!("  {mv:6} Δ-Beweglichkeit {dm:+3}   bedroht: {}",
                 if threats.is_empty() { "—".to_string() } else { threats });
    }
}
