//! Was ein erster Zug der eigenen Stellung bringt: Beweglichkeit vorher/nachher,
//! aufgeschlüsselt nach Figur. Gerechnet wird für jeden Sitz auf der
//! Startstellung, so als wäre er am Zug — die drei anderen haben zu diesem
//! Zeitpunkt noch nichts bewegt, was die eigenen Linien berührt.

use std::collections::BTreeMap;

use chaturaji_core::board::Board;
use chaturaji_core::notation::move_to_str;
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;

fn mobility(board: &Board, seat: Color) -> BTreeMap<String, usize> {
    let mut b = board.clone();
    b.to_move = seat;
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    for mv in Rules::legal_moves(&b) {
        let kind = b.piece_at(mv.from).map(|p| p.kind).unwrap_or(PieceKind::Pawn);
        *out.entry(format!("{kind:?}")).or_default() += 1;
    }
    out
}

fn main() {
    let start = Board::default();
    println!("{{");
    for (n, seat) in Color::ALL.iter().enumerate() {
        let before = mobility(&start, *seat);
        let total_before: usize = before.values().sum();
        let mut b = start.clone();
        b.to_move = *seat;
        println!("  \"{}\": {{ \"mobility_before\": {total_before}, \"moves\": [", seat.name());
        let moves = Rules::legal_moves(&b);
        for (i, mv) in moves.iter().enumerate() {
            let after_board = Rules::apply_with_effects(&b, *mv);
            let after = mobility(&after_board, *seat);
            let total_after: usize = after.values().sum();
            let mut gains: Vec<String> = Vec::new();
            for kind in ["Boat", "Knight", "Bishop", "King", "Pawn"] {
                let d = *after.get(kind).unwrap_or(&0) as i64
                      - *before.get(kind).unwrap_or(&0) as i64;
                if d != 0 { gains.push(format!("\"{kind}\": {d}")); }
            }
            let comma = if i + 1 == moves.len() { "" } else { "," };
            let kind = b.piece_at(mv.from).map(|p| format!("{:?}", p.kind)).unwrap_or_default();
            println!("    {{\"move\": \"{}\", \"piece\": \"{}\", \"mobility_after\": {}, \"delta\": {}, \"gain\": {{{}}}}}{}",
                     move_to_str(mv),
                     kind,
                     total_after, total_after as i64 - total_before as i64,
                     gains.join(", "), comma);
        }
        let comma = if n + 1 == Color::ALL.len() { "" } else { "," };
        println!("  ]}}{comma}");
    }
    println!("}}");
}
