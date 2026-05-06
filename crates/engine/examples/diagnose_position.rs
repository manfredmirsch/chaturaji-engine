// Lädt ein Spiel aus game_analysis und prüft die Engine-Auswahl an einer
// bestimmten Halbzug-Position.
//
// Aufruf:
//   cargo run -p chaturaji-engine --release --example diagnose_position -- <pfad-zur-game-json> <ply-1-indexed> <max-depth>

use std::env;

use chaturaji_core::board::{Board, Move};
use chaturaji_core::piece::Color;
use chaturaji_core::rules::Rules;
use chaturaji_engine::Engine;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).expect("Pfad zur game JSON fehlt");
    let ply: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(18);
    let depth: u8 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(5);

    let raw = std::fs::read_to_string(path).expect("Datei lesen fehlgeschlagen");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("JSON parse");
    let pgn = v["pgn4"].as_str().expect("kein pgn4 im JSON");

    let tokens = collect_move_tokens(pgn);
    let mut board = Board::default();
    for (i, t) in tokens.iter().take(ply - 1).enumerate() {
        let mv = resolve_move(&board, t).unwrap_or_else(|| {
            eprintln!("Zug {} ({:?}) konnte nicht aufgelöst werden", i + 1, t);
            std::process::exit(1);
        });
        board = Rules::apply_with_effects(&board, mv);
    }

    println!("Stellung vor Halbzug {} – am Zug: {:?}", ply, board.to_move);
    println!("Aktive Spieler: {:?}", board.active);
    println!("Punkte: {:?}", board.scores);

    // König-Position aller Spieler
    use chaturaji_core::piece::PieceKind;
    for c in Color::ALL {
        let kbb = board.pieces(c, PieceKind::King);
        if kbb != 0 {
            let s = kbb.trailing_zeros() as u8;
            println!("  {:?} König auf {}", c, sq_name(s));
        }
    }

    print_board(&board);

    let mut engine = Engine::new(64);
    let res = engine.search(&board, depth);
    let mover_idx = board.to_move.idx();
    println!("\nEngine-Wahl bei Tiefe {}: {:?}  scores={:?}",
             depth,
             res.best_move.map(|m| (sq_name(m.from), sq_name(m.to))),
             res.scores);

    println!("\nAlle legalen Züge mit Sicherheitsstatus:");
    let mut all = chaturaji_core::rules::Rules::legal_moves(&board);
    all.sort_by_key(|m| (m.from, m.to));
    for m in &all {
        let after = chaturaji_core::rules::Rules::apply_with_effects(&board, *m);
        let king = after.pieces(board.to_move, chaturaji_core::piece::PieceKind::King);
        let mut atk_who = vec![];
        for opp in chaturaji_core::piece::Color::ALL {
            if opp == board.to_move || !after.active[opp.idx()] { continue; }
            if chaturaji_core::rules::Rules::attacked_squares(&after, opp) & king != 0 {
                atk_who.push(format!("{:?}", opp));
            }
        }
        let unsafe_ = chaturaji_engine::search::leaves_king_capturable(&board, *m);
        println!("  {}-{}  unsafe={}  king_attacked_by={:?}",
                 sq_name(m.from), sq_name(m.to), unsafe_, atk_who);
    }

    println!("\nTop 5 nach Sicherheitsfilter:");
    for r in engine.top_n(&board, depth, 5) {
        println!("  {}-{}  score[mover]={}",
                 sq_name(r.mv.from), sq_name(r.mv.to), r.scores[mover_idx]);
    }
}

fn collect_move_tokens(pgn: &str) -> Vec<String> {
    // alles nach den Header-Tags
    let body_start = pgn.rfind(']').map(|i| i + 1).unwrap_or(0);
    let body = &pgn[body_start..];
    // Klammer-Kommentare entfernen
    let mut clean = String::new();
    let mut depth: i32 = 0;
    for c in body.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => clean.push(c),
            _ => {}
        }
    }
    clean.split_whitespace()
        .filter(|t| !t.ends_with('.'))
        .filter(|t| *t != "..")
        .filter(|t| *t != "*")
        .map(String::from)
        .collect()
}

fn resolve_move(board: &Board, token: &str) -> Option<Move> {
    let (from, to) = chaturaji_trainer::pgn_import::parse_move_token(token)?;
    Rules::legal_moves(board)
        .into_iter()
        .find(|m| m.from == from && m.to == to)
}

fn sq_name(s: u8) -> String {
    let f = (b'a' + (s & 7)) as char;
    let r = (s >> 3) + 1;
    format!("{}{}", f, r)
}

fn print_board(b: &Board) {
    use chaturaji_core::piece::PieceKind;
    println!("\n  +---+---+---+---+---+---+---+---+");
    for r in (0..8u8).rev() {
        print!("{} |", r + 1);
        for f in 0..8u8 {
            let s = r * 8 + f;
            let mut sym = String::from(" . ");
            for c in Color::ALL {
                for k in [PieceKind::King, PieceKind::Boat, PieceKind::Bishop, PieceKind::Knight, PieceKind::Pawn] {
                    if b.bb[c.idx()][k.idx()] & (1u64 << s) != 0 {
                        let cl = match c { Color::Red => 'R', Color::Blue => 'B', Color::Yellow => 'Y', Color::Green => 'G' };
                        let kk = match k { PieceKind::King => 'K', PieceKind::Boat => 'O', PieceKind::Bishop => 'b', PieceKind::Knight => 'n', PieceKind::Pawn => 'p' };
                        sym = format!("{}{} ", cl, kk);
                    }
                }
            }
            print!("{}|", sym);
        }
        println!("\n  +---+---+---+---+---+---+---+---+");
    }
    println!("    a   b   c   d   e   f   g   h");
}
