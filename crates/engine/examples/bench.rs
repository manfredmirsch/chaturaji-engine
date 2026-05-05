use chaturaji_core::board::Board;
use chaturaji_engine::{Engine, OpeningBook};
use std::time::Instant;

fn main() {
    let book_path = std::env::args().nth(1);
    let board     = Board::default();
    let mut engine = Engine::new(64);

    if let Some(p) = book_path.as_deref() {
        match OpeningBook::load(p) {
            Ok(b) => {
                println!("Buch geladen: {} Stellungen aus {}", b.len(), p);
                engine.set_book(b);
            }
            Err(e) => eprintln!("Konnte Buch '{}' nicht laden: {}", p, e),
        }
    }

    let t0 = Instant::now();
    let r  = engine.search(&board, 4);
    let dt = t0.elapsed();

    if r.depth == 0 && r.nodes == 0 {
        println!("Buchzug verwendet: {:?} (keine Suche, {:?})", r.best_move, dt);
    } else {
        let nps = (r.nodes as f64 / dt.as_secs_f64()) as u64;
        println!("Suche: depth={} nodes={} time={:?} nps={}", r.depth, r.nodes, dt, nps);
    }
}
