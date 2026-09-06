//! Compare the three search algorithms at equal *look-ahead in game rounds*.
//!
//! One round = every player has moved once. Paranoid and Max^n spend four plies
//! on a round; BRS spends two, because only the single best-replying opponent
//! moves and the other two pass. Comparing nominal depths is therefore
//! misleading — the fair comparison is paranoid/Max^n depth 4k against BRS
//! depth 2k, which is what this does.
//!
//! Usage: `cargo run --release -p chaturaji-engine --example brs_bench -- [rounds]`
//!
//! Start position, rounds = 2:
//! ```text
//! algo         depth       rounds        nodes         ms
//! paranoid         8            2      1234866       5211
//! brs              4            2         2229          7
//! maxn             8            2     73694748     241266
//! ```

use std::time::Instant;

use chaturaji_core::board::Board;
use chaturaji_engine::Engine;

fn main() {
    let board  = Board::default();
    let rounds: u8 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("{:<10} {:>7} {:>12} {:>12} {:>10}", "algo", "depth", "rounds", "nodes", "ms");

    let mut row = |name: &str, depth: u8, nodes: u64, ms: u128| {
        println!("{name:<10} {depth:>7} {rounds:>12} {nodes:>12} {ms:>10}");
    };

    let mut e = Engine::new(16);
    let t = Instant::now();
    let r = e.search_paranoid(&board, 4 * rounds, None);
    row("paranoid", 4 * rounds, r.nodes, t.elapsed().as_millis());

    let mut e = Engine::new(16);
    let t = Instant::now();
    let r = e.search_brs(&board, 2 * rounds, None);
    row("brs", 2 * rounds, r.nodes, t.elapsed().as_millis());

    let mut e = Engine::new(16);
    let t = Instant::now();
    let r = e.search(&board, 4 * rounds);
    row("maxn", 4 * rounds, r.nodes, t.elapsed().as_millis());
}
