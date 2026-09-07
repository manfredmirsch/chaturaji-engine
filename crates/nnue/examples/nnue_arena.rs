//! Turnier zweier NNUE-Netze gegeneinander.
//!
//! # Warum nicht einfach „A gegen B"
//!
//! Chaturaji ist kein symmetrisches Duell. Rot zieht zuerst, die vier Sitze
//! haben unterschiedliche Geometrie, und mit vier Spielern am Tisch entscheidet
//! auch mit, *neben wem* man sitzt. Der Aufbau ist deshalb derselbe wie in
//! `chaturaji-engine/examples/arena.rs`:
//!
//!   * Jede Partie besetzt zwei Sitze mit A und zwei mit B.
//!   * Alle sechs Aufteilungen von vier Sitzen auf 2+2 werden durchlaufen,
//!     sodass jedes Netz über sechs Partien jeden Sitz genau dreimal einnimmt.
//!   * Die sechs Partien einer Gruppe starten aus **derselben** zufälligen
//!     Eröffnung. Damit ist der Vergleich gepaart: der Unterschied je Gruppe
//!     misst die Netze, nicht die Eröffnung.
//!
//! # Wertung
//!
//! Gewertet wird die Platzierung, umgerechnet über `outcome::place_values`
//! (Platz 1 → +1, Platz 4 → −1) — dieselbe Größe, auf die auch trainiert wird.
//! Je Gruppe ergibt sich eine Differenz A−B; ausgewiesen werden ihr Mittel und
//! der Standardfehler über die Gruppen.
//!
//! # Ein Lauf ist kein Befund
//!
//! Der Standardfehler misst die Streuung über die Eröffnungen *eines* Seeds.
//! Zwischen zwei Seeds liegt erfahrungsgemäß noch einmal mehr. Ergebnisse
//! gehören über mehrere `--seed` wiederholt.
//!
//! Aufruf:
//!   cargo run --release -p chaturaji-nnue --example nnue_arena -- \
//!       --a weights.json --b weights-pretrained.json --groups 20 --depth 4 --beam-width 6

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::ZobristKeys;
use chaturaji_nnue::network::NnueNetwork;
use chaturaji_nnue::outcome::place_values;
use chaturaji_nnue::selfplay::nnue_best_move;

/// Die sechs Aufteilungen von vier Sitzen auf 2+2. Der Eintrag nennt die Sitze,
/// die Netz A besetzt; die beiden anderen gehören B.
const SPLITS: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];

struct Args {
    a: String,
    b: String,
    groups: usize,
    depth: u8,
    beam: usize,
    max_moves: usize,
    opening_plies: usize,
    seed: u64,
    shards: usize,
    shard: usize,
    out: String,
}

fn parse_args() -> Args {
    let v: Vec<String> = std::env::args().collect();
    let mut a = Args {
        a: "weights.json".into(),
        b: "weights-pretrained.json".into(),
        groups: 12,
        depth: 4,
        beam: 6,
        max_moves: 150,
        opening_plies: 8,
        seed: 1,
        shards: 1,
        shard: 0,
        out: String::new(),
    };
    let mut i = 1;
    while i < v.len() {
        let mut next = |i: &mut usize| { *i += 1; v.get(*i).cloned().unwrap_or_default() };
        match v[i].as_str() {
            "--a"             => a.a = next(&mut i),
            "--b"             => a.b = next(&mut i),
            "--groups"        => a.groups = next(&mut i).parse().unwrap_or(a.groups),
            "--depth"         => a.depth = next(&mut i).parse().unwrap_or(a.depth),
            "--beam-width"    => a.beam = next(&mut i).parse().unwrap_or(a.beam),
            "--max-moves"     => a.max_moves = next(&mut i).parse().unwrap_or(a.max_moves),
            "--opening-plies" => a.opening_plies = next(&mut i).parse().unwrap_or(a.opening_plies),
            "--seed"          => a.seed = next(&mut i).parse().unwrap_or(a.seed),
            "--shards"        => a.shards = next(&mut i).parse().unwrap_or(a.shards),
            "--shard"         => a.shard = next(&mut i).parse().unwrap_or(a.shard),
            "--out"           => a.out = next(&mut i),
            _ => {}
        }
        i += 1;
    }
    a
}

fn load(path: &str) -> NnueNetwork {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("'{path}' nicht lesbar: {e}"));
    let mut net: NnueNetwork = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("'{path}' ist kein NNUE-Netz: {e}"));
    net.ensure_input_size();
    net
}

/// Eine zufällige Eröffnung. Ohne sie spielten zwei deterministische Netze in
/// jeder Gruppe dieselbe Partie.
fn opening(plies: usize, seed: u64) -> Board {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut board = Board::default();
    for _ in 0..plies {
        if Rules::is_game_over(&board) { break; }
        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }
        board = Rules::apply_with_effects(&board, moves[rng.gen_range(0..moves.len())]);
    }
    board
}

/// Spielt eine Partie aus der gegebenen Stellung. `a_seats` sind die Sitze von
/// Netz A. Rückgabe: Platzwerte je Sitz und die Zahl der gespielten Halbzüge.
fn play(
    start: &Board, a_seats: [usize; 2],
    net_a: &NnueNetwork, net_b: &NnueNetwork,
    depth: u8, beam: usize, max_moves: usize, keys: &ZobristKeys,
) -> ([f32; 4], usize) {
    let mut board = start.clone();
    let mut tt: HashMap<u64, (u8, [f32; 4])> = HashMap::new();
    let mut plies = 0;

    while plies < max_moves && !Rules::is_game_over(&board) {
        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }
        let net = if a_seats.contains(&board.to_move.idx()) { net_a } else { net_b };
        tt.clear();
        let mv = nnue_best_move(net, &board, &moves, depth, beam, keys, &mut tt);
        board = Rules::apply_with_effects(&board, mv);
        plies += 1;
    }

    (place_values(Rules::final_scores(&board)), plies)
}

fn main() {
    let args = parse_args();
    let net_a = load(&args.a);
    let net_b = load(&args.b);
    let keys = ZobristKeys::new();

    // Gruppen dieses Shards: reihum, damit jeder Shard dieselbe Mischung an
    // Eröffnungen sieht.
    let groups: Vec<usize> = (0..args.groups)
        .filter(|g| g % args.shards.max(1) == args.shard)
        .collect();

    println!("A: {}  ({} Schritte)", args.a, net_a.steps);
    println!("B: {}  ({} Schritte)", args.b, net_b.steps);
    println!("{} Gruppen à 6 Partien | Tiefe {} Beam {} | Seed {}",
             groups.len(), args.depth, args.beam, args.seed);
    println!("{}", "-".repeat(64));

    // Je Gruppe die gepaarte Differenz A−B über die sechs Sitzaufteilungen.
    let results: Vec<(f64, f64, f64, f64)> = groups
        .par_iter()
        .map(|&g| {
            let start = opening(args.opening_plies, args.seed.wrapping_mul(1_000_003) ^ g as u64);
            let (mut sa, mut sb, mut plies) = (0.0f64, 0.0f64, 0.0f64);
            for split in SPLITS {
                let (vals, p) = play(&start, split, &net_a, &net_b,
                                     args.depth, args.beam, args.max_moves, &keys);
                for seat in 0..4 {
                    if split.contains(&seat) { sa += vals[seat] as f64; }
                    else                     { sb += vals[seat] as f64; }
                }
                plies += p as f64;
            }
            // 6 Partien × 2 Sitze = 12 Messwerte je Netz und Gruppe.
            (sa / 12.0, sb / 12.0, (sa - sb) / 12.0, plies / 6.0)
        })
        .collect();

    let n = results.len() as f64;
    if n == 0.0 { println!("Keine Gruppen in diesem Shard."); return; }

    let mean_a: f64 = results.iter().map(|r| r.0).sum::<f64>() / n;
    let mean_b: f64 = results.iter().map(|r| r.1).sum::<f64>() / n;
    let diffs: Vec<f64> = results.iter().map(|r| r.2).collect();
    let mean_d: f64 = diffs.iter().sum::<f64>() / n;
    let var: f64 = if n > 1.0 {
        diffs.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / (n - 1.0)
    } else { 0.0 };
    let se = (var / n).sqrt();
    let plies: f64 = results.iter().map(|r| r.3).sum::<f64>() / n;

    println!("Ø Platzwert A : {mean_a:+.4}");
    println!("Ø Platzwert B : {mean_b:+.4}");
    println!("Differenz A−B : {mean_d:+.4}  ± {se:.4} (SE)");
    if se > 0.0 { println!("t             : {:+.2}", mean_d / se); }
    println!("Ø Halbzüge    : {plies:.1}");

    if !args.out.is_empty() {
        // Summen statt Mittel: nur so lassen sich Shards korrekt zusammenlegen.
        let sum_d: f64 = diffs.iter().sum();
        let sum_d2: f64 = diffs.iter().map(|d| d * d).sum();
        let json = format!(
            "{{\"groups\":{},\"sum_a\":{},\"sum_b\":{},\"sum_d\":{},\"sum_d2\":{},\"sum_plies\":{}}}",
            results.len(),
            results.iter().map(|r| r.0).sum::<f64>(),
            results.iter().map(|r| r.1).sum::<f64>(),
            sum_d, sum_d2,
            results.iter().map(|r| r.3).sum::<f64>(),
        );
        std::fs::write(&args.out, json).expect("Ergebnisdatei nicht schreibbar");
        println!("Geschrieben: {}", args.out);
    }
}
