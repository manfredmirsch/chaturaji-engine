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
use chaturaji_nnue::selfplay::{nnue_best_move, nnue_best_move_timed, BeamOrder};
use chaturaji_nnue::mcts::{Mcts, MctsConfig};
use chaturaji_engine::move_features::MoveModel;
use chaturaji_engine::Engine;
use chaturaji_engine::search::SearchAlgo;

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
    /// Zugsortierung im Beam, je Seite getrennt.
    ///
    /// Damit lässt sich die Sortierung selbst messen und nicht nur das Netz:
    /// `--a weights.json --b weights.json --a-beam model --b-beam legacy`
    /// spielt dasselbe Netz gegen sich, einmal so und einmal so sortiert.
    a_beam: BeamOrder,
    b_beam: BeamOrder,
    /// Zeitbudget je Zug in Millisekunden. 0 = aus, dann zählt `depth`.
    ///
    /// Mit Budget bekommen beide Seiten dieselbe Rechenzeit und vertiefen so
    /// weit sie kommen — die Spalte `Ø-Tiefe` zeigt, wie weit das war. Erst so
    /// lässt sich eine teurere Sortierung fair gegen eine billigere stellen.
    time_ms: u64,
    max_depth: u8,
    a_search: SearchKind,
    b_search: SearchKind,
    tt_mb: usize,
    /// Simulationen je Zug für MCTS. Kein Gegenstück zu `depth` — die Suche
    /// verteilt sie selbst über den Baum.
    iters: u32,
    c_puct: f32,
    // Seitenweise Überschreibungen. `None` heißt „nimm den gemeinsamen Wert".
    // Gebraucht, um zwei Einstellungen derselben Suche gegeneinander zu
    // messen — c_puct etwa war nie gemessen, nur aus der Wertskala hergeleitet.
    a_iters:  Option<u32>,
    b_iters:  Option<u32>,
    a_c_puct: Option<f32>,
    b_c_puct: Option<f32>,
}

/// Welches Suchverfahren eine Seite benutzt.
///
/// Beide bewerten Blätter mit demselben NNUE; sie unterscheiden sich darin,
/// welchen Baum sie aufspannen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchKind {
    /// Das Self-Play-Max^n mit Beam (`nnue_maxn`). Jeder der vier Spieler
    /// maximiert seine eigene Komponente; der Beam begrenzt die Verzweigung.
    Maxn,
    /// Best-Reply Search aus der Engine. Nur der stärkste Gegner antwortet,
    /// die beiden anderen passen — eine Runde kostet zwei Halbzüge statt vier.
    /// Deshalb ist `depth` zwischen beiden Verfahren **nicht** vergleichbar;
    /// vergleichbar sind Runden, also BRS-Tiefe 2k gegen Max^n-Tiefe 4k.
    Brs,
    /// Monte-Carlo-Baumsuche mit gelerntem Zug-Prior (`chaturaji_nnue::mcts`).
    /// Kennt keine Tiefe, sondern `--iters` Simulationen je Zug.
    Mcts,
}

fn search_kind(s: &str) -> SearchKind {
    match s {
        "maxn" | "beam" => SearchKind::Maxn,
        "brs"           => SearchKind::Brs,
        "mcts"          => SearchKind::Mcts,
        _ => {
            eprintln!("Unbekanntes Suchverfahren '{s}' — erlaubt sind 'maxn', 'brs' und 'mcts'.");
            std::process::exit(2);
        }
    }
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
        a_beam: BeamOrder::Model,
        b_beam: BeamOrder::Model,
        time_ms: 0,
        max_depth: 10,
        a_search: SearchKind::Maxn,
        b_search: SearchKind::Maxn,
        tt_mb: 8,
        iters: 400,
        c_puct: chaturaji_nnue::mcts::DEFAULT_C_PUCT,
        a_iters: None, b_iters: None, a_c_puct: None, b_c_puct: None,
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
            "--a-beam"        => a.a_beam = beam_order(&next(&mut i)),
            "--b-beam"        => a.b_beam = beam_order(&next(&mut i)),
            "--time-ms"       => a.time_ms = next(&mut i).parse().unwrap_or(a.time_ms),
            "--max-depth"     => a.max_depth = next(&mut i).parse().unwrap_or(a.max_depth),
            "--a-search"      => a.a_search = search_kind(&next(&mut i)),
            "--b-search"      => a.b_search = search_kind(&next(&mut i)),
            "--tt-mb"         => a.tt_mb = next(&mut i).parse().unwrap_or(a.tt_mb),
            "--iters"         => a.iters = next(&mut i).parse().unwrap_or(a.iters),
            "--c-puct"        => a.c_puct = next(&mut i).parse().unwrap_or(a.c_puct),
            "--a-iters"       => a.a_iters  = next(&mut i).parse().ok(),
            "--b-iters"       => a.b_iters  = next(&mut i).parse().ok(),
            "--a-c-puct"      => a.a_c_puct = next(&mut i).parse().ok(),
            "--b-c-puct"      => a.b_c_puct = next(&mut i).parse().ok(),
            _ => {}
        }
        i += 1;
    }
    a
}

/// Kein stilles Zurückfallen auf die Vorgabe: ein vertippter Wert würde sonst
/// als „beide Seiten gleich" durchgehen, und der Lauf misst dann nichts,
/// sieht aber aus wie ein Ergebnis.
fn beam_order(s: &str) -> BeamOrder {
    BeamOrder::from_str(s).unwrap_or_else(|| {
        eprintln!("Unbekannte Beam-Sortierung '{s}' — erlaubt sind 'model' und 'legacy'.");
        std::process::exit(2);
    })
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
#[allow(clippy::too_many_arguments)]
fn play(
    start: &Board, a_seats: [usize; 2],
    net_a: &NnueNetwork, net_b: &NnueNetwork,
    a_beam: BeamOrder, b_beam: BeamOrder,
    a_search: SearchKind, b_search: SearchKind, tt_mb: usize,
    mcts_a: &MctsConfig, mcts_b: &MctsConfig,
    depth: u8, beam: usize, max_moves: usize, keys: &ZobristKeys,
    time_ms: u64, max_depth: u8,
) -> ([f32; 4], usize, [f64; 2], [f64; 2]) {
    let mut board = start.clone();
    let mut tt: HashMap<u64, (u8, [f32; 4])> = HashMap::new();
    let mut plies = 0;
    // [Summe der Tiefen, Zahl der Züge] je Seite — für die Spalte `Ø-Tiefe`.
    let mut tiefe_a = [0.0f64; 2];
    let mut tiefe_b = [0.0f64; 2];
    // Je Seite eine eigene Engine, damit die Transpositionstabellen der beiden
    // Verfahren sich nicht vermischen. Nur angelegt, wenn die Seite BRS spielt.
    let mut eng_a = (a_search == SearchKind::Brs).then(|| Engine::new(tt_mb));
    let mut eng_b = (b_search == SearchKind::Brs).then(|| Engine::new(tt_mb));
    // Baum und Zugmodell einmal je Partie; der Baum wird je Zug geleert, die
    // Allokationen bleiben erhalten.
    let mut baum   = Mcts::new();
    let modell     = MoveModel::default();

    while plies < max_moves && !Rules::is_game_over(&board) {
        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }
        let ist_a = a_seats.contains(&board.to_move.idx());
        let net   = if ist_a { net_a } else { net_b };
        let order = if ist_a { a_beam } else { b_beam };
        // Die TT speichert Scorevektoren, die von der Beam-Auswahl abhängen.
        // Bei zwei verschiedenen Sortierungen in einer Partie dürfen sich die
        // Seiten die Einträge deshalb nicht teilen — sonst läse A Ergebnisse,
        // die B mit anderer Auswahl erzeugt hat.
        tt.clear();
        let kind = if ist_a { a_search } else { b_search };
        let mv = match kind {
            SearchKind::Brs => {
                let engine = if ist_a { eng_a.as_mut() } else { eng_b.as_mut() }
                    .expect("Engine wurde für diese Seite angelegt");
                if time_ms > 0 {
                    // Der Engine fehlt eine eigene Uhr — bewusst, weil `Instant`
                    // unter wasm32 paniziert. Die Zeitquelle stellt der Aufrufer.
                    let frist = std::time::Instant::now()
                        + std::time::Duration::from_millis(time_ms);
                    engine.set_stop_check(move || std::time::Instant::now() >= frist);
                } else {
                    engine.set_stop_check(|| false);
                }
                let eval = |b: &Board| net.forward(b);
                let tiefe_max = if time_ms > 0 { max_depth } else { depth };
                let r = engine.search_deepening(&board, SearchAlgo::Brs, tiefe_max, Some(&eval));
                if time_ms > 0 {
                    let ziel = if ist_a { &mut tiefe_a } else { &mut tiefe_b };
                    ziel[0] += r.depth as f64;
                    ziel[1] += 1.0;
                }
                match r.best_move {
                    Some(mv) => mv,
                    None     => moves[0],
                }
            }
            SearchKind::Maxn if time_ms > 0 => {
                let (mv, _, erreicht) = nnue_best_move_timed(
                    net, &board, &moves, time_ms, max_depth, beam, order, keys, &mut tt);
                let ziel = if ist_a { &mut tiefe_a } else { &mut tiefe_b };
                ziel[0] += erreicht as f64;
                ziel[1] += 1.0;
                mv
            }
            SearchKind::Maxn => {
                nnue_best_move(net, &board, &moves, depth, beam, order, keys, &mut tt)
            }
            SearchKind::Mcts => {
                // MCTS kennt kein Zeitbudget; `--iters` steuert den Aufwand.
                // Die Zeitspalte bleibt für diese Seite deshalb leer.
                let cfg = if ist_a { mcts_a } else { mcts_b };
                baum.search(&|b: &Board| net.forward(b), &modell, &board, cfg)
                    .map(|r| r.best)
                    .unwrap_or(moves[0])
            }
        };
        board = Rules::apply_with_effects(&board, mv);
        plies += 1;
    }

    (place_values(Rules::final_scores(&board)), plies, tiefe_a, tiefe_b)
}

/// Ergebnis einer Gruppe: gepaarte Mittel und, bei Zeitbudget, die erreichte
/// Suchtiefe je Seite als [Summe, Anzahl].
struct GroupResult {
    a: f64,
    b: f64,
    d: f64,
    plies: f64,
    d_a: [f64; 2],
    d_b: [f64; 2],
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
    if args.a_beam != args.b_beam {
        println!("Beam-Sortierung: A {:?}, B {:?}", args.a_beam, args.b_beam);
    }
    if args.a_search != args.b_search {
        println!("Suchverfahren: A {:?}, B {:?}", args.a_search, args.b_search);
        println!("Achtung: die Tiefen sind nicht direkt vergleichbar — BRS gibt \
                  zwei Halbzüge je Runde aus, Max^n vier.");
        if args.a_search == SearchKind::Mcts || args.b_search == SearchKind::Mcts {
            println!("MCTS A: {} Simulationen, c_puct {} | B: {} Simulationen, c_puct {}",
                     args.a_iters.unwrap_or(args.iters), args.a_c_puct.unwrap_or(args.c_puct),
                     args.b_iters.unwrap_or(args.iters), args.b_c_puct.unwrap_or(args.c_puct));
        }
    }
    println!("{}", "-".repeat(64));

    // Je Gruppe die gepaarte Differenz A−B über die sechs Sitzaufteilungen.
    let results: Vec<GroupResult> = groups
        .par_iter()
        .map(|&g| {
            let start = opening(args.opening_plies, args.seed.wrapping_mul(1_000_003) ^ g as u64);
            let (mut sa, mut sb, mut plies) = (0.0f64, 0.0f64, 0.0f64);
            let (mut d_a, mut d_b) = ([0.0f64; 2], [0.0f64; 2]);
            for split in SPLITS {
                let (vals, p, ta, tb) = play(&start, split, &net_a, &net_b,
                                     args.a_beam, args.b_beam,
                                     args.a_search, args.b_search, args.tt_mb,
                                     &MctsConfig { iterations: args.a_iters.unwrap_or(args.iters),
                                                   c_puct:     args.a_c_puct.unwrap_or(args.c_puct) },
                                     &MctsConfig { iterations: args.b_iters.unwrap_or(args.iters),
                                                   c_puct:     args.b_c_puct.unwrap_or(args.c_puct) },
                                     args.depth, args.beam, args.max_moves, &keys,
                                     args.time_ms, args.max_depth);
                d_a[0] += ta[0]; d_a[1] += ta[1];
                d_b[0] += tb[0]; d_b[1] += tb[1];
                for seat in 0..4 {
                    if split.contains(&seat) { sa += vals[seat] as f64; }
                    else                     { sb += vals[seat] as f64; }
                }
                plies += p as f64;
            }
            // 6 Partien × 2 Sitze = 12 Messwerte je Netz und Gruppe.
            GroupResult {
                a: sa / 12.0, b: sb / 12.0, d: (sa - sb) / 12.0, plies: plies / 6.0,
                d_a, d_b,
            }
        })
        .collect();

    let n = results.len() as f64;
    if n == 0.0 { println!("Keine Gruppen in diesem Shard."); return; }

    let mean_a: f64 = results.iter().map(|r| r.a).sum::<f64>() / n;
    let mean_b: f64 = results.iter().map(|r| r.b).sum::<f64>() / n;
    let diffs: Vec<f64> = results.iter().map(|r| r.d).collect();
    let mean_d: f64 = diffs.iter().sum::<f64>() / n;
    let var: f64 = if n > 1.0 {
        diffs.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / (n - 1.0)
    } else { 0.0 };
    let se = (var / n).sqrt();
    let plies: f64 = results.iter().map(|r| r.plies).sum::<f64>() / n;

    println!("Ø Platzwert A : {mean_a:+.4}");
    println!("Ø Platzwert B : {mean_b:+.4}");
    println!("Differenz A−B : {mean_d:+.4}  ± {se:.4} (SE)");
    if se > 0.0 { println!("t             : {:+.2}", mean_d / se); }
    println!("Ø Halbzüge    : {plies:.1}");
    if args.time_ms > 0 {
        // Bei gleichem Zeitbudget ist die erreichte Tiefe das, was die
        // Sortierung kostet oder einspart — ohne diese Spalte wäre der
        // Vergleich nicht zu deuten.
        let mittel = |f: fn(&GroupResult) -> [f64; 2]| {
            let (s, k): (f64, f64) = results.iter().map(f)
                .fold((0.0, 0.0), |(a, b), x| (a + x[0], b + x[1]));
            if k > 0.0 { s / k } else { 0.0 }
        };
        println!("Ø Tiefe A     : {:.2}", mittel(|r| r.d_a));
        println!("Ø Tiefe B     : {:.2}", mittel(|r| r.d_b));
    }

    if !args.out.is_empty() {
        // Summen statt Mittel: nur so lassen sich Shards korrekt zusammenlegen.
        let sum_d: f64 = diffs.iter().sum();
        let sum_d2: f64 = diffs.iter().map(|d| d * d).sum();
        let json = format!(
            "{{\"groups\":{},\"sum_a\":{},\"sum_b\":{},\"sum_d\":{},\"sum_d2\":{},\"sum_plies\":{}}}",
            results.len(),
            results.iter().map(|r| r.a).sum::<f64>(),
            results.iter().map(|r| r.b).sum::<f64>(),
            sum_d, sum_d2,
            results.iter().map(|r| r.plies).sum::<f64>(),
        );
        std::fs::write(&args.out, json).expect("Ergebnisdatei nicht schreibbar");
        println!("Geschrieben: {}", args.out);
    }
}
