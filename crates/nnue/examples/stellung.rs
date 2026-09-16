//! Eine einzelne Stellung aus einer Partie unter die Lupe nehmen.
//!
//! Gebaut, um einen gemeldeten Zweifelsfall nachzuprüfen: „die Baumsuche
//! schlägt hier etwas anderes vor als Best-Reply, und Best-Reply hat recht".
//! Ohne Werkzeug bleibt so ein Bericht eine Meinung gegen eine Messung.
//!
//! Gezeigt wird je Kandidatenzug:
//!
//! * was das Netz von der Folgestellung hält (aus Sicht des Ziehenden),
//! * was der Zug-Prior sagt (die Größe, nach der MCTS die Besuche verteilt),
//! * und das **Schiedsurteil**: eine tiefe Max^n-Suche mit breitem Beam, also
//!   deutlich mehr Rechnung als beide Verfahren im Frontend haben.
//!
//! # Vorsicht mit dem Schiedsurteil
//!
//! Es ist **nicht** stabil. Dieselbe Stellung (Partie 108644945, Halbzug 26),
//! zwei Einstellungen des Schiedsrichters:
//!
//! | Zug   | Tiefe 6 / Beam 8 | Tiefe 7 / Beam 12 |
//! |-------|------------------|-------------------|
//! | f3e5  | 0,343 (Platz 14) | 0,444 (Platz 5)   |
//! | f2e2  | 0,335 (Platz 15) | 0,440 (Platz 6)   |
//! | f3h2  | 0,455 (Platz 1)  | außerhalb der 10  |
//! | d1c2  | 0,434 (Platz 4)  | 0,400 (Platz 9)   |
//!
//! Die Rangfolge dreht sich fast vollständig. Der Beam ist der Grund: was
//! außerhalb der besten `b` Züge liegt, sieht die Suche nicht, und eine
//! Widerlegung, die bei Beam 8 herausfällt, taucht bei Beam 12 auf. Die
//! Abstände zwischen den Zügen (0,05 bis 0,1 Platzwert) liegen damit innerhalb
//! dessen, was die Methode selbst an Rauschen erzeugt.
//!
//! **Folgerung:** Dieses Werkzeug taugt, um eine Stellung zu verstehen — wer
//! greift was an, was sieht das Netz, wohin gehen die Besuche. Es taugt
//! **nicht**, um zwei Suchverfahren gegeneinander zu entscheiden. Dafür bleibt
//! die Arena zuständig, mit hunderten Partien und mehreren Seeds.
//!
//! ```text
//! cargo run --release -p chaturaji-nnue --example stellung -- \
//!     --game ~/chaturaji/game_data/108644945.json --ply 107 \
//!     --weights crates/wasm/www/weights.json --iters 25600
//! ```

use std::collections::HashMap;

use chaturaji_core::board::{Board, Move};
use chaturaji_core::notation::move_to_str;
use chaturaji_core::rules::Rules;
use chaturaji_engine::mcts::{Mcts, MctsConfig, RootChoice};
use chaturaji_engine::move_features::{fast_features, MoveFeatureContext, MoveModel};
use chaturaji_engine::nnue_network::NnueNetwork;
use chaturaji_engine::search::{Engine, SearchAlgo};
use chaturaji_nnue::pgn_import::parse_positions_from_pgn;
use chaturaji_nnue::selfplay::{nnue_maxn, BeamOrder};
use chaturaji_core::zobrist::ZobristKeys;

fn main() {
    let mut game    = String::new();
    let mut weights = "crates/wasm/www/weights.json".to_string();
    let mut ply     = 0usize;
    let mut iters   = 25_600u32;
    let mut brs_depth = 4u8;
    // Das Schiedsurteil. Tiefe 6 mit Beam 10 kostet ein Vielfaches von allem,
    // was im Frontend läuft — genau darum taugt es als Maßstab.
    let mut ref_depth = 6u8;
    let mut ref_beam  = 10usize;
    let mut ref_brs_depth = 8u8;
    // Zug(folge), die vor der Analyse noch ausgeführt wird. Damit lässt sich
    // fragen: was kommt nach dem Zug, den die Suche vorschlägt?
    let mut danach_zuege = String::new();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let wert = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--game"      => { game      = wert; i += 1; }
            "--weights"   => { weights   = wert; i += 1; }
            "--ply"       => { ply       = wert.parse().unwrap_or(0); i += 1; }
            "--iters"     => { iters     = wert.parse().unwrap_or(iters); i += 1; }
            "--brs-depth" => { brs_depth = wert.parse().unwrap_or(brs_depth); i += 1; }
            "--ref-depth" => { ref_depth = wert.parse().unwrap_or(ref_depth); i += 1; }
            "--ref-beam"  => { ref_beam  = wert.parse().unwrap_or(ref_beam); i += 1; }
            "--ref-brs"   => { ref_brs_depth = wert.parse().unwrap_or(ref_brs_depth); i += 1; }
            "--then"      => { danach_zuege = wert; i += 1; }
            other => { eprintln!("Unbekannte Option {other}"); std::process::exit(2); }
        }
        i += 1;
    }
    if game.is_empty() { eprintln!("--game fehlt"); std::process::exit(2); }

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&game).expect("Partie lesbar"))
            .expect("Partie ist JSON");
    let pgn4 = json["pgn4"].as_str().expect("kein pgn4 in der Partie");
    let stellungen = parse_positions_from_pgn(pgn4).expect("pgn4 nachspielbar");
    println!("Partie {}: {} Stellungen", json["gameNr"], stellungen.len());
    if ply >= stellungen.len() {
        eprintln!("--ply {ply} liegt hinter dem Partieende ({})", stellungen.len());
        std::process::exit(2);
    }

    let mut brett = stellungen[ply].clone();
    for zug in danach_zuege.split_whitespace() {
        let legal = Rules::legal_moves(&brett);
        let mv = chaturaji_core::notation::parse_move(&brett, zug)
            .ok()
            .and_then(|m| legal.iter().find(|l| l.from == m.from && l.to == m.to).copied())
            .unwrap_or_else(|| { eprintln!("Zug {zug} ist hier nicht legal"); std::process::exit(2); });
        brett = Rules::apply_with_effects(&brett, mv);
        println!("nach {zug}:");
    }
    let board = &brett;
    let sitz  = board.to_move.idx();
    println!("Halbzug {ply} — am Zug: {:?} (Sitz {sitz})", board.to_move);
    println!("Punkte: {:?} | aktiv: {:?}", board.scores.as_array(), board.active);
    println!("{}", brett_text(board));

    let mut netz: NnueNetwork =
        serde_json::from_str(&std::fs::read_to_string(&weights).expect("Netz lesbar"))
            .expect("Netz ist JSON");
    netz.ensure_input_size();
    println!("Netz: {} Schritte\n", netz.steps);

    let modell = MoveModel::default();
    let keys   = ZobristKeys::new();
    let zuege  = Rules::legal_moves(board);

    // ── Was die beiden Verfahren wählen ──────────────────────────────────────
    let r = Mcts::new().search(&|b: &Board| netz.forward(b), &modell, board,
                               &MctsConfig { iterations: iters, c_puct: chaturaji_engine::mcts::DEFAULT_C_PUCT,
                                             root: RootChoice::Visits,
                                             fpu: chaturaji_engine::mcts::Fpu::Null,
                                             reuse: false })
        .expect("Stellung hat Züge");
    let summe: u32 = r.visits.iter().map(|(_, n)| n).sum();
    println!("Baumsuche, {iters} Simulationen — Wurzelwert {:.3} für Sitz {sitz}", r.value[sitz]);
    for (mv, n) in r.visits.iter().take(5) {
        println!("   {:<8} {:>6} Besuche ({:>4.1} %)", move_to_str(mv), n,
                 100.0 * *n as f32 / summe.max(1) as f32);
    }

    let mut engine = Engine::new(64);
    let eval = |b: &Board| netz.forward(b);
    let brs = engine.search_deepening(board, SearchAlgo::Brs, brs_depth, Some(&eval));
    println!("\nBest-Reply Tiefe {brs_depth}: {}",
             brs.best_move.map(|m| move_to_str(&m)).unwrap_or_else(|| "—".into()));

    // ── Schiedsurteil ────────────────────────────────────────────────────────
    //
    // Jeder legale Zug wird von zwei unabhängigen, teuren Suchen beurteilt:
    // Max^n mit breitem Beam und Best-Reply mit doppelter Tiefe. Beide bekommen
    // deutlich mehr Rechnung als die Verfahren im Frontend. Stimmen sie
    // überein, ist die Rangfolge belastbar; widersprechen sie sich, sagt auch
    // das etwas — dann hängt das Urteil am Verfahren und nicht an der Stellung.
    let ctx = MoveFeatureContext::new(board);
    let prior_scores: Vec<(Move, f32)> = zuege.iter()
        .map(|mv| (*mv, modell.score_features(&fast_features(board, mv, &ctx))))
        .collect();
    let max_score = prior_scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
    let summe_exp: f32 = prior_scores.iter().map(|(_, s)| (s - max_score).exp()).sum();

    let besuche: HashMap<(u8, u8), u32> =
        r.visits.iter().map(|(m, n)| ((m.from, m.to), *n)).collect();

    let mut tt: HashMap<u64, (u8, [f32; 4])> = HashMap::new();
    let mut ref_engine = Engine::new(256);

    struct Zeile { mv: Move, netz: f32, prior: f32, besuche: u32, maxn: f32 }
    let mut zeilen: Vec<Zeile> = Vec::new();
    for mv in &zuege {
        let danach = Rules::apply_with_effects(board, *mv);
        let netzwert = netz.forward(&danach)[sitz];
        let prior = prior_scores.iter().find(|(m, _)| m == mv)
            .map(|(_, s)| (s - max_score).exp() / summe_exp).unwrap_or(0.0);

        // Max^n aus der Stellung *nach* dem Zug, also der Wert, den der Zug
        // einbringt — eine Tiefe weniger, weil der Zug selbst schon getan ist.
        tt.clear();
        let maxn = nnue_maxn(&netz, &danach, ref_depth.saturating_sub(1), ref_beam,
                             BeamOrder::Model, &mut tt, &keys)[sitz];

        zeilen.push(Zeile { mv: *mv, netz: netzwert, prior,
                            besuche: *besuche.get(&(mv.from, mv.to)).unwrap_or(&0),
                            maxn });
    }
    zeilen.sort_by(|a, b| b.maxn.partial_cmp(&a.maxn).unwrap_or(std::cmp::Ordering::Equal));

    println!("\nSchiedsurteil — Max^n Tiefe {ref_depth}, Beam {ref_beam}, alle {} Züge:", zuege.len());
    println!("{:<9} {:>8} {:>8} {:>9} {:>9}",
             "Zug", "Netz", "Prior", "Besuche", "Max^n");
    for z in zeilen.iter() {
        let woher = if Some(z.mv) == brs.best_move { "  ← Best-Reply" }
                    else if z.mv == r.best        { "  ← Baumsuche" }
                    else { "" };
        println!("{:<9} {:>8.3} {:>7.1} % {:>9} {:>9.3}{woher}",
                 move_to_str(&z.mv), z.netz, 100.0 * z.prior, z.besuche, z.maxn);
    }

    // Bleibt Best-Reply bei seinem Zug, wenn es mehr Tiefe bekommt? Wenn nicht,
    // war die Tiefe 4 zu flach und nicht das Verfahren im Recht.
    println!("\nBest-Reply bei wachsender Tiefe:");
    for d in 2..=ref_brs_depth {
        let t0 = std::time::Instant::now();
        let rr = ref_engine.search_deepening(board, SearchAlgo::Brs, d, Some(&eval));
        println!("   Tiefe {d}: {:<9} ({:>9} Knoten, {:>6} ms)",
                 rr.best_move.map(|m| move_to_str(&m)).unwrap_or_else(|| "—".into()),
                 rr.nodes, t0.elapsed().as_millis());
    }
}

/// Brett als Text, aus der Sicht von Rot unten.
fn brett_text(board: &Board) -> String {
    let mut s = String::new();
    for rank in (0..8).rev() {
        s.push_str(&format!("{}  ", rank + 1));
        for file in 0..8 {
            let sq = rank * 8 + file;
            match board.piece_at(sq) {
                Some(p) => {
                    let c = match p.kind {
                        chaturaji_core::piece::PieceKind::Pawn   => 'P',
                        chaturaji_core::piece::PieceKind::Knight => 'N',
                        chaturaji_core::piece::PieceKind::Bishop => 'B',
                        chaturaji_core::piece::PieceKind::Boat   => 'R',
                        chaturaji_core::piece::PieceKind::King   => 'K',
                    };
                    let f = p.color.name().chars().next().unwrap_or('?');
                    s.push(f.to_ascii_lowercase());
                    s.push(c);
                    s.push(' ');
                }
                None => s.push_str(" . "),
            }
        }
        s.push('\n');
    }
    s.push_str("    a  b  c  d  e  f  g  h\n");
    s
}
