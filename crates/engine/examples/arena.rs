//! Selbstspiel-Turnier: zwei Konfigurationen gegeneinander.
//!
//! # Warum das nicht trivial ist
//!
//! Chaturaji ist kein symmetrisches Duell. Rot zieht zuerst, die vier Sitze
//! haben unterschiedliche Geometrie, und mit vier Spielern am Tisch entscheidet
//! auch mit, *neben wem* man sitzt. Ein naives „A spielt Rot, B spielt Blau"
//! misst überwiegend den Sitzvorteil.
//!
//! Deshalb:
//!   * Jede Partie besetzt zwei Sitze mit A und zwei mit B.
//!   * Es werden alle sechs Aufteilungen von vier Sitzen auf 2+2 durchlaufen,
//!     sodass jede Konfiguration über sechs Partien jeden Sitz genau dreimal
//!     einnimmt — Sitz- und Nachbarschaftseffekte heben sich weg.
//!   * Die ersten Halbzüge werden zufällig gespielt (gesetzter Seed), sonst
//!     spielen deterministische Engines immer dieselbe Partie.
//!
//! # Vergleichbarkeit der Suchtiefe
//!
//! `--rounds` zählt Spielrunden, nicht Halbzüge: Paranoid und Max^n brauchen
//! vier Plies pro Runde, BRS zwei. Nur so vergleicht man gleiche Vorausschau
//! statt gleicher Knotenzahl.
//!
//! Das beantwortet allerdings nur die halbe Frage. Gleiche Vorausschau ist
//! nicht gleich teuer — bei `--rounds 2` braucht Paranoid für eine Eröffnung
//! über eine Stunde, BRS eine Minute. Wer unter Turnierbedingungen stärker
//! spielt, entscheidet deshalb `--time-ms <budget>`: dann bekommt jeder Zug
//! dasselbe Zeitbudget, jede Konfiguration vertieft so weit sie kommt, und die
//! Spalte `Ø-Tiefe` zeigt, wie viel Vorausschau daraus jeweils wurde.
//!
//! # Ein Lauf ist kein Befund
//!
//! Das ausgewiesene SE misst die Streuung über die Eröffnungen *eines* Laufs.
//! Die Eröffnungen selbst hängen aber am Seed, weshalb zwischen zwei Seeds
//! noch einmal rund Faktor 1.5 mehr Streuung liegt. Jedes Ergebnis gehört
//! über mehrere `--seed` wiederholt; Vorzeichenwechsel sind der Normalfall,
//! wenn kein echter Effekt vorliegt.
//!
//! # Wertung
//!
//! Gewertet wird nach Platzierung, weil das die Zielgröße des Spiels ist. Wird
//! eine Partie durch das Ply-Limit abgeschnitten, ist der Punktestand trotzdem
//! eine gültige Rangfolge — anders als im Schach gibt es kein „unentschieden
//! weil unbeendet".
//!
//! Beispiel:
//! ```text
//! cargo run --release -p chaturaji-engine --example arena -- \
//!     --games 60 --rounds 2 --a new --b notransform
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use chaturaji_engine::eval::EvalParams;
use chaturaji_engine::search::SearchAlgo;
use chaturaji_engine::{Engine, SearchResult};

// ─── Konfigurationen ──────────────────────────────────────────────────────────

/// Das Verfahren kommt aus der Engine, damit Arena und Frontend dieselbe
/// Aufzählung und dieselbe Rundenschrittweite benutzen.
use SearchAlgo as Algo;

#[derive(Clone)]
struct Config {
    name:   String,
    algo:   Algo,
    params: EvalParams,
}

impl Config {
    /// Halbzüge, die eine Spielrunde Vorausschau kostet.
    fn step(&self) -> u8 { self.algo.step() }

    /// Suchtiefe in Halbzügen für `rounds` Spielrunden.
    fn depth(&self, rounds: u8) -> u8 { self.step() * rounds }
}

/// Wie einem Zug sein Rechenaufwand zugeteilt wird.
///
/// Die beiden Modi beantworten verschiedene Fragen, und nur zusammen ergeben
/// sie ein Bild: `Rounds` fragt „wer spielt bei gleicher Vorausschau besser",
/// `TimeMs` fragt „wer spielt bei gleicher Rechenzeit besser". Für die
/// Verfahren ist das der entscheidende Unterschied — BRS erreicht dieselbe
/// Vorausschau um Größenordnungen billiger, was bei fester Tiefe niemand sieht.
#[derive(Clone, Copy)]
enum Budget {
    Rounds(u8),
    TimeMs(u64),
}

/// Benannte Presets. Jedes isoliert genau eine Modellentscheidung, damit ein
/// Ergebnis auch zuordenbar ist.
fn preset(name: &str) -> Option<Config> {
    let d = EvalParams::default();
    let cfg = |algo, params| Config { name: name.to_string(), algo, params };

    Some(match name {
        // Der aktuelle Stand.
        "new"          => cfg(Algo::Brs, d),
        // Das Modell vor dem Umbau, inklusive alter Suche.
        "legacy"       => cfg(Algo::Paranoid, EvalParams::legacy()),
        // Alte Bewertung, neue Suche — isoliert die Bewertung.
        "legacyeval"   => cfg(Algo::Brs, EvalParams::legacy()),
        // Neue Bewertung, alte Suche — isoliert die Suche.
        "paranoid"     => cfg(Algo::Paranoid, d),
        "maxn"         => cfg(Algo::Maxn, d),

        // Einzelne Stellschrauben gegen den Standard.
        "notransform"  => cfg(Algo::Brs, EvalParams { rank_transform: false, ..d }),
        "elim10k"      => cfg(Algo::Brs, EvalParams { elimination_score: Some(-10_000), ..d }),
        "mat100"       => cfg(Algo::Brs, EvalParams { board_material: 100, ..d }),
        "mat20"        => cfg(Algo::Brs, EvalParams { board_material: 20, ..d }),
        "nogain"       => cfg(Algo::Brs, EvalParams { threat_gain: 0, ..d }),
        "gain120"      => cfg(Algo::Brs, EvalParams { threat_gain: 120, ..d }),
        "flattempo"    => cfg(Algo::Brs, EvalParams { graded_tempo: false, ..d }),
        "ks5000"       => cfg(Algo::Brs, EvalParams { ks_direct_imminent: 5_000, ..d }),
        _ => return None,
    })
}

const PRESETS: &[&str] = &[
    "new", "legacy", "legacyeval", "paranoid", "maxn", "notransform", "elim10k",
    "mat100", "mat20", "nogain", "gain120", "flattempo", "ks5000",
];

// ─── Zufall ───────────────────────────────────────────────────────────────────

/// xorshift64*, damit das Beispiel ohne Abhängigkeit auskommt.
struct Rng(u64);

impl Rng {
    /// Streut den Seed über splitmix64, bevor er zum Zustand wird. Ein blankes
    /// `seed | 1` hätte jeden geraden Seed auf den folgenden ungeraden
    /// abgebildet — Seed 22 und 23 wären dieselbe Partienserie gewesen.
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }
}

// ─── Eine Partie ──────────────────────────────────────────────────────────────

/// Spielt eine Partie. `seats[i]` ist der Konfigurationsindex für Farbe `i`.
/// Rückgabe: Endpunkte je Farbe und gespielte Halbzüge.
/// Spielt zufällige Eröffnungszüge und liefert die Ausgangsstellung.
/// Ohne Streuung spielen deterministische Engines jede Partie identisch.
fn random_opening(random_plies: u32, rng: &mut Rng) -> Board {
    let mut board = Board::default();
    for _ in 0..random_plies {
        if Rules::is_game_over(&board) { break; }
        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }
        board = Rules::apply_with_effects(&board, moves[rng.below(moves.len())]);
    }
    board
}

/// Sucht einen Zug auf feste Tiefe.
fn search_fixed(engine: &mut Engine, algo: Algo, board: &Board, depth: u8) -> SearchResult {
    match algo {
        Algo::Brs      => engine.search_brs(board, depth, None),
        Algo::Paranoid => engine.search_paranoid(board, depth, None),
        Algo::Maxn     => engine.search(board, depth),
    }
}

/// Sucht unter einem Zeitbudget.
///
/// Die eigentliche Arbeit macht die Engine: sie vertieft rundenweise und
/// bricht ab, sobald die hier gesetzte Bedingung greift. Anders als eine
/// Steuerung von außen kann sie das *mitten* in einer Iteration, das Budget
/// wird also nicht nur eingehalten, sondern auch ausgeschöpft.
///
/// Tiefe 40 ist kein erreichbares Ziel, sondern der Deckel, den
/// `search_deepening` braucht, um ohne Uhr terminieren zu können.
fn search_timed(
    engine:    &mut Engine,
    algo:      Algo,
    board:     &Board,
    budget_ms: u64,
) -> (SearchResult, u8) {
    let deadline = Instant::now() + Duration::from_millis(budget_ms);
    engine.set_stop_check(move || Instant::now() >= deadline);
    let result = engine.search_deepening(board, algo, 40, None);
    engine.clear_stop_check();

    let reached = result.depth;
    (result, reached)
}

/// Spielt eine Partie aus `opening`. `seats[i]` ist der Konfigurationsindex
/// für Farbe `i`. Rückgabe: Endpunkte je Farbe, gespielte Halbzüge sowie
/// Summe und Anzahl der erreichten Suchtiefen je Sitz.
fn play_game(
    configs:   &[Config; 2],
    seats:     [usize; 4],
    opening:   &Board,
    budget:    Budget,
    max_plies: u32,
) -> ([i32; 4], u32, [u64; 4], [u32; 4]) {
    let mut engines: Vec<Engine> = configs.iter()
        .map(|c| Engine::new(16).with_eval_params(c.params))
        .collect();
    for e in &mut engines { e.new_game(); }

    let mut board      = opening.clone();
    let mut ply        = 0u32;
    let mut depth_sum  = [0u64; 4];
    let mut depth_n    = [0u32; 4];

    while ply < max_plies && !Rules::is_game_over(&board) {
        if Rules::legal_moves(&board).is_empty() { break; }

        let seat   = board.to_move.idx();
        let ci     = seats[seat];
        let cfg    = &configs[ci];
        let engine = &mut engines[ci];

        let (result, reached) = match budget {
            Budget::Rounds(r) => {
                let d = cfg.depth(r);
                (search_fixed(engine, cfg.algo, &board, d), d)
            }
            Budget::TimeMs(ms) => search_timed(engine, cfg.algo, &board, ms),
        };
        depth_sum[seat] += reached as u64;
        depth_n[seat]   += 1;

        let mv = match result.best_move {
            Some(m) => m,
            None    => break,
        };

        board = Rules::apply_with_effects(&board, mv);
        ply += 1;
    }

    (board.scores.as_array(), ply, depth_sum, depth_n)
}

// ─── Auswertung ───────────────────────────────────────────────────────────────

/// Punkte → Platzierung (1-basiert), Punktgleiche teilen sich den Mittelwert.
fn placements(points: [i32; 4]) -> [f64; 4] {
    std::array::from_fn(|i| {
        let better = (0..4).filter(|&j| points[j] > points[i]).count();
        let tied   = (0..4).filter(|&j| points[j] == points[i]).count();
        // Belegt die Plätze better+1 .. better+tied.
        (better + 1) as f64 + (tied - 1) as f64 / 2.0
    })
}

#[derive(Default, Clone)]
struct Stats {
    seats:     u32,   // Anzahl besetzter Sitze über alle Partien
    place:     f64,   // Summe der Platzierungen
    points:    f64,   // Summe der Punkte
    wins:      u32,   // Anzahl erster Plätze (geteilte zählen anteilig nicht)
    depth_sum: u64,   // Summe der erreichten Suchtiefen über alle Züge
    depth_n:   u32,   // Anzahl gesuchter Züge
}

impl Stats {
    fn avg_place(&self)  -> f64 { self.place  / self.seats.max(1) as f64 }
    fn avg_points(&self) -> f64 { self.points / self.seats.max(1) as f64 }
    /// Ø erreichte Suchtiefe in Halbzügen. Bei fester Tiefe konstant, im
    /// Zeitmodus die eigentlich interessante Größe: sie zeigt, wie viel
    /// Vorausschau eine Konfiguration aus demselben Budget herausholt.
    fn avg_depth(&self)  -> f64 { self.depth_sum as f64 / self.depth_n.max(1) as f64 }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    let mut openings     = 10u32;
    let mut rounds       = 2u8;
    let mut time_ms      = 0u64;   // 0 = aus, dann zählt `rounds`
    let mut random_plies = 6u32;
    let mut max_plies    = 200u32;
    let mut seed         = 0xC0FF_EEu64;
    let mut a_name       = "new".to_string();
    let mut b_name       = "legacyeval".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let flag = args[i].clone();
        let mut val = || -> String { i += 1; args.get(i).cloned().unwrap_or_default() };
        match flag.as_str() {
            "--openings"     => openings     = val().parse().unwrap_or(openings),
            "--rounds"       => rounds       = val().parse().unwrap_or(rounds),
            "--time-ms"      => time_ms      = val().parse().unwrap_or(time_ms),
            "--random-plies" => random_plies = val().parse().unwrap_or(random_plies),
            "--max-plies"    => max_plies    = val().parse().unwrap_or(max_plies),
            "--seed"         => seed         = val().parse().unwrap_or(seed),
            "--a"            => a_name       = val(),
            "--b"            => b_name       = val(),
            "--list"         => { println!("Presets: {}", PRESETS.join(", ")); return; }
            other => { eprintln!("Unbekannte Option: {other}"); return; }
        }
        i += 1;
    }

    let (a, b) = match (preset(&a_name), preset(&b_name)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("Unbekanntes Preset. Verfügbar: {}", PRESETS.join(", "));
            return;
        }
    };
    let configs = [a.clone(), b.clone()];

    // Alle sechs 2+2-Aufteilungen der vier Sitze. Über je sechs Partien sitzt
    // jede Konfiguration dreimal auf jedem Sitz.
    const SPLITS: [[usize; 4]; 6] = [
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    ];

    let budget = if time_ms > 0 { Budget::TimeMs(time_ms) } else { Budget::Rounds(rounds) };

    let games = openings * SPLITS.len() as u32;
    println!("Arena: {} vs {}", a.name, b.name);
    println!("{openings} Eröffnungen × {} Sitzaufteilungen = {games} Partien", SPLITS.len());
    match budget {
        Budget::Rounds(r) => {
            println!("Vorausschau {r} Runden, Zufalls-Eröffnung {random_plies} Halbzüge, Seed {seed}");
            println!("Tiefen: {} = {}, {} = {}", a.name, a.depth(r), b.name, b.depth(r));
        }
        Budget::TimeMs(ms) => {
            println!("Zeitbudget {ms} ms/Zug, Zufalls-Eröffnung {random_plies} Halbzüge, Seed {seed}");
            println!("Tiefe wird erspielt — vertieft wird in Schritten von {} bzw. {} Halbzügen",
                     a.step(), b.step());
        }
    }
    println!("{}", "-".repeat(70));

    let mut rng   = Rng::new(seed);
    let mut stats: HashMap<usize, Stats> = HashMap::new();
    let mut total_plies = 0u64;
    // Differenz Ø-Platz (B − A) je Eröffnung, für die Streuung über Eröffnungen.
    let mut per_opening: Vec<f64> = Vec::with_capacity(openings as usize);
    let start = Instant::now();

    for o in 0..openings {
        // Eine Eröffnung, gespielt unter allen sechs Sitzaufteilungen. Dadurch
        // erlebt jede Konfiguration exakt dieselben Ausgangsstellungen in
        // spiegelbildlichen Rollen — Eröffnungsglück kürzt sich heraus, was
        // ohne diese Paarung die mit Abstand größte Rauschquelle war.
        let opening = random_opening(random_plies, &mut rng);
        let mut op_sum = [0.0f64; 2];
        let mut op_n   = [0u32; 2];

        for seats in SPLITS {
            let (points, plies, depth_sum, depth_n) =
                play_game(&configs, seats, &opening, budget, max_plies);
            total_plies += plies as u64;

            let place = placements(points);
            for seat in 0..4 {
                let ci = seats[seat];
                let s = stats.entry(ci).or_default();
                s.seats     += 1;
                s.place     += place[seat];
                s.points    += points[seat] as f64;
                s.depth_sum += depth_sum[seat];
                s.depth_n   += depth_n[seat];
                if place[seat] == 1.0 { s.wins += 1; }
                op_sum[ci] += place[seat];
                op_n[ci]   += 1;
            }
        }
        per_opening.push(op_sum[1] / op_n[1] as f64 - op_sum[0] / op_n[0] as f64);

        let sa = stats.get(&0).cloned().unwrap_or_default();
        let sb = stats.get(&1).cloned().unwrap_or_default();
        println!(
            "Eröffnung {:>3}/{openings} | {:<12} Ø-Platz {:.3} | {:<12} Ø-Platz {:.3}",
            o + 1, a.name, sa.avg_place(), b.name, sb.avg_place(),
        );
    }

    let sa = stats.get(&0).cloned().unwrap_or_default();
    let sb = stats.get(&1).cloned().unwrap_or_default();

    println!("{}", "-".repeat(66));
    println!("{:<14} {:>8} {:>11} {:>11} {:>8} {:>10}",
             "Konfig", "Sitze", "Ø-Platz", "Ø-Punkte", "Siege", "Ø-Tiefe");
    for (c, s) in [(&a, &sa), (&b, &sb)] {
        println!("{:<14} {:>8} {:>11.3} {:>11.2} {:>8} {:>10.2}",
                 c.name, s.seats, s.avg_place(), s.avg_points(), s.wins, s.avg_depth());
    }

    let delta = sb.avg_place() - sa.avg_place();

    // Streuung über Eröffnungen: die Eröffnungen sind voneinander unabhängig,
    // die sechs Partien innerhalb einer nicht. Deshalb ist die Eröffnung die
    // richtige Einheit für den Standardfehler — über Partien gerechnet käme
    // eine viel zu optimistische Zahl heraus.
    let n    = per_opening.len().max(1) as f64;
    let mean = per_opening.iter().sum::<f64>() / n;
    let var  = if n > 1.0 {
        per_opening.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else { 0.0 };
    let se = (var / n).sqrt();

    println!("{}", "-".repeat(70));
    println!("Ø-Platz-Differenz: {:+.3} ± {:.3} (SE) zugunsten von {}",
             delta.abs(), se, if delta > 0.0 { &a.name } else { &b.name });
    if se > 0.0 {
        // Schwelle 3 statt der lehrbuchüblichen 2: das SE oben misst nur die
        // Streuung *innerhalb* eines Laufs. Die Zufalls-Eröffnungen hängen aber
        // selbst am Seed, sodass zwischen zwei Seeds noch einmal deutlich mehr
        // Streuung liegt — gemessen rund Faktor 1.5. Bei Schwelle 2 hat diese
        // Arena bereits zweimal Scheinbefunde geliefert, die sich mit einem
        // zweiten Seed in Luft auflösten.
        let t = delta.abs() / se;
        println!("|t| = {t:.2}  →  {}", if t >= 3.0 {
            "über dem Rauschen"
        } else if t >= 2.0 {
            "grenzwertig — ohne zweiten Seed kein Befund"
        } else {
            "NICHT vom Rauschen unterscheidbar"
        });
    }
    println!("Partien à {:.0} Halbzüge, gesamt {:.1?}",
             total_plies as f64 / games.max(1) as f64, start.elapsed());
    println!();
    println!("Ø-Platz 2.5 = Gleichstand.");
    println!("Ein einzelner Lauf ist kein Befund: JEDES Ergebnis über mindestens");
    println!("zwei bis drei `--seed` wiederholen und auf gleiches Vorzeichen prüfen.");
    println!("Kalibrierung: NICHT `--a X --b X` — bei identischen Konfigurationen");
    println!("spielen die drei komplementären Sitzaufteilungen paarweise dieselbe");
    println!("Partie, die Differenz ist dann exakt 0 und misst kein Rauschen. Um");
    println!("das Rauschen zu prüfen, denselben Vergleich über mehrere `--seed`");
    println!("laufen lassen: die Deltas müssen sich innerhalb weniger SE decken.");
}
