//! Partien aus dem chess.com-Export mit der aktuellen Engine durchrechnen.
//!
//! Ersetzt den Weg über `analyze-game.js`: dort läuft die Engine als WASM, und
//! `crates/wasm/src/network.rs` bringt ein eigenes, festverdrahtetes Netz
//! (56 → 128 → 64 → 4) mit. Das NNUE dieser Kiste passt da nicht hinein — es
//! hat andere Eingaben und L1 = 256. Nativ fällt beides weg: echtes NNUE,
//! kein WASM-Aufschlag, und rayon statt eines Node-Prozesses je Partie.
//!
//! Das Ausgabeformat ist mit dem bisherigen identisch (Spieldaten + `depth` +
//! `evals`), damit die Weboberfläche und `analyze-all.py` unverändert
//! weiterarbeiten. Zusätzlich steht in jedem Eintrag das vollständige
//! Bewertungs-Quadrupel `eval`, aus dem sich `score` später neu berechnen
//! lässt, ohne die Suche noch einmal laufen zu lassen.
//!
//! # Zwei Maßzahlen, die nicht dasselbe messen
//!
//! `score` ist das Verhältnis der **Handbewertung** vor und nach dem Zug. Es
//! hat mit der Suche nichts zu tun und ist am Partieanfang unbrauchbar (siehe
//! [`calc_score`]). `loss` dagegen kommt aus der Suche: der Abstand zwischen
//! dem besten und dem gespielten Zug auf der Platzwertskala. Beide werden
//! geschrieben; eingestuft wird nach `loss`, wo es vorliegt.
//!
//! Das Nachrechnen allein ändert nichts — das Bewertungsnetz ist dasselbe.
//! Der Gewinn steckt in `--search mcts`: die Best-Reply-Suche, die die alten
//! Dateien erzeugt hat, liegt gemessen rund einen Platzwert unter der Engine,
//! die im Frontend spielt.
//!
//!   cargo run --release --bin analyze_games -- \
//!     --in ../game_data --out ../chaturaji-engine/crates/wasm/www/game_analysis \
//!     --search mcts --iters 25600 --weights weights.json [--jobs 4] [--force]
//!
//! `--calibrate` rechnet, schreibt nichts und gibt am Ende die Verteilung der
//! Verluste aus — damit `--loss-min` gesetzt statt geraten wird.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

use chaturaji_core::board::Board;
use chaturaji_core::notation::move_to_str;
use chaturaji_core::rules::Rules;
use chaturaji_engine::eval::{evaluate_with, EvalParams};
use chaturaji_engine::mcts::{Mcts, MctsConfig, MIN_VISITS_FOR_Q};
use chaturaji_engine::policy::PolicyNet;
use chaturaji_engine::search::Engine;
use chaturaji_nnue::dist::load_or_init_weights;
use chaturaji_nnue::network::NnueNetwork;
use chaturaji_nnue::pgn_import::parse_move_token;

/// Welche Suche das Urteil fällt.
///
/// `Brs` ist der bisherige Stand und bleibt erhalten, damit sich beide Urteile
/// über denselben Partien gegenüberstellen lassen — die Umstellung soll
/// nachweisbar sein, nicht geglaubt werden müssen.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchKind { Mcts, Brs }

struct Args {
    in_dir:  String,
    out_dir: String,
    depth:   u8,
    weights: Option<String>,
    jobs:    usize,
    limit:   usize,
    force:   bool,
    tt_mb:   usize,
    shard:   usize,
    shards:  usize,
    threshold: i32,
    search:  SearchKind,
    iters:   u32,
    /// Halbzüge am Partieanfang, die ungeprüft bleiben.
    ///
    /// Das sind Buchzüge. Sie als „Fehler" zu markieren, weil die Engine etwas
    /// anderes spielt, sagt nichts über den Spieler aus — und die Suche dort
    /// ist der teuerste Teil der Partie, weil das Brett noch voll ist.
    skip_book: usize,
    /// Verlustschwelle, ab der eine Anmerkung entsteht (Platzwert).
    loss_min: f32,
    /// Nur die Verteilung der Verluste ausgeben, nichts schreiben.
    calibrate: bool,
}

fn main() {
    let a = parse_args();

    let net: Option<NnueNetwork> = match &a.weights {
        Some(p) => match load_or_init_weights(p, 0.0, 0.9) {
            Ok(n)  => { eprintln!("Netz: {p} ({} Parameter, {} Schritte)", n.param_count(), n.steps); Some(n) }
            Err(e) => { eprintln!("Netz '{p}' nicht lesbar: {e}"); std::process::exit(1); }
        },
        None => { eprintln!("Kein Netz — es wird mit der Handbewertung gesucht."); None }
    };

    let out = PathBuf::from(&a.out_dir);
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("Zielverzeichnis '{}' nicht anlegbar: {e}", a.out_dir);
        std::process::exit(1);
    }

    let mut games: Vec<PathBuf> = match std::fs::read_dir(Path::new(&a.in_dir)) {
        Ok(rd) => rd.flatten().map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect(),
        Err(e) => { eprintln!("Verzeichnis '{}' nicht lesbar: {e}", a.in_dir); std::process::exit(1); }
    };
    games.sort();
    // Erst teilen, dann das Vorhandene aussortieren: so bearbeitet Shard i
    // immer dieselben Partien, egal wie viel davon schon fertig ist.
    if a.shards > 1 {
        games = games.into_iter().enumerate()
            .filter(|(i, _)| i % a.shards == a.shard)
            .map(|(_, p)| p).collect();
    }
    // Beim Kalibrieren wird nichts geschrieben — dann sind gerade die schon
    // analysierten Partien die interessanten, weil sich ihre alten und neuen
    // Anmerkungsraten vergleichen lassen.
    if !a.force && !a.calibrate {
        games.retain(|p| !out.join(p.file_name().unwrap()).exists());
    }
    if a.limit > 0 { games.truncate(a.limit); }

    if games.is_empty() {
        eprintln!("Nichts zu tun — alle Analysen liegen schon vor (--force überschreibt).");
        return;
    }

    if a.search == SearchKind::Mcts && net.is_none() {
        eprintln!("MCTS braucht ein Bewertungsnetz — --weights angeben oder --search brs wählen.");
        std::process::exit(1);
    }
    // Einmal geparst, von allen Threads geteilt: je Thread aus `include_str!`
    // neu aufzubauen kostet bei 6.367 Partien mehr als die Suche selbst.
    let prior = PolicyNet::default();

    rayon::ThreadPoolBuilder::new().num_threads(a.jobs).build_global().ok();
    if a.shards > 1 { eprintln!("Shard {}/{}", a.shard, a.shards); }
    match a.search {
        SearchKind::Mcts => eprintln!(
            "{} Partien, MCTS {} Simulationen, {} Threads, Anmerkungen ab {:.2} Platzwert{}",
            games.len(), a.iters, a.jobs, a.loss_min,
            if a.calibrate { " (Kalibrierlauf: es wird nichts geschrieben)" } else { "" }),
        SearchKind::Brs => eprintln!(
            "{} Partien, Best-Reply Tiefe {}, {} Threads, Anmerkungen ab {} %",
            games.len(), a.depth, a.jobs, a.threshold),
    }
    if a.skip_book > 0 { eprintln!("Die ersten {} Halbzüge bleiben ungeprüft (Buch).", a.skip_book); }

    let done   = AtomicUsize::new(0);
    let plies  = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let total  = games.len();
    let t0     = Instant::now();
    let alle   = std::sync::Mutex::new(Vec::<f32>::new());

    games.par_iter()
        .map_init(|| Sucher { engine: Engine::new(a.tt_mb), baum: Mcts::new(), hilfs: Mcts::new() },
        |s, path| {
            let res = analyze_file(s, &prior, path, &out, &a, net.as_ref());
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            match res {
                Ok((p, v)) => {
                    plies.fetch_add(p, Ordering::Relaxed);
                    if !v.is_empty() { alle.lock().unwrap().extend(v); }
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    eprintln!("\n{}: {e}", path.display());
                }
            }
            if n % 25 == 0 || n == total {
                let el   = t0.elapsed().as_secs_f64();
                let rest = el / n as f64 * (total - n) as f64;
                eprint!("\r{n}/{total}  {:.1} Halbzüge/s  Rest {:.0}h{:02.0}m   ",
                        plies.load(Ordering::Relaxed) as f64 / el,
                        rest / 3600.0, (rest % 3600.0) / 60.0);
            }
        })
        .for_each(|_| ());

    eprintln!("\nFertig in {:.1} min — {} Partien, {} Halbzüge, {} Fehler",
              t0.elapsed().as_secs_f64() / 60.0,
              done.load(Ordering::Relaxed), plies.load(Ordering::Relaxed),
              failed.load(Ordering::Relaxed));

    let v = alle.into_inner().unwrap();
    if !v.is_empty() {
        // Quantile lassen sich nicht zusammenlegen: aus den Quantilen zweier
        // Shards folgt das Quantil der Vereinigung nicht. Beim Kalibrieren
        // über mehrere Runner müssen deshalb die Rohwerte heraus.
        if a.calibrate {
            let datei = out.join(format!("losses-{:03}.txt", a.shard));
            let text: String = v.iter().map(|x| format!("{x:.5}\n")).collect();
            match std::fs::write(&datei, text) {
                Ok(())  => eprintln!("{} Verluste nach {} geschrieben.", v.len(), datei.display()),
                Err(e)  => eprintln!("Verluste nicht schreibbar ({}): {e}", datei.display()),
            }
        }
        verteilung(v, a.loss_min);
    }
}

/// Die Verteilung der Verluste, aus der die Schwellen kommen.
///
/// Ohne sie wären die Grenzen geraten — und ob geraten zu hoch oder zu tief
/// liegt, fiele erst nach dem vollen Lauf auf, also nach Stunden.
fn verteilung(mut v: Vec<f32>, loss_min: f32) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let q = |p: f64| v[((n - 1) as f64 * p).round() as usize];
    eprintln!("\nVerluste über {n} geprüfte Halbzüge (Platzwert):");
    for p in [0.50, 0.75, 0.90, 0.95, 0.99] {
        eprintln!("  {:>3.0} %  {:.3}", p * 100.0, q(p));
    }
    eprintln!("  Mittel {:.3}   Maximum {:.3}", v.iter().sum::<f32>() / n as f32, v[n - 1]);

    let anteil = |s: f32| v.iter().filter(|x| **x >= s).count() as f64 / n as f64 * 100.0;
    eprintln!("Bei --loss-min {loss_min:.2}: {:.1} % aller geprüften Halbzüge bekämen eine Anmerkung",
              anteil(loss_min));
    eprintln!("  davon Ungenauigkeit {:.1} %, Fehler {:.1} %, Blunder {:.1} %",
              anteil(loss_min) - anteil(loss_min * LOSS_FEHLER),
              anteil(loss_min * LOSS_FEHLER) - anteil(loss_min * LOSS_BLUNDER),
              anteil(loss_min * LOSS_BLUNDER));
}

// ─── Eine Partie ──────────────────────────────────────────────────────────────

/// Ein aufgezeichneter Halbzug. `None` steht für ausgelassene Züge (`--`) und
/// Zeitüberschreitungen (`T`) — die Engine rechnet dort nichts, der Platz im
/// Array bleibt aber erhalten, weil `analyze-all.py` die Zugtexte über den
/// Index zuordnet.
struct Ply {
    player:   usize,
    played:   String,
    best:     Option<String>,
    top:      Vec<(String, u8)>,
    material: [i32; 4],
    eval:     [i32; 4],
    /// `Q(bester Zug) − Q(gespielter Zug)` auf der Platzwertskala, ≥ 0.
    ///
    /// Anders als `score` stammt der Wert aus der Suche. `None` bei der
    /// Best-Reply-Suche, die keine Mittelwerte führt, und bei übersprungenen
    /// Buchzügen.
    loss:     Option<f32>,
    /// Der Zugtext, wie er im pgn4 steht (`Nf6-d7+`). Die Annotation zeigt ihn
    /// in der Notation der Oberfläche, nicht in der internen.
    cc:       String,
    /// Zugnummer aus dem pgn4 (`12.`), nicht die Runde: bei ausgeschiedenen
    /// Spielern laufen die beiden auseinander.
    full_nr:  u32,
}

#[allow(clippy::too_many_arguments)]
fn analyze_file(
    s:         &mut Sucher,
    prior:     &PolicyNet,
    path:      &Path,
    out:       &Path,
    a:         &Args,
    net:       Option<&NnueNetwork>,
) -> Result<(usize, Vec<f32>), String> {
    let depth     = a.depth;
    let threshold = a.threshold;
    let text  = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let game: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let pgn4  = game["pgn4"].as_str().ok_or("kein pgn4-Feld")?;

    let plies = analyze_game(s, prior, pgn4, depth, net, a.search, a.iters, a.skip_book)?;
    let n     = plies.iter().filter(|p| p.is_some()).count();
    let verluste: Vec<f32> = plies.iter().flatten().filter_map(|p| p.loss).collect();

    // Wie im bisherigen Werkzeug: verglichen wird die Bewertung des Ziehenden
    // vor seinem Zug mit derselben Komponente an der nächsten aufgezeichneten
    // Stelle.
    let scores: Vec<Option<i32>> = plies.iter().enumerate().map(|(i, p)| {
        p.as_ref().map(|e| match plies[i + 1..].iter().flatten().next() {
            Some(nx) => calc_score(e.eval[e.player], nx.eval[e.player]),
            None     => 100,
        })
    }).collect();

    let evals: Vec<Value> = plies.iter().zip(&scores).map(|(p, sc)| match p {
        None    => Value::Null,
        Some(e) => json!({
            "player":   e.player,
            "played":   e.played,
            "best":     e.best,
            "score":    sc.unwrap_or(100),
            "top":      e.top.iter().map(|(mv, pct)| json!({"mv": mv, "pct": pct}))
                             .collect::<Vec<_>>(),
            "material": e.material,
            "eval":     e.eval,
            // Nur bei MCTS vorhanden. `null` wäre gleichbedeutend, kostet aber
            // über 828.000 Halbzüge unnötig Platz.
            "loss":     e.loss,
        })
    }).collect();

    if a.calibrate { return Ok((n, verluste)); }

    let annotations = annotate(&game, &plies, &scores, threshold, a.loss_min);

    let mut doc = game.clone();
    if let Some(obj) = doc.as_object_mut() {
        obj.insert("depth".into(), json!(depth));
        obj.insert("evals".into(), Value::Array(evals));
        obj.insert("annotations".into(), Value::Array(annotations));
        // Frühere Läufe haben ihre Meldungen in den Chat geschrieben. Sie
        // gehören nicht zur Partie und würden sich mit jedem Lauf häufen.
        if let Some(chat) = game["chat"].as_array() {
            let sauber: Vec<Value> = chat.iter()
                .filter(|m| m["source"].as_str() != Some("analyze-report"))
                .cloned().collect();
            obj.insert("chat".into(), Value::Array(sauber));
        }
    }
    let dest = out.join(path.file_name().unwrap());
    std::fs::write(&dest, serde_json::to_string(&doc).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Ok((n, verluste))
}

/// Die Auffälligkeiten einer Partie, wie sie die Oberfläche anzeigt.
///
/// Portiert aus `analyze-all.py`, damit die Auswertung in einem Werkzeug
/// steckt: Zugtexte zweimal zu parsen — dort in Python, in der Analyse in
/// JavaScript — hatte die beiden Parser auseinanderlaufen lassen, und
/// Zeitüberschreitungen verschoben in 1.131 Partien die Zuordnung.
/// Abstände zur Ungenauigkeitsschwelle, in Vielfachen von `loss_min`.
///
/// Zum Einordnen: 0,051 Platzwert ist eine Verdopplung des Suchaufwands, und
/// der Median zwischen bestem und zweitbestem Zug liegt bei 0,039. Ein Fehler
/// wirft also mehr weg, als doppelte Rechenzeit einbringt; ein Blunder ein
/// Mehrfaches davon.
const LOSS_FEHLER:  f32 = 3.0;
const LOSS_BLUNDER: f32 = 8.0;

fn annotate(
    game:      &Value,
    plies:     &[Option<Ply>],
    scores:    &[Option<i32>],
    threshold: i32,
    loss_min:  f32,
) -> Vec<Value> {
    plies.iter().zip(scores).enumerate().filter_map(|(i, (p, sc))| {
        let e     = p.as_ref()?;
        let score = (*sc)?;
        let best  = e.best.as_deref()?;
        // Kein Befund, wenn die Engine denselben Zug gespielt hätte.
        if best == e.played { return None; }

        // Urteilt die Suche (MCTS), wird nach dem Verlust eingestuft; ohne sie
        // bleibt es beim Handbewertungs-Quotienten, damit `--search brs`
        // Zeichen für Zeichen das Alte liefert.
        let label = match e.loss {
            Some(l) => {
                if l < loss_min { return None; }
                if      l >= loss_min * LOSS_BLUNDER { "⚡ Blunder" }
                else if l >= loss_min * LOSS_FEHLER  { "⚠  Fehler" }
                else                                 { "?  Ungenauigkeit" }
            }
            None => {
                if score > threshold { return None; }
                match score {
                    s if s < 50 => "⚡ Blunder",
                    s if s < 75 => "⚠  Fehler",
                    s if s < 90 => "?  Ungenauigkeit",
                    _           => "· auffällig",  // nur erreichbar, wenn threshold ≥ 90
                }
            }
        };

        let seat  = e.player + 1;
        let round = i / 4 + 1;
        let symbol = label.split_whitespace().next().unwrap_or("·");
        // Die Maßzahl im Text ist die, nach der eingestuft wurde — sonst
        // stünde neben „Blunder" eine Prozentzahl, die ihn nicht begründet.
        let mass = match e.loss {
            Some(l) => format!("−{l:.2}"),
            None    => format!("{score}%"),
        };
        Some(json!({
            "moveIdx":    i,
            "fullMoveNr": e.full_nr,
            "player":     e.player,
            "playerId":   game[format!("uid{seat}")].as_i64().unwrap_or(0),
            "username":   game[format!("username{seat}")].as_str().unwrap_or(""),
            "message":    format!("{symbol} Zug {round}: {} → {}  ({mass})",
                                  cc_to_display(&e.cc), engine_to_display(best)),
            "score":      score,
            "loss":       e.loss,
            "source":     "analyze-report",
        }))
    }).collect()
}

/// `Nf6-d7+` → `Nc3-a4+`: Figurenbuchstabe und Suffix bleiben, die Felder
/// werden aus den chess.com-Koordinaten (Dateien d–k, Ränge 4–11) in die
/// interne Notation übersetzt. `=R` wird zu `=B`, weil ein Bauer im Chaturaji
/// zum Boot wird.
fn cc_to_display(cc: &str) -> String {
    let piece  = cc.chars().next().filter(|c| "KBNR".contains(*c)).map(|c| c.to_string()).unwrap_or_default();
    let suffix = if cc.ends_with('#') { "#" } else if cc.ends_with('+') { "+" } else { "" };
    let body   = cc.trim_start_matches(['K', 'B', 'N', 'R']).trim_end_matches(['+', '#']);
    let (sep, idx) = match body.find('x') {
        Some(i) => ('x', i),
        None    => match body.find('-') { Some(i) => ('-', i), None => return cc.to_string() },
    };
    let to_raw = body[idx + 1..].trim_start_matches(['K', 'B', 'N', 'R']);
    let promo  = if to_raw.contains("=R") { "=B" } else { "" };
    let to_cl  = to_raw.split('=').next().unwrap_or(to_raw);
    match (cc_square(&body[..idx]), cc_square(to_cl)) {
        (Some(a), Some(b)) => format!("{piece}{a}{sep}{b}{suffix}{promo}"),
        _                  => cc.to_string(),
    }
}

/// `e9` → `b6`; `None`, wenn das Feld außerhalb des 8×8-Bretts liegt.
fn cc_square(sq: &str) -> Option<String> {
    let mut ch = sq.chars();
    let file = ch.next()? as i32 - 'a' as i32 - 3;
    let rank: i32 = ch.as_str().parse().ok()?;
    let rank = rank - 3;
    if !(0..8).contains(&file) || !(1..=8).contains(&rank) { return None; }
    Some(format!("{}{}", (b'a' + file as u8) as char, rank))
}

/// `d2d3` → `d2-d3`, `g2h1p` → `g2-h1=B`.
fn engine_to_display(mv: &str) -> String {
    if mv.len() < 4 { return mv.to_string(); }
    let promo = if mv.len() == 5 && mv.ends_with('p') { "=B" } else { "" };
    format!("{}-{}{promo}", &mv[0..2], &mv[2..4])
}

/// Alles, was je Thread einmal angelegt und über alle Partien wiederverwendet
/// wird. `PolicyNet` steckt bewusst nicht darin — er ist unveränderlich und
/// wird von allen Threads geteilt, statt je Thread erneut aus JSON geparst zu
/// werden.
struct Sucher {
    engine: Engine,
    /// Der Baum, der die Partie entlangwandert.
    baum:   Mcts,
    /// Zweiter Baum für die Nachmessung schwach besuchter Züge. Er darf den
    /// ersten nicht anfassen: dessen Wurzel gehört zur laufenden Partie, und
    /// eine Suche darin verwürfe die geerbten Besuche.
    hilfs:  Mcts,
}

#[allow(clippy::too_many_arguments)]
fn analyze_game(
    s:         &mut Sucher,
    prior:     &PolicyNet,
    pgn4:      &str,
    depth:     u8,
    net:       Option<&NnueNetwork>,
    kind:      SearchKind,
    iters:     u32,
    skip_book: usize,
) -> Result<Vec<Option<Ply>>, String> {
    let f;
    let net_eval: Option<&dyn Fn(&Board) -> [f32; 4]> = match net {
        Some(n) => { f = |b: &Board| n.forward(b); Some(&f) }
        None    => None,
    };
    let cfg = MctsConfig { iterations: iters, ..MctsConfig::default() };
    s.baum.reset();
    let params = EvalParams::default();
    let mut board = Board::default();
    let mut out: Vec<Option<Ply>> = Vec::new();

    for (token, full_nr) in move_tokens(pgn4) {
        if token == "--" { out.push(None); continue; }
        // `T` Zeitüberschreitung, `R` Aufgabe: beides beendet die Teilnahme.
        // Der König bleibt stehen und wird oft noch geschlagen — deshalb nur
        // `active` löschen, nichts vom Brett nehmen.
        // Der Aufgabe entspricht kein Zug, `advance` kann ihr also nicht
        // folgen. Ohne Verwerfen gehörte der Baum ab hier zu einer Stellung,
        // die es nicht mehr gibt.
        if token == "T" || token == "R" {
            forfeit(&mut board); s.baum.reset(); out.push(None); continue;
        }

        let (from, to) = match parse_move_token(token) {
            Some(x) => x,
            None    => continue,          // Anmerkungen und Unbekanntes
        };
        let legal = Rules::legal_moves(&board);
        let mv = match legal.iter().find(|m| m.from == from && m.to == to) {
            Some(m) => *m,
            // Stellungsmismatch: hier weiterzurechnen hieße, ab jetzt eine
            // andere Partie zu analysieren als die aufgezeichnete.
            None => return Err(format!("Zug '{token}' ist in der Stellung nicht legal (Halbzug {})", out.len())),
        };

        let player   = board.to_move.idx();
        let material = board.scores.as_array();
        let eval     = evaluate_with(&board, &params);

        // Buchzüge und erzwungene Züge: nichts zu entscheiden, also auch nichts
        // zu suchen. Der Eintrag bleibt trotzdem stehen, weil die Oberfläche
        // die Zugtexte über den Index zuordnet.
        let ueberspringen = out.len() < skip_book || legal.len() < 2;

        let (best, top, loss) = if ueberspringen {
            (None, Vec::new(), None)
        } else {
            match kind {
                SearchKind::Brs => {
                    // Eine Suche statt zweier: die Rangliste liefert den besten Zug gleich mit.
                    let ranked = s.engine.top_n_brs(&board, depth, 3, net_eval);
                    let best   = ranked.first().map(|r| move_to_str(&r.mv));
                    let head   = ranked.first().map(|r| r.scores[player]).unwrap_or(1).max(1);
                    let top: Vec<(String, u8)> = ranked.iter().map(|r| {
                        let raw = r.scores[player];
                        let pct = ((raw.max(0) as f64 / head as f64) * 100.0).round().min(100.0) as u8;
                        (move_to_str(&r.mv), pct)
                    }).collect();
                    (best, top, None)
                }
                SearchKind::Mcts => {
                    let eval_fn = net_eval.ok_or("MCTS braucht ein Netz (--weights)")?;
                    let r = s.baum.search(eval_fn, prior, &board, &cfg)
                        .ok_or("MCTS fand keinen Zug in einer Stellung mit legalen Zügen")?;

                    let best = Some(move_to_str(&r.best));
                    // Anteil an den Besuchen: was die Suche tatsächlich
                    // erarbeitet hat. Der bisherige Quotient aus
                    // Utility-Werten hat hier kein Gegenstück.
                    let gesamt: u32 = r.visits.iter().map(|(_, n)| *n).sum::<u32>().max(1);
                    let top: Vec<(String, u8)> = r.visits.iter().take(3).map(|(m, n)| {
                        (move_to_str(m), ((*n as f64 / gesamt as f64) * 100.0).round() as u8)
                    }).collect();

                    let q_best = r.root_q.iter().find(|(m, _, _)| *m == r.best)
                        .map(|(_, q, _)| *q).unwrap_or(0.0);
                    let gespielt = r.root_q.iter().find(|(m, _, _)| *m == mv).copied();

                    let q_gespielt = match gespielt {
                        Some((_, q, n)) if n >= MIN_VISITS_FOR_Q => Some(q),
                        // Zu wenige Besuche: der Mittelwert ist Rauschen. Ein
                        // Blunder bekommt gerade deshalb wenige Besuche — ihn
                        // aus zwei Simulationen zu bewerten, hieße den Fall zu
                        // verfehlen, für den die Zahl da ist.
                        _ => {
                            let danach = Rules::apply_with_effects(&board, mv);
                            s.hilfs.reset();
                            s.hilfs.search(eval_fn, prior, &danach, &MctsConfig {
                                iterations: (iters / 8).max(64), ..cfg
                            }).map(|h| h.value[player])
                        }
                    };
                    // Der beste Zug kann nicht schlechter sein als ein anderer;
                    // wo die Nachmessung etwas anderes sagt, ist der Abstand
                    // Rauschen und kein Fund.
                    (best, top, q_gespielt.map(|q| (q_best - q).max(0.0)))
                }
            }
        };

        out.push(Some(Ply {
            player,
            played: move_to_str(&mv),
            best,
            top,
            material,
            eval,
            loss,
            cc: token.to_string(),
            full_nr,
        }));
        // Die Wurzel auf die Folgestellung setzen, bevor `board` sie ersetzt —
        // `advance` will die Stellung *vor* dem Zug. Kam der Zug im Baum nicht
        // vor (übersprungen, oder nie erweitert), verwirft `advance` selbst.
        s.baum.advance(&board, mv);
        board = Rules::apply_with_effects(&board, mv);
    }
    Ok(out)
}

/// Ausscheiden ohne Zug: `T` (Zeit überschritten) und `R` (aufgegeben).
fn forfeit(board: &mut Board) {
    let c = board.to_move;
    if !board.active[c.idx()] { return; }
    board.active[c.idx()] = false;
    let mut next = c.next();
    for _ in 0..4 {
        if board.active[next.idx()] { break; }
        next = next.next();
    }
    board.to_move = next;
}

/// Prozentzahl wie bisher: das Verhältnis der eigenen Bewertung vor und nach
/// dem Zug.
///
/// Die Kennzahl ist am Partieanfang unbrauchbar — dort liegt die Bewertung nahe
/// null, und das Verhältnis kippt: gemessen bekommt Runde 1 im Mittel 68 %
/// statt der 91 %, die danach üblich sind. Sie wird hier trotzdem unverändert
/// berechnet, damit alte und neue Analysen vergleichbar bleiben; wer sie
/// ersetzen will, kann das aus dem mitgeschriebenen Feld `eval` tun, ohne die
/// Suche zu wiederholen.
fn calc_score(before: i32, after: i32) -> i32 {
    if before <= 0 { return 100; }
    ((after as f64 / before as f64 * 100.0).round() as i32).clamp(0, 100)
}

/// Zugtokens aus dem pgn4-Text.
///
/// Mehrzeilige Header (`[StartFen4 "…"` erstreckt sich über 14 Brettzeilen)
/// werden explizit übersprungen — ein zeilenweiser `^\[.*\]$`-Filter lässt die
/// FEN-Zeilen als vermeintliche Züge stehen.
fn move_tokens(pgn4: &str) -> Vec<(&str, u32)> {
    let mut out: Vec<(&str, u32)> = Vec::new();
    let mut full_nr = 1u32;
    let mut in_header = false;
    for line in pgn4.lines() {
        let line = line.trim();
        if in_header {
            if line.ends_with(']') { in_header = false; }
            continue;
        }
        if line.starts_with('[') {
            if !line.ends_with(']') { in_header = true; }
            continue;
        }
        let mut depth = 0i32;      // { date=… clock=… } überspringen
        let mut start = 0usize;
        let bytes = line.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            match b {
                b'{' => { push_tokens(&line[start..i], &mut out, &mut full_nr); depth += 1; }
                b'}' => { depth -= 1; start = i + 1; }
                _    => {}
            }
            if depth == 0 && i + 1 == bytes.len() {
                push_tokens(&line[start..], &mut out, &mut full_nr);
            }
        }
    }
    out
}

fn push_tokens<'a>(chunk: &'a str, out: &mut Vec<(&'a str, u32)>, full_nr: &mut u32) {
    for t in chunk.split_whitespace() {
        if let Some(n) = t.strip_suffix('.') {
            if let Ok(v) = n.parse::<u32>() { *full_nr = v; }
            continue;
        }
        if t == ".." || t == "*" { continue; }
        out.push((t, *full_nr));
    }
}

// ─── Argumente ────────────────────────────────────────────────────────────────

fn parse_args() -> Args {
    let mut a = Args {
        in_dir:  "game_data".into(),
        out_dir: "game_analysis".into(),
        depth:   4,
        weights: None,
        jobs:    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        limit:   0,
        force:   false,
        tt_mb:   16,
        shard:   0,
        shards:  1,
        threshold: 89,
        search:  SearchKind::Mcts,
        iters:   25_600,
        skip_book: 16,
        loss_min: 0.05,
        calibrate: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let v = args.get(i + 1).cloned();
        match args[i].as_str() {
            "--in"      => { if let Some(v) = v { a.in_dir  = v; } i += 1; }
            "--out"     => { if let Some(v) = v { a.out_dir = v; } i += 1; }
            "--weights" => { a.weights = v; i += 1; }
            "--depth"   => { if let Some(v) = v { a.depth = v.parse().unwrap_or(a.depth); } i += 1; }
            "--jobs"    => { if let Some(v) = v { a.jobs  = v.parse().unwrap_or(a.jobs);  } i += 1; }
            "--limit"   => { if let Some(v) = v { a.limit = v.parse().unwrap_or(0);       } i += 1; }
            "--tt-mb"   => { if let Some(v) = v { a.tt_mb = v.parse().unwrap_or(a.tt_mb); } i += 1; }
            "--shard"   => { if let Some(v) = v { a.shard  = v.parse().unwrap_or(0); } i += 1; }
            "--shards"  => { if let Some(v) = v { a.shards = v.parse().unwrap_or(1).max(1); } i += 1; }
            "--threshold" => { if let Some(v) = v { a.threshold = v.parse().unwrap_or(89); } i += 1; }
            "--iters"   => { if let Some(v) = v { a.iters = v.parse().unwrap_or(a.iters); } i += 1; }
            "--skip-book" => { if let Some(v) = v { a.skip_book = v.parse().unwrap_or(a.skip_book); } i += 1; }
            "--loss-min"  => { if let Some(v) = v { a.loss_min  = v.parse().unwrap_or(a.loss_min);  } i += 1; }
            "--search"  => {
                match v.as_deref() {
                    Some("mcts") => a.search = SearchKind::Mcts,
                    Some("brs")  => a.search = SearchKind::Brs,
                    other => { eprintln!("--search kennt nur 'mcts' und 'brs', nicht {other:?}"); std::process::exit(1); }
                }
                i += 1;
            }
            "--calibrate" => a.calibrate = true,
            "--force"   => a.force = true,
            other       => eprintln!("unbekanntes Argument: {other}"),
        }
        i += 1;
    }
    a
}
