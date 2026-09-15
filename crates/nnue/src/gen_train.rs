//! Generationentraining: überwachtes Lernen auf dem Partieausgang statt TD(λ).
//!
//! # Warum es das gibt
//!
//! Das TD-Training aus [`crate::dist::learn`] ist nach einer Runde stehen
//! geblieben. Gemessen gegen denselben vortrainierten Ausgangsstand:
//!
//! | | Seed 1 | Seed 42 |
//! |---|---|---|
//! | nach Runde 1 | +0,6510 | +0,6782 |
//! | nach Runde 4 | +0,5515 | +0,5694 |
//!
//! Die erste Runde bringt viel, die drei folgenden nichts mehr — eher etwas
//! weniger. Zwei Eigenschaften des TD-Ziels erklären das:
//!
//! 1. **Es ist an sich selbst gebunden.** Das Ziel für Stellung *t* ist
//!    V(s_{t+1}), also die eigene Meinung des Netzes über die nächste Stellung.
//!    Nur am Partieende steht ein echter Ausgang. Wenn das Netz sich irrt, lernt
//!    es seinen Irrtum.
//! 2. **Der echte Ausgang kommt kaum an.** 80 % der Selbstspielpartien laufen
//!    ins Halbzug-Limit (ausgezählt an einem Shard aus Runde 4: 319 von 400,
//!    Median genau 150). Bei λ = 0,7 ist das Gewicht des Endergebnisses 50
//!    Halbzüge vor Schluss auf 0,7⁵⁰ ≈ 2·10⁻⁸ gefallen. Die Eröffnung lernt
//!    praktisch nichts vom Ausgang.
//!
//! # Was hier anders ist
//!
//! Der Ansatz stammt aus `Foobork/flutteraji` (MIT, Patrick Davin) — ein
//! zweites Chaturaji-Projekt mit demselben Plateau-Problem. Dort läuft das
//! Training AlphaZero-artig in Generationen: eine Runde Selbstspiel, dann
//! **überwachtes** Lernen auf festen Zielen, dann ein Turnier gegen den
//! amtierenden Champion, und nur ein Sieger wird übernommen.
//!
//! Das Ziel je Stellung ist dort (`nnue/dataset.py`, `nnue/train.py`):
//!
//! ```text
//! ziel = (1 − q) · Partieausgang  +  q · Suchbewertung der Stellung
//! ```
//!
//! Beide Anteile sind *fest*, keiner hängt an der momentanen Meinung des
//! Netzes über eine Nachbarstellung:
//!
//! * Der **Partieausgang** ([`final_targets`]) ist geerdet und erreicht jede
//!   Stellung der Partie mit vollem Gewicht — kein λ-Zerfall.
//! * Die **Suchbewertung** ist der Scorevektor, den Max^n mit `engine_depth`
//!   Halbzügen Vorausschau an der Wurzel liefert. Sie ist genauer als das, was
//!   das Netz ohne Suche sagt, und trägt die Information, die das TD-Training
//!   bisher weggeworfen hat: Das Netz lernt, die Suche einzuholen. Steht sie
//!   nicht zur Verfügung (Buchzug oder ε-Zufallszug), zählt nur der Ausgang.
//!
//! Dazu zwei Dinge, die beim TD-Weg gar nicht möglich waren:
//!
//! * **Mischen über die ganze Runde.** TD muss eine Partie in Zugreihenfolge
//!   durchlaufen, weil die Traces daran hängen. Hier sind die Stellungen
//!   unabhängige Beispiele und werden vor jeder Epoche gemischt. Aufeinander
//!   folgende Stellungen einer Partie sind fast identisch; sie nacheinander zu
//!   lernen zieht den Optimierer in eine Richtung, die nur für diese eine
//!   Partie gilt.
//! * **Mehrere Epochen.** Dieselben Daten können mehrfach durchlaufen werden.
//!
//! # Was hier *nicht* anders ist
//!
//! Netz, Merkmale, Optimierer und Platzwertung bleiben, wie sie sind. Der
//! Unterschied liegt allein im Zielwert und in der Reihenfolge — damit ein
//! Arena-Vergleich zwischen beiden Wegen auch wirklich den Weg misst.
//!
//! Ob es besser ist, entscheidet die Arena, nicht dieser Kommentar.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use chaturaji_core::board::Board;

use crate::dist::{
    load_opt_state, load_or_init_weights, replay, save_opt_state, save_weights, GameLine, Progress,
};
use crate::selfplay::final_targets;

pub struct GenTrainConfig {
    pub weights_path:   String,
    pub weights_out:    String,
    pub opt_state_path: String,
    pub progress_path:  String,
    /// Verzeichnis mit den JSONL-Dateien aller Shards dieser Runde.
    pub games_dir:      String,
    pub lr:             f32,
    pub epochs:         u32,
    /// Gewicht der Suchbewertung gegenüber dem Partieausgang, 0…1.
    ///
    /// `Ziel = (1 − q) · Partieausgang + q · Suchbewertung`.
    ///
    /// Stand seit Einführung auf 0,10 und war bis 2026-09-15 nie geprüft. Dann
    /// nachgemessen: vier `gen`-Runden mit 0,30 gegen vier mit 0,10, gleiches
    /// Startnetz, gleicher `run_seed`, sonst identisch. In der Arena über vier
    /// Seeds +0,0174 / −0,0017 / −0,0185 / +0,0122, im Mittel **+0,002** —
    /// kein Befund, und damit ein weiterer Regler, der den Fixpunkt des
    /// Selbstspiels nicht bewegt.
    ///
    /// Der Trainingsverlust sinkt dabei deutlich (0,181 statt 0,270). Das ist
    /// erwartbar und aussagelos: mit höherem `q` besteht das Ziel zu einem
    /// größeren Teil aus dem, was das Netz ohnehin denkt.
    pub q_weight:       f32,
}

pub struct GenTrainStats {
    pub games:      u64,
    pub positions:  usize,
    pub with_q:     usize,
    pub first_loss: f32,
    pub avg_loss:   f32,
    pub steps:      u64,
    pub skipped:    u64,
    pub seconds:    f64,
}

/// Ein Trainingsbeispiel: Stellung und fester Zielvektor.
struct Sample {
    board:  Board,
    target: [f32; 4],
}

/// Baut die Zielwerte einer Partie.
///
/// `search_values` darf kürzer sein als `boards` (ältere Läufe ohne das Feld,
/// oder ein abgeschnittener Datensatz) — fehlende Einträge zählen als „keine
/// Suchbewertung vorhanden".
fn targets_for_game(
    boards:        &[Board],
    result:        [f32; 4],
    search_values: &[Option<[f32; 4]>],
    q_weight:      f32,
) -> (Vec<[f32; 4]>, usize) {
    let q = q_weight.clamp(0.0, 1.0);
    let mut with_q = 0;
    let targets = (0..boards.len())
        .map(|t| match search_values.get(t).copied().flatten() {
            Some(sv) if q > 0.0 => {
                with_q += 1;
                std::array::from_fn(|i| (1.0 - q) * result[i] + q * sv[i])
            }
            _ => result,
        })
        .collect();
    (targets, with_q)
}

/// Liest alle JSONL-Dateien einer Runde, global nach Partienummer sortiert.
fn load_round(games_dir: &str) -> io::Result<(Vec<GameLine>, usize)> {
    let mut files: Vec<PathBuf> = fs::read_dir(games_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "jsonl").unwrap_or(false))
        .collect();
    files.sort();

    let mut games: Vec<GameLine> = Vec::new();
    for path in &files {
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() { continue; }
            match serde_json::from_str::<GameLine>(&line) {
                Ok(g)  => games.push(g),
                Err(e) => eprintln!("Warnung: unlesbare Zeile in {}: {e}", path.display()),
            }
        }
    }
    games.sort_by_key(|g| g.index);
    Ok((games, files.len()))
}

/// Eine Runde Generationentraining.
pub fn learn_generational(cfg: GenTrainConfig) -> io::Result<GenTrainStats> {
    let start        = Instant::now();
    let mut progress = Progress::load(&cfg.progress_path);
    let mut net      = load_or_init_weights(&cfg.weights_path, cfg.lr, 0.9)?;

    match load_opt_state(&cfg.opt_state_path, &mut net) {
        Ok(true)  => println!("Adam-Momente übernommen."),
        Ok(false) => println!("Keine Adam-Momente gefunden — Start bei 0."),
        Err(e)    => eprintln!("Warnung: Optimizer-Zustand nicht lesbar ({e}); Start bei 0."),
    }
    // Feste Lernrate über die Runde. Der Zerfall aus dem TD-Weg hängt an der
    // Partienummer und würde bei mehreren Epochen dasselbe Beispiel mit
    // verschiedenen Schrittweiten lernen, je nachdem, wann es drankommt.
    net.lr = cfg.lr;

    let (games, n_files) = load_round(&cfg.games_dir)?;
    println!("{} Partien aus {n_files} Dateien geladen.", games.len());

    // ─── Beispiele bauen ────────────────────────────────────────────────────
    //
    // Alle Stellungen einer Runde liegen gleichzeitig im Speicher — anders ist
    // das Mischen nicht zu haben. Ein `Board` ist rund 200 Byte, eine Runde mit
    // 6.400 Partien à ~140 Halbzügen kommt damit auf gut 200 MB. Das passt auf
    // einen GitHub-Runner (7 GB), ist aber der Grund, warum hier nicht noch
    // mehr Runden auf einmal verarbeitet werden.
    let mut samples: Vec<Sample> = Vec::new();
    let mut skipped = 0u64;
    let mut learned = 0u64;
    let mut with_q  = 0usize;
    let mut plies   = 0u64;

    for game in &games {
        let (boards, final_board) = match replay(&game.moves) {
            Ok(v)  => v,
            Err(e) => {
                eprintln!("Warnung: Partie {} nicht nachspielbar ({e}) — übersprungen.", game.index);
                skipped += 1;
                continue;
            }
        };

        // Dieselbe Integritätsprüfung wie im TD-Weg: laufen Erzeuger und Lerner
        // auf verschiedenen Regelständen, liefe das Nachspielen still auseinander.
        if final_board.scores.as_array() != game.scores {
            eprintln!(
                "Warnung: Partie {} ergibt beim Nachspielen {:?} statt {:?} — übersprungen. \
                 Laufen Erzeuger und Lerner auf demselben Commit?",
                game.index, final_board.scores.as_array(), game.scores,
            );
            skipped += 1;
            continue;
        }

        let result = final_targets(&final_board);
        let (targets, q_here) =
            targets_for_game(&boards, result, &game.search_values, cfg.q_weight);

        with_q += q_here;
        plies  += boards.len() as u64;
        learned += 1;
        samples.extend(
            boards.into_iter().zip(targets).map(|(board, target)| Sample { board, target }),
        );
    }

    if samples.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "keine verwertbare Stellung in dieser Runde",
        ));
    }

    let q_share = with_q as f32 / samples.len() as f32 * 100.0;
    println!(
        "{} Stellungen aus {learned} Partien | {:.0} % mit Suchbewertung | q = {:.2}",
        samples.len(), q_share, cfg.q_weight,
    );

    // ─── Epochen ────────────────────────────────────────────────────────────
    //
    // Der Seed hängt an Lauf-Seed, Runde und Epoche: derselbe Lauf mischt
    // wieder gleich, zwei Epochen aber verschieden.
    let epochs = cfg.epochs.max(1);
    let mut first_loss = 0.0f32;
    let mut last_loss  = 0.0f32;

    for epoch in 1..=epochs {
        let mut rng = SmallRng::seed_from_u64(
            progress.run_seed ^ ((progress.round as u64) << 32) ^ epoch as u64,
        );
        samples.shuffle(&mut rng);

        let mut sum = 0.0f64;
        for s in &samples {
            let cache = net.forward_full(&s.board);
            let error: [f32; 4] = std::array::from_fn(|i| s.target[i] - cache.a3[i]);
            sum += net.apply_supervised_gradient(&cache, &error) as f64;
        }
        let loss = (sum / samples.len() as f64) as f32;

        if epoch == 1 { first_loss = loss; }
        last_loss = loss;
        println!(
            "  Epoche {epoch}/{epochs} | ∅Loss {loss:.5} | Schritte {} | {:.1} s",
            net.steps, start.elapsed().as_secs_f64(),
        );
    }

    // ─── Fortschritt und Speichern ──────────────────────────────────────────
    let avg_plies = if learned > 0 { plies as f32 / learned as f32 } else { 0.0 };
    progress.round      += 1;
    progress.games_done += learned;
    progress.last_avg_loss  = Some(last_loss);
    progress.last_avg_plies = Some(avg_plies);

    save_weights(&cfg.weights_out, &net)?;
    save_opt_state(&cfg.opt_state_path, &net)?;
    progress.save(&cfg.progress_path)?;

    println!(
        "Runde {} generationsweise gelernt | {learned} Partien | ∅Loss {:.5} → {:.5} | \
         ∅Züge {avg_plies:.1} | Schritte {} | lr {:.6}",
        progress.round, first_loss, last_loss, net.steps, net.lr,
    );

    Ok(GenTrainStats {
        games:      learned,
        positions:  samples.len(),
        with_q,
        first_loss,
        avg_loss:   last_loss,
        steps:      net.steps,
        skipped,
        seconds:    start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::NnueNetwork;

    fn boards(n: usize) -> Vec<Board> {
        (0..n).map(|_| Board::default()).collect()
    }

    #[test]
    fn without_search_values_the_target_is_the_plain_result() {
        let result = [1.0, 1.0 / 3.0, -1.0 / 3.0, -1.0];
        let (t, with_q) = targets_for_game(&boards(3), result, &[], 0.5);
        assert_eq!(with_q, 0, "ohne Suchwerte darf nichts gemischt werden");
        for row in t {
            assert_eq!(row, result);
        }
    }

    #[test]
    fn the_blend_uses_exactly_q_weight() {
        let result = [1.0, 0.0, 0.0, -1.0];
        let sv     = [0.0, 1.0, -1.0, 0.0];
        let (t, with_q) = targets_for_game(&boards(1), result, &[Some(sv)], 0.25);
        assert_eq!(with_q, 1);
        for i in 0..4 {
            let want = 0.75 * result[i] + 0.25 * sv[i];
            assert!((t[0][i] - want).abs() < 1e-6, "Komponente {i}: {} statt {want}", t[0][i]);
        }
    }

    /// q = 0 muss den Suchwert vollständig ignorieren — sonst wäre der
    /// Vergleich „nur Ausgang" gegen „Ausgang + Suche" nicht sauber.
    #[test]
    fn q_zero_ignores_the_search_value() {
        let result = [1.0, 0.0, 0.0, -1.0];
        let (t, with_q) = targets_for_game(&boards(1), result, &[Some([9.0; 4])], 0.0);
        assert_eq!(with_q, 0);
        assert_eq!(t[0], result);
    }

    /// Eine Partie, bei der nur manche Halbzüge gesucht wurden (Buch- und
    /// ε-Züge dazwischen), muss stellungsweise entscheiden.
    #[test]
    fn mixed_games_blend_only_where_a_search_ran() {
        let result = [1.0, 0.0, 0.0, -1.0];
        let sv     = [0.0, 0.0, 0.0, 0.0];
        let (t, with_q) = targets_for_game(
            &boards(3), result, &[None, Some(sv), None], 1.0,
        );
        assert_eq!(with_q, 1);
        assert_eq!(t[0], result, "Buchzug: nur der Ausgang");
        assert_eq!(t[1], sv,     "gesucht und q = 1: nur der Suchwert");
        assert_eq!(t[2], result, "ε-Zug: nur der Ausgang");
    }

    /// Mehr `search_values` als Stellungen darf nicht paniken, weniger auch
    /// nicht — beides kann bei alten oder abgeschnittenen Dateien vorkommen.
    #[test]
    fn length_mismatches_are_tolerated() {
        let result = [1.0, 0.0, 0.0, -1.0];
        let (t, _) = targets_for_game(&boards(2), result, &[Some([0.0; 4])], 0.5);
        assert_eq!(t.len(), 2);
        let (t, _) = targets_for_game(&boards(1), result, &[Some([0.0; 4]); 5], 0.5);
        assert_eq!(t.len(), 1);
    }

    /// Der eigentliche Zweck: wiederholtes überwachtes Lernen auf einem festen
    /// Ziel muss den Fehler verkleinern. Ohne das wäre der ganze Weg nutzlos.
    #[test]
    fn supervised_steps_move_the_net_towards_the_target() {
        let mut net = NnueNetwork::new(0.001, 0.9);
        net.init_momentum();
        let board  = Board::default();
        let target = [1.0f32, 1.0 / 3.0, -1.0 / 3.0, -1.0];

        let err = |n: &NnueNetwork| {
            let p = n.forward(&board);
            (0..4).map(|i| (p[i] - target[i]).powi(2)).sum::<f32>() / 4.0
        };

        let before = err(&net);
        for _ in 0..20 {
            let cache = net.forward_full(&board);
            let e: [f32; 4] = std::array::from_fn(|i| target[i] - cache.a3[i]);
            net.apply_supervised_gradient(&cache, &e);
        }
        assert!(err(&net) < before, "Loss muss sinken: {before:.5} → {:.5}", err(&net));
    }
}
