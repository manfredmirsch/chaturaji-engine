//! Verteiltes NNUE-Training: Partien erzeugen und Partien lernen getrennt.
//!
//! Warum getrennt?
//!
//! `td::run` ist strikt sequentiell — Partie spielen, Gewichte aktualisieren,
//! nächste Partie mit dem aktualisierten Netz. Das ist auf einer Maschine
//! richtig, lässt sich aber nicht auf viele Runner verteilen: der zweite Runner
//! bräuchte das Ergebnis des ersten.
//!
//! Hier wird daraus ein Rundenverfahren:
//!
//!   Runde r:  N Shards spielen parallel Partien gegen ein *eingefrorenes*
//!             Netz w_r und legen nur die Zugfolgen ab.
//!             Ein Lern-Schritt spielt alle Partien der Runde in fester
//!             Reihenfolge nach und wendet dieselben TD(λ)-Updates an wie
//!             `td::run` → w_{r+1}.
//!
//! Der Unterschied zum lokalen Lauf ist genau einer: die Züge einer Runde
//! stammen von einem Netz, das bis zu eine Runde alt ist. Innerhalb einer
//! Partie war das Netz auch bisher schon eingefroren (die Updates liefen erst
//! nach Partieende), die Staleness wächst also von „eine Partie" auf „eine
//! Runde". Bei Rundengrößen in der Größenordnung des bisherigen
//! `save_every` (200 Partien) ist das dieselbe Größenordnung wie das ohnehin
//! vorhandene Checkpoint-Intervall.
//!
//! Alles, was den Trainingsverlauf bestimmt — ε, Lernrate, RNG-Seed — hängt
//! allein an der globalen Partienummer und nicht am Prozessstart. Nur so ist
//! ein über viele Jobs verteilter Lauf mit einem durchgehenden lokalen Lauf
//! vergleichbar.

use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use chaturaji_core::board::Board;
use chaturaji_core::notation::parse_move;
use chaturaji_core::rules::Rules;
use chaturaji_core::zobrist::ZobristKeys;
use chaturaji_engine::book::OpeningBook;

use crate::features::INPUT_SIZE;
use crate::network::{Layer, NnueNetwork, Traces, H1, H2, OUTPUT};
use crate::selfplay::{final_targets, game_seed, play_batch, GameJob, SelfPlayConfig};

// ─── Zeitplan: ε und Lernrate aus der globalen Partienummer ───────────────────

/// ε nach `games_done` gespielten Partien.
///
/// Entspricht der Schleife in `td::run`, nur in geschlossener Form: dort wird ε
/// nach jeder Partie mit `epsilon_decay` multipliziert und bei `epsilon_end`
/// gekappt.
pub fn epsilon_at(cfg: &SelfPlayConfig, games_done: u64) -> f32 {
    let decayed = cfg.epsilon_start * cfg.epsilon_decay.powf(games_done as f32);
    decayed.max(cfg.epsilon_end)
}

/// Lernrate nach `games_done` Partien.
///
/// `td::run` senkt die Lernrate einmal je `save_every` Partien um `lr_decay`.
pub fn lr_at(lr0: f32, lr_decay: f32, save_every: u32, games_done: u64) -> f32 {
    let decays = games_done / save_every.max(1) as u64;
    (lr0 * lr_decay.powf(decays as f32)).max(1e-6)
}

// ─── Zustand zwischen den Runden ──────────────────────────────────────────────

/// Der Trainingsfortschritt, der zwischen Runden weitergereicht wird.
///
/// Bewusst klein und menschenlesbar: diese Datei ist im CI der einzige Ort, an
/// dem steht, wie weit der Lauf ist.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Progress {
    pub round:          u32,
    pub games_done:     u64,
    /// Seed des gesamten Laufs; die Partie-Seeds werden daraus abgeleitet.
    pub run_seed:       u64,
    pub last_avg_loss:  Option<f32>,
    pub last_avg_plies: Option<f32>,
}

impl Default for Progress {
    fn default() -> Self {
        Self { round: 0, games_done: 0, run_seed: 42, last_avg_loss: None, last_avg_plies: None }
    }
}

impl Progress {
    pub fn load(path: &str) -> Self {
        match fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()) {
            Some(p) => p,
            None    => Progress::default(),
        }
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)
    }
}

// ─── Gewichte als Datei ───────────────────────────────────────────────────────

/// Lädt Gewichte aus einer JSON-Datei; ohne Datei ein frisches Netz.
pub fn load_or_init_weights(path: &str, lr: f32, momentum: f32) -> io::Result<NnueNetwork> {
    if !Path::new(path).exists() {
        let mut net = NnueNetwork::new(lr, momentum);
        net.init_momentum();
        return Ok(net);
    }
    let text = fs::read_to_string(path)?;
    let mut net: NnueNetwork = serde_json::from_str(&text)?;
    net.init_momentum();   // enthält ensure_input_size
    Ok(net)
}

pub fn save_weights(path: &str, net: &NnueNetwork) -> io::Result<()> {
    fs::write(path, serde_json::to_string(net)?)
}

// ─── Adam-Momente ─────────────────────────────────────────────────────────────
//
// `Layer::mw/vw/mb/vb` sind `#[serde(skip)]`, stehen also nicht in der
// Gewichtsdatei. Lokal war das folgenlos, weil ein Lauf ein Prozess war. Über
// Runden hinweg ist es das nicht: mit m = v = 0 und großem `steps` ist die
// Bias-Korrektur praktisch 1, und der erste Schritt nach dem Neustart fällt um
// den Faktor ~3 zu groß aus (m/√v ≈ 0,1·g / 0,0316·|g|). Bei einem Lauf über
// viele Runden passiert das bei jeder Runde einmal.
//
// Deshalb ein eigenes, kompaktes Binärformat: rohe f32 in fester Reihenfolge,
// ~2,6 MB statt ~8 MB als JSON.

const OPT_MAGIC: &[u8; 8] = b"NNUEOPT1";

fn write_f32s(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32s(r: &mut impl Read, v: &mut [f32]) -> io::Result<()> {
    let mut buf = [0u8; 4];
    for x in v.iter_mut() {
        r.read_exact(&mut buf)?;
        *x = f32::from_le_bytes(buf);
    }
    Ok(())
}

fn layer_moments_mut(l: &mut Layer) -> (&mut Vec<Vec<f32>>, &mut Vec<f32>, &mut Vec<Vec<f32>>, &mut Vec<f32>) {
    (&mut l.mw, &mut l.mb, &mut l.vw, &mut l.vb)
}

pub fn save_opt_state(path: &str, net: &NnueNetwork) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(OPT_MAGIC)?;
    // Dimensionen mitschreiben, damit ein Format-/Architekturwechsel auffällt,
    // statt still falsche Zahlen einzulesen.
    for d in [INPUT_SIZE as u32, H1 as u32, H2 as u32, OUTPUT as u32] {
        w.write_all(&d.to_le_bytes())?;
    }
    for layer in [&net.l1, &net.l2, &net.l3] {
        for row in &layer.mw { write_f32s(&mut w, row)?; }
        write_f32s(&mut w, &layer.mb)?;
        for row in &layer.vw { write_f32s(&mut w, row)?; }
        write_f32s(&mut w, &layer.vb)?;
    }
    w.flush()
}

/// Lädt die Adam-Momente in ein bereits initialisiertes Netz.
///
/// Fehlt die Datei oder passen die Dimensionen nicht, bleibt es bei Nullen —
/// mit einer Warnung, denn das kostet Trainingsqualität und soll nicht
/// unbemerkt passieren.
pub fn load_opt_state(path: &str, net: &mut NnueNetwork) -> io::Result<bool> {
    if !Path::new(path).exists() {
        return Ok(false);
    }
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != OPT_MAGIC {
        eprintln!("Warnung: '{path}' hat kein bekanntes Optimizer-Format; Momente starten bei 0.");
        return Ok(false);
    }
    let mut dims = [0u32; 4];
    for d in dims.iter_mut() {
        let mut buf = [0u8; 4];
        r.read_exact(&mut buf)?;
        *d = u32::from_le_bytes(buf);
    }
    if dims != [INPUT_SIZE as u32, H1 as u32, H2 as u32, OUTPUT as u32] {
        eprintln!("Warnung: '{path}' passt nicht zur Architektur {dims:?}; Momente starten bei 0.");
        return Ok(false);
    }
    for layer in [&mut net.l1, &mut net.l2, &mut net.l3] {
        let (mw, mb, vw, vb) = layer_moments_mut(layer);
        for row in mw.iter_mut() { read_f32s(&mut r, row)?; }
        read_f32s(&mut r, mb)?;
        for row in vw.iter_mut() { read_f32s(&mut r, row)?; }
        read_f32s(&mut r, vb)?;
    }
    Ok(true)
}

// ─── Partien als Datei ────────────────────────────────────────────────────────

/// Eine erzeugte Partie, wie sie zwischen Shard und Lern-Schritt reist.
///
/// Nur die Zugfolge, nicht die Stellungen: der Lern-Schritt spielt sie ohnehin
/// nach. Das hält die Artefakte bei wenigen KB je Partie.
#[derive(Serialize, Deserialize)]
pub struct GameLine {
    pub index:  u64,
    pub moves:  String,
    pub plies:  usize,
    pub scores: [i32; 4],
    pub winner: Option<String>,
}

// ─── Erzeugen ─────────────────────────────────────────────────────────────────

pub struct GenerateConfig {
    pub weights_path: String,
    pub progress_path: String,
    pub out_path:     String,
    /// Anzahl Shards in dieser Runde und Nummer dieses Shards (0-basiert).
    pub shards:       u64,
    pub shard:        u64,
    /// Partien, die diese Runde insgesamt erzeugen soll (über alle Shards).
    pub games_total:  u64,
    pub selfplay:     SelfPlayConfig,
    pub book:         Option<OpeningBook>,
    /// Harte Obergrenze in Sekunden (0 = keine). Wird zwischen Blöcken geprüft;
    /// GitHub Actions bricht Jobs nach 6 h hart ab, und ein hart abgebrochener
    /// Job lädt kein Artefakt hoch — die Partien wären verloren.
    pub max_seconds:  u64,
}

pub struct GenerateStats {
    pub games:      u64,
    pub avg_plies:  f32,
    pub seconds:    f64,
    pub hit_budget: bool,
}

/// Spielt den Anteil dieses Shards an der Runde und schreibt ihn als JSONL.
pub fn generate(cfg: GenerateConfig) -> io::Result<GenerateStats> {
    let progress = Progress::load(&cfg.progress_path);
    let net      = load_or_init_weights(&cfg.weights_path, 0.001, 0.9)?;
    let keys     = ZobristKeys::new();
    let start    = Instant::now();

    // Partienummern dieses Shards: reihum, damit jeder Shard denselben
    // ε-Bereich sieht — ein zusammenhängender Block würde den letzten Shard
    // systematisch mit kleinerem ε spielen lassen.
    let jobs: Vec<GameJob> = (0..cfg.games_total)
        .filter(|i| i % cfg.shards.max(1) == cfg.shard)
        .map(|i| {
            let index = progress.games_done + i;
            GameJob {
                index,
                seed:    game_seed(progress.run_seed, index),
                epsilon: epsilon_at(&cfg.selfplay, index),
            }
        })
        .collect();

    println!(
        "Runde {} | Shard {}/{} | {} Partien | Tiefe {} Beam {} | ε {:.4}..{:.4}",
        progress.round, cfg.shard, cfg.shards, jobs.len(),
        cfg.selfplay.engine_depth, cfg.selfplay.beam_width,
        jobs.first().map(|j| j.epsilon).unwrap_or(0.0),
        jobs.last().map(|j| j.epsilon).unwrap_or(0.0),
    );

    let threads   = rayon::current_num_threads();
    // Blockweise, damit Zwischenstände auf die Platte gehen und das Zeitbudget
    // regelmäßig geprüft wird, statt erst am Ende.
    let chunk     = (threads * 4).max(1);
    let mut out   = BufWriter::new(File::create(&cfg.out_path)?);
    let mut done  = 0u64;
    let mut plies = 0u64;
    let mut hit_budget = false;

    for block in jobs.chunks(chunk) {
        if cfg.max_seconds > 0 && start.elapsed().as_secs() >= cfg.max_seconds {
            hit_budget = true;
            println!("Zeitbudget erreicht — {done} Partien werden gesichert.");
            break;
        }

        for (index, result) in play_batch(&net, &cfg.selfplay, block, cfg.book.as_ref(), &keys) {
            let fb = &result.final_board;
            let line = GameLine {
                index,
                moves:  result.move_log.join(" "),
                plies:  result.steps.len(),
                scores: fb.scores.as_array(),
                winner: result.winner.map(|c| c.name().to_string()),
            };
            writeln!(out, "{}", serde_json::to_string(&line)?)?;
            plies += result.steps.len() as u64;
            done  += 1;
        }

        let elapsed = start.elapsed().as_secs_f64().max(1e-6);
        println!(
            "  {done}/{} Partien | {:.1} Partien/min | {} Threads",
            jobs.len(), done as f64 / elapsed * 60.0, threads,
        );
    }

    out.flush()?;
    Ok(GenerateStats {
        games:     done,
        avg_plies: if done > 0 { plies as f32 / done as f32 } else { 0.0 },
        seconds:   start.elapsed().as_secs_f64(),
        hit_budget,
    })
}

// ─── Lernen ───────────────────────────────────────────────────────────────────

pub struct LearnConfig {
    pub weights_path:   String,
    pub weights_out:    String,
    pub opt_state_path: String,
    pub progress_path:  String,
    /// Verzeichnis mit den JSONL-Dateien aller Shards dieser Runde.
    pub games_dir:      String,
    pub lambda:         f32,
    pub lr0:            f32,
    pub lr_decay:       f32,
    pub save_every:     u32,
}

pub struct LearnStats {
    pub games:     u64,
    pub avg_loss:  f32,
    pub avg_plies: f32,
    pub steps:     u64,
    pub lr:        f32,
    pub skipped:   u64,
}

/// Spielt eine Zugfolge nach und gibt die Stellungen *vor* jedem Zug zurück,
/// dazu die Endstellung.
///
/// Das ist genau die Folge, die `selfplay::play_game` in `steps` sammelt: dort
/// wird vor jedem Zug die aktuelle Stellung abgelegt.
fn replay(moves: &str) -> Result<(Vec<Board>, Board), String> {
    let mut board  = Board::default();
    let mut boards = Vec::new();
    for tok in moves.split_whitespace() {
        let mv = parse_move(&board, tok)?;
        boards.push(board.clone());
        board = Rules::apply_with_effects(&board, mv);
    }
    Ok((boards, board))
}

/// Wendet die TD(λ)-Updates einer ganzen Runde an.
///
/// Die Update-Schleife je Partie ist Zeile für Zeile dieselbe wie in
/// `td::run`; abweichend ist nur, woher die Partie kommt.
pub fn learn(cfg: LearnConfig) -> io::Result<LearnStats> {
    let mut progress = Progress::load(&cfg.progress_path);
    let mut net      = load_or_init_weights(&cfg.weights_path, cfg.lr0, 0.9)?;

    match load_opt_state(&cfg.opt_state_path, &mut net) {
        Ok(true)  => println!("Adam-Momente übernommen."),
        Ok(false) => println!("Keine Adam-Momente gefunden — Start bei 0."),
        Err(e)    => eprintln!("Warnung: Optimizer-Zustand nicht lesbar ({e}); Start bei 0."),
    }

    // Alle Shard-Dateien einsammeln und global nach Partienummer sortieren:
    // die Reihenfolge der Updates muss von der Nummer abhängen, nicht davon,
    // welcher Runner zuerst fertig war.
    let mut files: Vec<PathBuf> = fs::read_dir(&cfg.games_dir)?
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
    println!("{} Partien aus {} Dateien geladen.", games.len(), files.len());

    let mut acc_loss  = 0.0f32;
    let mut acc_plies = 0u64;
    let mut learned   = 0u64;
    let mut skipped   = 0u64;

    for game in &games {
        net.lr = lr_at(cfg.lr0, cfg.lr_decay, cfg.save_every, game.index);

        let (boards, final_board) = match replay(&game.moves) {
            Ok(v)  => v,
            Err(e) => {
                eprintln!("Warnung: Partie {} nicht nachspielbar ({e}) — übersprungen.", game.index);
                skipped += 1;
                continue;
            }
        };

        // Integritätsprüfung: wenn Erzeuger und Lerner unterschiedliche
        // Regelstände hätten, liefe das Nachspielen still auseinander.
        if final_board.scores.as_array() != game.scores {
            eprintln!(
                "Warnung: Partie {} ergibt beim Nachspielen {:?} statt {:?} — übersprungen. \
                 Laufen Erzeuger und Lerner auf demselben Commit?",
                game.index, final_board.scores.as_array(), game.scores,
            );
            skipped += 1;
            continue;
        }

        // Werte des Netzes *vor* den Updates dieser Partie — wie im lokalen
        // Lauf, wo sie während des Spielens entstanden und die Updates erst
        // danach liefen.
        let values: Vec<[f32; 4]> = boards.iter().map(|b| net.forward(b)).collect();
        let final_target = final_targets(&final_board);
        let n_steps = boards.len();

        let mut traces    = Traces::new();
        let mut game_loss = 0.0f32;

        for t in (0..n_steps).rev() {
            let v_next: [f32; 4] = if t + 1 < n_steps { values[t + 1] } else { final_target };
            let td_error: [f32; 4] = std::array::from_fn(|i| v_next[i] - values[t][i]);
            game_loss += td_error.iter().map(|e| e * e).sum::<f32>() / 4.0;

            let cache = net.forward_full(&boards[t]);
            net.backward_into_traces(&cache, &mut traces, cfg.lambda);
            net.apply_td_update(&traces, &td_error);
        }

        acc_loss  += if n_steps > 0 { game_loss / n_steps as f32 } else { 0.0 };
        acc_plies += n_steps as u64;
        learned   += 1;
    }

    let avg_loss  = if learned > 0 { acc_loss / learned as f32 } else { 0.0 };
    let avg_plies = if learned > 0 { acc_plies as f32 / learned as f32 } else { 0.0 };

    // Fortschritt zählt die *gelernten* Partien: eine übersprungene Partie darf
    // den ε-Zeitplan nicht weiterdrehen.
    progress.round      += 1;
    progress.games_done += learned;
    progress.last_avg_loss  = Some(avg_loss);
    progress.last_avg_plies = Some(avg_plies);

    net.lr = lr_at(cfg.lr0, cfg.lr_decay, cfg.save_every, progress.games_done);

    save_weights(&cfg.weights_out, &net)?;
    save_opt_state(&cfg.opt_state_path, &net)?;
    progress.save(&cfg.progress_path)?;

    println!(
        "Runde {} gelernt | {} Partien | ∅Loss {:.5} | ∅Züge {:.1} | Schritte {} | lr {:.6}",
        progress.round, learned, avg_loss, avg_plies, net.steps, net.lr,
    );

    Ok(LearnStats {
        games:     learned,
        avg_loss,
        avg_plies,
        steps:     net.steps,
        lr:        net.lr,
        skipped,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use crate::selfplay::play_game;

    /// Der geschlossene ε-Ausdruck muss dieselbe Folge liefern wie die
    /// Schleife in `td::run` — sonst lernt ein verteilter Lauf mit einem
    /// anderen Explorationsplan als ein lokaler.
    #[test]
    fn epsilon_schedule_matches_loop() {
        let cfg = SelfPlayConfig::default();
        let mut eps = cfg.epsilon_start;
        for games_done in 0..500u64 {
            assert!(
                (epsilon_at(&cfg, games_done) - eps).abs() < 1e-4,
                "ε weicht bei Partie {games_done} ab: {} vs {eps}",
                epsilon_at(&cfg, games_done),
            );
            eps = (eps * cfg.epsilon_decay).max(cfg.epsilon_end);
        }
    }

    #[test]
    fn lr_schedule_matches_loop() {
        let (lr0, decay, save_every) = (0.001f32, 0.99f32, 200u32);
        let mut lr = lr0;
        for game in 1..=2000u64 {
            if game % save_every as u64 == 0 {
                lr = (lr * decay).max(1e-6);
            }
            let expected = lr;
            let got = lr_at(lr0, decay, save_every, game);
            assert!((got - expected).abs() < 1e-9, "lr bei Partie {game}: {got} vs {expected}");
        }
    }

    /// Das Nachspielen muss exakt die Stellungsfolge rekonstruieren, die das
    /// Self-Play gesehen hat. Bricht das, lernt der Lern-Schritt auf anderen
    /// Stellungen als denen, die gespielt wurden — und zwar lautlos.
    #[test]
    fn replay_reconstructs_selfplay_positions() {
        let net = NnueNetwork::new(0.001, 0.9);
        let cfg = SelfPlayConfig { max_moves: 40, ..Default::default() };
        let mut rng = SmallRng::seed_from_u64(7);
        let keys = ZobristKeys::new();
        let result = play_game(&net, &cfg, 1.0, &mut rng, None, &keys);

        let (boards, final_board) = replay(&result.move_log.join(" ")).expect("nachspielbar");
        assert_eq!(boards.len(), result.steps.len());
        for (i, (a, b)) in boards.iter().zip(&result.steps).enumerate() {
            assert_eq!(a.bb, b.board.bb, "Stellung {i} weicht ab");
            assert_eq!(a.to_move, b.board.to_move, "Zugrecht {i} weicht ab");
        }
        assert_eq!(final_board.scores.as_array(), result.final_board.scores.as_array());
    }

    /// Gleicher Seed, gleiche Partie — unabhängig davon, in welchem Shard sie
    /// gespielt wird.
    #[test]
    fn game_seed_is_position_independent() {
        let a = game_seed(42, 1234);
        let b = game_seed(42, 1234);
        assert_eq!(a, b);
        assert_ne!(game_seed(42, 1234), game_seed(42, 1235));
    }

    #[test]
    fn opt_state_roundtrip() {
        let dir = std::env::temp_dir().join("nnue_opt_roundtrip");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("opt.bin");
        let path = path.to_str().unwrap();

        let mut net = NnueNetwork::new(0.001, 0.9);
        net.init_momentum();
        net.l1.mw[3][17] = 0.25;
        net.l3.vb[2]     = 0.5;
        save_opt_state(path, &net).unwrap();

        let mut loaded = NnueNetwork::new(0.001, 0.9);
        loaded.init_momentum();
        assert!(load_opt_state(path, &mut loaded).unwrap());
        assert_eq!(loaded.l1.mw[3][17], 0.25);
        assert_eq!(loaded.l3.vb[2], 0.5);

        fs::remove_file(path).ok();
    }
}
