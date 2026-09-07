//! Prüft die Regelimplementierung gegen echte chess.com-Partien.
//!
//! Jede Partie aus `game_data/` wird nachgespielt und der errechnete Endstand
//! mit `points1..4` aus der Aufzeichnung verglichen. Das ist die schärfste
//! Probe, die es für die Regeln gibt: Schlagwerte, Doppel- und
//! Dreifachschach-Bonus, Aufgabe und der Zuschlag für stehengebliebene Könige
//! müssen alle stimmen, sonst geht die Summe nicht auf.
//!
//! Bricht das Nachspielen ab, sagt das Werkzeug warum: Wem gehört das Feld,
//! von dem gezogen werden soll, und ist dieser Spieler am Zug? Genau so wurde
//! gefunden, dass ein Viertel aller Partien verworfen wurde — `R` und `T`
//! belegen einen Zugslot und verschoben das Zugrecht um einen Spieler.
//!
//! Aufruf:
//!   cargo run --release -p chaturaji-nnue --example diagnose_pgn -- <verz> [anzahl]

use std::collections::BTreeMap;

use chaturaji_core::board::{bit, Board};
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::notation::square_name;
use chaturaji_core::rules::Rules;
use chaturaji_nnue::pgn_import::parse_move_token;

/// Bitmaske: welche gegnerischen Könige greift `mover` an?
fn attacked_king_set(board: &Board, mover: Color) -> u8 {
    let attacked = Rules::attacked_squares(board, mover);
    let mut set = 0u8;
    for c in Color::ALL {
        if c != mover && board.active[c.idx()]
           && board.bb[c.idx()][PieceKind::King.idx()] & attacked != 0 {
            set |= 1 << c.idx();
        }
    }
    set
}

/// Wer steht auf dem Feld?
fn piece_at(board: &Board, sq: u8) -> Option<(Color, PieceKind)> {
    for c in Color::ALL {
        for k in PieceKind::ALL {
            if board.bb[c.idx()][k.idx()] & bit(sq) != 0 {
                return Some((c, k));
            }
        }
    }
    None
}

/// chess.com-Feldname aus internem Index.
fn ext_name(sq: u8) -> String {
    format!("{}{}", (b'd' + (sq & 7)) as char, (sq >> 3) + 4)
}

struct Failure {
    file:     String,
    ply:      usize,
    token:    String,
    category: String,
    detail:   String,
}

/// Zerlegt eine einzelne Partie Zug für Zug und zeigt jeden Punktezuwachs mit
/// seiner Ursache. Damit lässt sich eine Abweichung gegen die Aufzeichnung bei
/// chess.com Zeile für Zeile vergleichen.
fn detail(path: &str) {
    let text = std::fs::read_to_string(path).expect("Datei nicht lesbar");
    let json: serde_json::Value = serde_json::from_str(&text).expect("kein JSON");
    let pgn4 = json["pgn4"].as_str().expect("kein pgn4");

    let mut move_text = String::new();
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
        if !line.is_empty() { move_text.push(' '); move_text.push_str(line); }
    }

    let tokens: Vec<&str> = move_text
        .split_whitespace()
        .filter(|t| *t != ".." && *t != "*")
        .collect();

    let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
    println!("Partie {name}");
    println!("https://www.chess.com/variants/chaturaji/game/{name}\n");
    println!("{:>4} {:<7} {:<12} {:<11} {:<24} {:>7}  {}",
             "Zug", "Spieler", "pgn4", "8x8-Brett", "Ereignis", "Punkte", "Stand");
    println!("{}", "─".repeat(104));

    let mut board  = Board::default();
    let mut movenum = String::from("?");

    for token in &tokens {
        // Zugnummer aus der Aufzeichnung uebernehmen - genau die zeigt
        // chess.com an.
        if token.ends_with('.') {
            movenum = token.trim_end_matches('.').to_string();
            continue;
        }
        if Rules::is_game_over(&board) {
            println!("{:>5}  {:<12} {:<26}", "", "", "Partie zu Ende (≤1 aktiv)");
            break;
        }

        if *token == "R" || *token == "T" {
            let who = board.to_move;
            board = Rules::resign(&board);
            println!("{:>4} {:<7} {:<12} {:<11} {:<24} {:>7}  {:?}",
                     movenum, who.name(), token, "",
                     "scheidet aus", "", board.scores.as_array());
            continue;
        }

        let (from_sq, to_sq) = match parse_move_token(token) {
            Some(sq) => sq,
            None     => continue,
        };
        let legal = Rules::legal_moves(&board);
        let mv = match legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
            Some(m) => *m,
            None    => { println!("  ABBRUCH bei {token:?}"); return; }
        };

        let mover  = board.to_move;
        let before = board.scores.get(mover);
        let checks_before = Rules::count_attacked_kings(&board, mover);
        let next   = Rules::apply_with_effects(&board, mv);
        let gained = next.scores.get(mover) - before;

        if gained != 0 {
            let checks_after = Rules::count_attacked_kings(&next, mover);
            let capture = mv.captured
                .map(|c| format!("schlägt {} {:?} ({})",
                                 c.color.name(), c.kind, c.kind.capture_value()))
                .unwrap_or_default();
            let bonus = match checks_after {
                2 => " +1 Doppelschach",
                3 => " +5 Dreifachschach",
                _ => "",
            };
            println!("{:>4} {:<7} {:<12} {:<11} {:<24} {:>+7}  {:?}   [{} → {} Könige im Schach]",
                     movenum, mover.name(), token,
                     format!("{}-{}", square_name(from_sq), square_name(to_sq)),
                     format!("{}{}", capture, bonus),
                     gained, next.scores.as_array(),
                     checks_before, checks_after);
        }

        board = next;
    }

    let replayed = Rules::final_scores(&board);
    let recorded = [
        json["points1"].as_i64().unwrap_or(0) as i32,
        json["points2"].as_i64().unwrap_or(0) as i32,
        json["points3"].as_i64().unwrap_or(0) as i32,
        json["points4"].as_i64().unwrap_or(0) as i32,
    ];
    println!("{}", "─".repeat(88));
    println!("Stand am Brettende : {:?}", board.scores.as_array());
    println!("mit Endstands-Regel: {replayed:?}");
    println!("chess.com          : {recorded:?}");
    let diff: Vec<i32> = (0..4).map(|i| recorded[i] - replayed[i]).collect();
    println!("Differenz          : {diff:?}   (chess.com minus nachgespielt)");
    println!("noch aktiv         : {:?}", Color::ALL.into_iter()
             .filter(|c| board.active[c.idx()]).map(|c| c.name()).collect::<Vec<_>>());
}

/// Sucht Partien, in denen ein Spieler zweimal nacheinander im Doppelschach
/// steht: erst erzeugt sein Zug es (vorher < 2, nachher 2), danach bleibt es
/// bei einem weiteren seiner Züge bestehen (vorher 2, nachher 2).
fn find_double(dir: &str, want: usize) {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("Verzeichnis nicht lesbar")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    paths.sort();

    let mut found = 0usize;

    for path in &paths {
        if found >= want { return; }
        let text = match std::fs::read_to_string(path) { Ok(t) => t, Err(_) => continue };
        let json: serde_json::Value = match serde_json::from_str(&text) { Ok(v) => v, Err(_) => continue };
        let pgn4 = match json["pgn4"].as_str() { Some(s) => s, None => continue };

        let mut move_text = String::new();
        let mut in_header = false;
        for line in pgn4.lines() {
            let line = line.trim();
            if in_header { if line.ends_with(']') { in_header = false; } continue; }
            if line.starts_with('[') { if !line.ends_with(']') { in_header = true; } continue; }
            if !line.is_empty() { move_text.push(' '); move_text.push_str(line); }
        }
        let tokens: Vec<&str> = move_text.split_whitespace()
            .filter(|t| !t.ends_with('.') && *t != ".." && *t != "*").collect();

        let mut board = Board::default();
        // je Spieler: das zuletzt erzeugte Doppelschach (Halbzug, Token)
        let mut created: [Option<(usize, String)>; 4] = [None, None, None, None];
        let mut hit: Option<(Color, (usize, String), (usize, String))> = None;

        for (ply, token) in tokens.iter().enumerate() {
            if Rules::is_game_over(&board) { break; }
            if *token == "R" || *token == "T" { board = Rules::resign(&board); continue; }
            let (from_sq, to_sq) = match parse_move_token(token) { Some(v) => v, None => continue };
            let legal = Rules::legal_moves(&board);
            let mv = match legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
                Some(m) => *m, None => break,
            };
            let mover  = board.to_move;
            let before = Rules::count_attacked_kings(&board, mover);
            let next   = Rules::apply_with_effects(&board, mv);
            let after  = Rules::count_attacked_kings(&next, mover);

            if after == 2 && before < 2 {
                created[mover.idx()] = Some((ply, (*token).to_string()));
            } else if after == 2 && before == 2 {
                if let Some(first) = created[mover.idx()].clone() {
                    hit = Some((mover, first, (ply, (*token).to_string())));
                    break;
                }
            } else if after < 2 {
                created[mover.idx()] = None;
            }
            board = next;
        }

        if let Some((mover, first, second)) = hit {
            let name = path.file_stem().unwrap().to_string_lossy();
            println!("Partie {name}   https://www.chess.com/variants/chaturaji/game/{name}");
            println!("  Spieler: {}", mover.name());
            println!("    Halbzug {:>4}  {:<12} erzeugt das Doppelschach (vorher <2, nachher 2)",
                     first.0, first.1);
            println!("    Halbzug {:>4}  {:<12} Doppelschach bestand bereits (vorher 2, nachher 2)",
                     second.0, second.1);
            println!("  Notation: erstes {:?}, zweites {:?}\n",
                     first.1.matches('+').count(), second.1.matches('+').count());
            found += 1;
        }
    }
    if found == 0 { println!("keine solche Partie gefunden"); }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir   = args.get(1).cloned().unwrap_or_else(|| "game_data".to_string());
    let limit = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    // Einzelne Datei → Detailansicht statt Massenauswertung.
    if dir.ends_with(".json") {
        detail(&dir);
        return;
    }

    // Suchlauf: zwei Doppelschachs desselben Spielers nacheinander, wobei das
    // zweite schon vorher bestand.
    if args.iter().any(|a| a == "--find-double") {
        find_double(&dir, args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5));
        return;
    }

    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("Verzeichnis nicht lesbar")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    paths.sort();
    paths.truncate(limit);

    let mut failures: Vec<Failure> = Vec::new();
    let mut ok             = 0usize;
    let mut score_match    = 0usize;
    let mut score_mismatch = 0usize;
    let mut quit_games     = 0usize;
    // Fehlbetrag je Spieler: bleibt hier etwas übrig, fehlt noch eine Regel.
    let mut diffs: BTreeMap<i32, usize> = BTreeMap::new();
    let mut variant_hits = [0usize; 4];
    // (Zahl der '+' in der Notation, von uns gezählte Könige im Schach)
    let mut check_matrix: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut fresh_matrix: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut plus2_pattern: BTreeMap<usize, usize> = BTreeMap::new();
    let mut other_pattern: BTreeMap<usize, usize> = BTreeMap::new();

    for path in &paths {
        let text = match std::fs::read_to_string(path) { Ok(t) => t, Err(_) => continue };
        let json: serde_json::Value = match serde_json::from_str(&text) { Ok(v) => v, Err(_) => continue };
        let pgn4 = match json["pgn4"].as_str() { Some(s) => s, None => continue };

        // Kopf überspringen — identisch zu `parse_positions_from_pgn`.
        let mut move_text = String::new();
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
            if !line.is_empty() { move_text.push(' '); move_text.push_str(line); }
        }

        let tokens: Vec<&str> = move_text
            .split_whitespace()
            .filter(|t| !t.ends_with('.') && *t != ".." && *t != "*")
            .collect();

        let mut board  = Board::default();
        let mut failed = false;
        let mut quits  = 0usize;

        for (ply, token) in tokens.iter().enumerate() {
            if Rules::is_game_over(&board) { break; }

            if *token == "R" || *token == "T" {
                quits += 1;
                board = Rules::resign(&board);
                continue;
            }

            let (from_sq, to_sq) = match parse_move_token(token) {
                Some(sq) => sq,
                None     => continue,
            };

            let legal = Rules::legal_moves(&board);
            if let Some(mv) = legal.iter().find(|m| m.from == from_sq && m.to == to_sq) {
                let mover = board.to_move;
                let next  = Rules::apply_with_effects(&board, *mv);
                // Die Notation sagt selbst, wie viele Könige der Zug ins Schach
                // setzt: ein `+` je König. Das ist der schärfste Maßstab für
                // unsere Erkennung, den es gibt.
                let noted = token.chars().filter(|&c| c == '+').count();
                let ours  = Rules::count_attacked_kings(&next, mover);
                // Neu ins Schach gesetzt: nach dem Zug angegriffen, vorher nicht.
                let before_set = attacked_king_set(&board, mover);
                let after_set  = attacked_king_set(&next,  mover);
                let fresh = (after_set & !before_set).count_ones() as usize;
                *check_matrix.entry((noted, ours)).or_default() += 1;
                *fresh_matrix.entry((noted, fresh)).or_default() += 1;
                board = next;
                continue;
            }

            // Bruch — Stellung befragen.
            let mover  = board.to_move;
            let source = piece_at(&board, from_sq);
            let target = piece_at(&board, to_sq);

            let (category, detail) = match source {
                None => (
                    "Startfeld leer".to_string(),
                    format!("{} ist leer; am Zug wäre {}", ext_name(from_sq), mover.name()),
                ),
                Some((owner, kind)) if owner != mover => (
                    "falscher Spieler am Zug".to_string(),
                    format!("auf {} steht {} {:?}, am Zug ist aber {}",
                            ext_name(from_sq), owner.name(), kind, mover.name()),
                ),
                Some((_, kind)) => (
                    format!("Zug nicht erzeugt ({kind:?})"),
                    format!("{} {:?} {} → {} fehlt; Ziel: {}",
                            mover.name(), kind, ext_name(from_sq), ext_name(to_sq),
                            match target {
                                Some((c, k)) => format!("{} {:?}", c.name(), k),
                                None => "leer".to_string(),
                            }),
                ),
            };

            failures.push(Failure {
                file: path.file_name().unwrap().to_string_lossy().to_string(),
                ply, token: (*token).to_string(), category, detail,
            });
            failed = true;
            break;
        }

        if failed { continue; }
        ok += 1;
        if quits > 0 { quit_games += 1; }

        // Vier Varianten des Endstands in einem Durchlauf vergleichen.
        let base = board.scores.as_array();
        let survivors: Vec<Color> = Color::ALL.into_iter()
            .filter(|c| board.active[c.idx()]).collect();
        let standing_kings = Color::ALL.iter()
            .filter(|&&c| !board.active[c.idx()])
            .filter(|&&c| board.pieces(c, PieceKind::King) != 0)
            .count() as i32;
        let variant = |per_king: i32, per_survivor: i32| -> [i32; 4] {
            let mut sc = base;
            if survivors.len() == 1 {
                sc[survivors[0].idx()] += standing_kings * per_king;
            } else {
                for c in &survivors { sc[c.idx()] += per_survivor; }
            }
            sc
        };
        let cands = [variant(3, 0), variant(1, 0), variant(3, 2), variant(1, 2)];

        let replayed = Rules::final_scores(&board);
        let recorded: Option<[i32; 4]> = (|| Some([
            json["points1"].as_i64()? as i32,
            json["points2"].as_i64()? as i32,
            json["points3"].as_i64()? as i32,
            json["points4"].as_i64()? as i32,
        ]))();
        if let Some(rec) = recorded {
            for (i, sc) in cands.iter().enumerate() {
                if *sc == rec { variant_hits[i] += 1; }
            }
            if rec == replayed {
                score_match += 1;
            } else {
                score_mismatch += 1;
                for i in 0..4 { *diffs.entry(rec[i] - replayed[i]).or_default() += 1; }
                // Muster prüfen: bekommt jeder Überlebende +2 und sonst keiner
                // etwas? Genau so sah das erste Beispiel aus.
                let survivors = Color::ALL.into_iter().filter(|c| board.active[c.idx()]).count();
                let two_each = Color::ALL.into_iter().all(|c| {
                    let d = rec[c.idx()] - replayed[c.idx()];
                    if board.active[c.idx()] { d == 2 } else { d == 0 }
                });
                if two_each { *plus2_pattern.entry(survivors).or_default() += 1; }
                else        { *other_pattern.entry(survivors).or_default() += 1; }
                if score_mismatch <= 8 {
                    println!("  Abweichung: {} | nachgespielt {:?} | chess.com {:?} | Aufgaben: {}",
                             path.file_name().unwrap().to_string_lossy(), replayed, rec, quits);
                }
            }
        }
    }

    let pct = |n: usize, of: usize| if of == 0 { 0.0 } else { n as f64 * 100.0 / of as f64 };

    println!("\nDateien             : {}", paths.len());
    println!("nachspielbar        : {} ({:.1} %)", ok, pct(ok, paths.len()));
    println!("davon mit Aufgabe   : {}", quit_games);
    println!("Endstand exakt      : {} ({:.1} %)", score_match, pct(score_match, ok));
    println!("Endstand abweichend : {}", score_mismatch);

    if !failures.is_empty() {
        let mut by_cat: BTreeMap<&str, usize> = BTreeMap::new();
        for f in &failures { *by_cat.entry(f.category.as_str()).or_default() += 1; }
        println!("\nAbbruchgründe:");
        for (cat, n) in &by_cat { println!("  {n:>5}×  {cat}"); }
        println!("\nErste fünf:");
        for f in failures.iter().take(5) {
            println!("  {} | Halbzug {} | {:?}\n      {}", f.file, f.ply, f.token, f.detail);
        }
    }

    println!("\nSchach-Erkennung: Notation '+' gegen eigene Zaehlung");
    println!("  {:>10} {:>10} {:>10}", "Notation", "wir", "Zuege");
    for ((noted, ours), n) in &check_matrix {
        let mark = if noted == ours { "" } else { "   <- weicht ab" };
        println!("  {noted:>10} {ours:>10} {n:>10}{mark}");
    }

    let agree = |m: &BTreeMap<(usize, usize), usize>| -> (usize, usize) {
        let mut ok = 0; let mut bad = 0;
        for ((a, b), n) in m { if a == b { ok += n } else { bad += n } }
        (ok, bad)
    };
    let (a1, b1) = agree(&check_matrix);
    let (a2, b2) = agree(&fresh_matrix);
    println!("\nNotation gegen Zustand nach dem Zug : {a1} gleich, {b1} abweichend");
    println!("Notation gegen neu gegebene Schachs: {a2} gleich, {b2} abweichend");
    println!("  {:>10} {:>10} {:>10}", "Notation", "neu", "Zuege");
    for ((noted, fresh), n) in &fresh_matrix {
        let mark = if noted == fresh { "" } else { "   <- weicht ab" };
        println!("  {noted:>10} {fresh:>10} {n:>10}{mark}");
    }

    println!("\nEndstands-Varianten, exakte Treffer von {ok}:");
    for (i, name) in ["A: 3/Koenig", "B: 1/Koenig", "C: 3/Koenig + 2/Ueberlebender", "D: 1/Koenig + 2/Ueberlebender"].iter().enumerate() {
        println!("  {name:<30} {:>6}  ({:.1} %)", variant_hits[i],
                 variant_hits[i] as f64 * 100.0 / ok as f64);
    }

    println!("\nMuster \"+2 fuer jeden Ueberlebenden, sonst 0\" nach Zahl der Ueberlebenden:");
    for (n, c) in &plus2_pattern { println!("  {n} uebrig: {c:>5}x  <- passt" ); }
    println!("uebrige Abweichungen:");
    for (n, c) in &other_pattern { println!("  {n} uebrig: {c:>5}x" ); }

    let mut dv: Vec<_> = diffs.iter().filter(|(d, _)| **d != 0).collect();
    if !dv.is_empty() {
        dv.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        println!("\nVerbleibender Fehlbetrag je Spieler (chess.com minus nachgespielt):");
        for (d, n) in dv.iter().take(10) { println!("  {d:>+5}  {n:>6}×"); }
    }
}
