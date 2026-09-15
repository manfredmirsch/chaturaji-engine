//! Den Zug-Prior auf die Besuchsverteilung der eigenen Suche trainieren.
//!
//! # Der Kreislauf
//!
//! Die Baumsuche spielt besser als ihr eigener Prior — sonst brächte sie
//! nichts. Bei 800 Simulationen verteilt sie die Besuche zwar überwiegend nach
//! dem Prior, aber eben nicht nur: Züge, die sich in den Simulationen als gut
//! erweisen, bekommen mehr, als der Prior ihnen zugedacht hatte. Genau diese
//! Differenz ist das Lernsignal.
//!
//! Trainiert man den Prior darauf, die Besuchsverteilung nachzubilden, wird er
//! besser als das menschliche Vorbild, mit dem er heute lernt. Und weil die
//! Suche dem Prior folgt, wird die Suche dadurch besser und erzeugt beim
//! nächsten Durchgang wiederum bessere Ziele. Das ist der Kreislauf, den
//! AlphaZero fährt; hier fehlte bisher nur der Policy-Kopf dafür.
//!
//! Das Bewertungsnetz steckt dagegen an einem Fixpunkt fest, gegen den weder
//! ein besseres Startnetz noch mehr Kapazität noch ein stärkerer Lehrer
//! geholfen haben. Der Prior ist der Teil, den das Selbstspiel bisher gar
//! nicht trainiert hat.
//!
//! # Was die erste Umdrehung ergab: nichts Gutes
//!
//! Am 2026-09-15 durchgerechnet — 6.400 Selbstspielpartien mit MCTS 800,
//! 71.947 Stellungen, daraus ein Netz allein auf der Besuchsverteilung. Es
//! sagt die Wahl der Suche deutlich besser vorher als das menschlich
//! trainierte (Top-1 42,3 gegen 37,3 %) und **spielt trotzdem schlechter**:
//!
//! | Seed | 1 | 42 | 7 | 99 | Mittel |
//! |---|---|---|---|---|---|
//! | v2 gegen v1 | +0,0046 | −0,0567 | −0,0480 | −0,0463 | **−0,037** |
//!
//! Zwei Erklärungen, die beide zutreffen dürften:
//!
//! * **Die Datenmenge.** v1 lernte aus 862.206 Entscheidungen von Spielern ab
//!   2400 Elo — eine Wissensquelle von außen. v2 ersetzte das durch 71.947
//!   Stellungen aus dem eigenen Spiel, also zwölfmal weniger Daten von einem
//!   schwächeren Lehrer. Ersetzen war der Fehler, nicht Ergänzen.
//! * **Das Ziel enthält den Prior selbst.** Die Besuche stammen aus einer
//!   Suche, die v1 folgt. Wer sie nachbildet, lernt „v1 plus Korrektur" — und
//!   übernimmt dabei auch v1s systematische Fehler überall dort, wo die Suche
//!   sie nicht korrigiert hat. Bei 800 Simulationen ist dieser
//!   Verbesserungsschritt schwach, der Nachahmungsfehler aber nicht.
//!
//! # Die zweite Umdrehung: auch nichts
//!
//! | Variante | Mittel gegen v1 |
//! |---|---|
//! | v2 — nur Suche, Kaltstart | −0,037 |
//! | v3 — nur Suche, Warmstart von v1 (lr 0,001) | −0,038 |
//! | v5 — menschliche **und** Suchdaten, Gewicht 4:1 | ±0,000 |
//!
//! Die erste Erklärung oben war falsch: es sind 719.463 Stellungen mit Suche,
//! nicht 71.947 — letzteres war der Testteil. An zu wenig Daten lag es nicht.
//! Der Warmstart bringt ebenfalls nichts, weil der Startpunkt nach wenigen
//! Epochen vergessen ist. Und gemeinsames Training holt v1s Niveau zurück,
//! ohne darüber hinauszukommen: **die Suchkorrektur trägt bei 800
//! Simulationen nichts bei.**
//!
//! An den Rohdaten gemessen (45.244 Stellungen): der Abstand zwischen
//! Spitzenzug und Platz 2 liegt im Median bei 0,163 der Besuche — aber in
//! **29,6 %** der Stellungen unter 0,05. Dort ist die Wahl der Suche ein
//! Münzwurf, und das Netz lernt ihn mit. Kein reines Rauschen, aber ein
//! knappes Drittel davon.
//!
//! Wer es weiterverfolgt: Stellungen mit knappem Abstand herausfiltern oder
//! nach dem Abstand gewichten, und die Partien mit deutlich mehr Simulationen
//! erzeugen, damit der Lehrer klarer über seinem eigenen Prior steht.
//!
//! # Aufruf
//!
//! ```text
//! # Partien mit Besuchsverteilung erzeugen
//! train_nnue --generate spiele.jsonl --weights netz.json --generator mcts \
//!            --iters 800 --record-visits --games 2000
//!
//! # darauf trainieren
//! policy_selfplay --games spiele.jsonl --out policy_v2.json
//! ```

use std::collections::HashMap;

use chaturaji_core::board::Board;
use chaturaji_core::notation::parse_move;
use chaturaji_core::rules::Rules;
use chaturaji_engine::move_features::{fast_features, MoveFeatureContext};
use chaturaji_engine::policy::{MovePrior, PolicyNet};
use chaturaji_trainer::move_model::{accuracy, fit_policy_from, print_accuracy, Decision};

fn main() {
    let mut games  = String::new();
    let mut out    = "policy_selfplay.json".to_string();
    let mut epochs = 30u32;
    let mut lr     = 0.01f32;
    let mut batch  = 4096usize;
    /// Von einem vorhandenen Netz aus weitertrainieren statt bei Zufall zu
    /// beginnen. `"eingebaut"` nimmt Policy v1 aus der Engine.
    let mut init   = String::new();
    /// Menschliche Partien zusätzlich einbeziehen statt nur die eigenen.
    ///
    /// Der erste Anlauf ersetzte 862.206 menschliche Entscheidungen durch
    /// 71.947 eigene und verlor damit 0,037 Platzwert. Beide Quellen zusammen
    /// zu nehmen ist die naheliegende Verbesserung: das Wissen von außen
    /// bleibt, die Suchkorrektur kommt dazu.
    let mut human  = String::new();
    /// Wie stark die Selbstspiel-Stellungen gegenüber den menschlichen zählen.
    ///
    /// Ohne das entschiede die schiere Menge: zwölfmal mehr menschliche
    /// Entscheidungen hieße, die Suchkorrektur ginge im Rauschen unter.
    let mut mix    = 4.0f32;
    // Anteil der Partien, der zum Messen zurückgehalten wird.
    let mut test_anteil = 10usize;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let wert = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--games"  => { games  = wert; i += 1; }
            "--out"    => { out    = wert; i += 1; }
            "--epochs" => { epochs = wert.parse().unwrap_or(epochs); i += 1; }
            "--lr"     => { lr     = wert.parse().unwrap_or(lr); i += 1; }
            "--batch"  => { batch  = wert.parse().unwrap_or(batch); i += 1; }
            "--init"   => { init   = wert; i += 1; }
            "--human"  => { human  = wert; i += 1; }
            "--mix"    => { mix    = wert.parse().unwrap_or(mix); i += 1; }
            "--help" | "-h" => {
                println!("policy_selfplay --games <jsonl> [--out datei] [--epochs n] [--lr f] [--batch n]");
                println!("                [--init eingebaut|<datei>]   von dort aus weitertrainieren");
                println!("                [--human <verz>] [--mix f]   menschliche Partien dazunehmen");
                return;
            }
            other => { eprintln!("unbekanntes Argument: {other}"); std::process::exit(2); }
        }
        i += 1;
    }
    if games.is_empty() { eprintln!("--games fehlt"); std::process::exit(2); }

    let t0 = std::time::Instant::now();
    let (daten, partien, ohne_suche) = sammle(&games);
    if daten.is_empty() {
        eprintln!("Keine Stellungen mit Besuchsverteilung — wurde mit --record-visits erzeugt?");
        std::process::exit(1);
    }
    println!(
        "{partien} Partien, {} Stellungen mit Suche ({ohne_suche} ohne), {:.1} s",
        daten.len(), t0.elapsed().as_secs_f32(),
    );

    // Nach Partien schneiden, nicht nach Stellungen: Stellungen derselben
    // Partie in Lern- und Testteil wären eine verdeckte Überschneidung.
    let schnitt = daten.len() * (100 - test_anteil.min(50)) / 100;
    let mut daten = daten;
    let test  = daten.split_off(schnitt);
    let mut train = daten;
    let aus_suche = train.len();

    // Menschliche Entscheidungen dazu, mit eigenem Gewicht. Getestet wird
    // weiterhin nur gegen die Suche — die Frage ist ja, ob das Ergebnis die
    // Suche besser nachbildet, nicht ob es Menschen besser nachahmt.
    if !human.is_empty() {
        let mut menschlich = chaturaji_trainer::move_model::collect(&human, 0);
        for d in &mut menschlich { d.weight = 1.0; }
        // Die Selbstspiel-Stellungen hochgewichten, sonst entschiede die
        // schiere Menge.
        for d in &mut train { d.weight = mix; }
        println!("dazu {} menschliche Entscheidungen (Gewicht 1 gegen {mix})", menschlich.len());
        train.extend(menschlich);
    }
    println!("Lernen auf {} ({aus_suche} aus der Suche), Test auf {}\n", train.len(), test.len());

    let (start, woher) = match init.as_str() {
        "" => (PolicyNet::new(chaturaji_engine::move_features::N_FEATURES, 20260915),
               "Zufall".to_string()),
        "eingebaut" => (PolicyNet::default(), "Policy v1 (eingebaut)".to_string()),
        pfad => {
            let txt = std::fs::read_to_string(pfad)
                .unwrap_or_else(|e| { eprintln!("{pfad}: {e}"); std::process::exit(2) });
            let n: PolicyNet = serde_json::from_str(&txt)
                .unwrap_or_else(|e| { eprintln!("{pfad}: {e}"); std::process::exit(2) });
            (n, pfad.to_string())
        }
    };
    println!("─── Policy-Netz auf der Besuchsverteilung, Start: {woher} ───");
    let (netz, letzte, erste) =
        fit_policy_from(start, &train, !human.is_empty(), epochs, lr, batch, 20260915);
    println!("  {} Parameter, {:+.5} → {:+.5}", netz.param_count(), erste, letzte);

    // Gemessen wird gegen den meistbesuchten Zug: trifft das Modell den Zug,
    // den die Suche am Ende gespielt hätte? Das ist dieselbe Größe wie bei den
    // menschlichen Partien, nur mit der Suche als Maßstab.
    println!("\n═══ Trefferquoten gegen die Wahl der Suche ═══\n");
    let zufall = accuracy(&test, |_| 0.0);
    println!("  {} Stellungen, ∅ {:.1} legale Züge\n", zufall.n, zufall.avg_moves);
    print_accuracy("Zufall", &zufall);
    let eingebaut = PolicyNet::default();
    print_accuracy("Policy v1 (menschlich)", &accuracy(&test, |f| eingebaut.logit(f)));
    print_accuracy("Policy v2 (Suche)",      &accuracy(&test, |f| netz.logit(f)));

    let mut netz = netz;
    netz.note = format!("Policy auf Besuchsverteilung aus {partien} Partien, Start: {woher}");
    match std::fs::write(&out, serde_json::to_string(&netz).unwrap_or_default()) {
        Ok(()) => println!("\nGeschrieben: {out}"),
        Err(e) => eprintln!("\nSchreiben fehlgeschlagen: {e}"),
    }
    println!("Gesamtzeit {:.1} s", t0.elapsed().as_secs_f32());
}

/// Liest die JSONL-Datei, spielt jede Partie nach und baut je Stellung mit
/// Besuchsverteilung eine Entscheidung.
///
/// Rückgabe: Entscheidungen, Zahl der Partien, Zahl der übersprungenen
/// Halbzüge (Buchzug, ε-Zufallszug — dort hat keine Suche stattgefunden).
fn sammle(pfad: &str) -> (Vec<Decision>, usize, usize) {
    let text = match std::fs::read_to_string(pfad) {
        Ok(t) => t,
        Err(e) => { eprintln!("{pfad}: {e}"); std::process::exit(1); }
    };

    let mut alle = Vec::new();
    let mut partien = 0usize;
    let mut ohne_suche = 0usize;

    for zeile in text.lines() {
        let json: serde_json::Value = match serde_json::from_str(zeile) {
            Ok(j) => j,
            Err(_) => continue,
        };
        let zuege: Vec<&str> = json["moves"].as_str().unwrap_or("").split_whitespace().collect();
        let besuche = match json["visits"].as_array() {
            Some(v) => v,
            None => continue,
        };
        partien += 1;

        let mut board = Board::default();
        for (ply, zug_text) in zuege.iter().enumerate() {
            let legal = Rules::legal_moves(&board);
            if legal.is_empty() { break; }

            // Die Besuchsverteilung dieses Halbzugs, als (von, nach) → Besuche.
            let flach: Vec<u32> = besuche.get(ply)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default();

            if flach.len() >= 3 {
                let mut karte: HashMap<(u8, u8), u32> = HashMap::new();
                for t in flach.chunks(3) {
                    if t.len() == 3 { karte.insert((t[0] as u8, t[1] as u8), t[2]); }
                }
                let summe: f32 = karte.values().map(|&n| n as f32).sum::<f32>().max(1.0);

                let ctx = MoveFeatureContext::new(&board);
                let feats: Vec<_> = legal.iter()
                    .map(|mv| fast_features(&board, mv, &ctx))
                    .collect();
                let target: Vec<f32> = legal.iter()
                    .map(|mv| *karte.get(&(mv.from, mv.to)).unwrap_or(&0) as f32 / summe)
                    .collect();
                // Der meistbesuchte Zug — die Wahl, die die Suche getroffen hat.
                let chosen = target.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);

                if legal.len() > 1 {
                    alle.push(Decision { feats, chosen, weight: 1.0, target: Some(target) });
                }
            } else {
                ohne_suche += 1;
            }

            let mv = match parse_move(&board, zug_text)
                .ok()
                .and_then(|m| legal.iter().find(|l| l.from == m.from && l.to == m.to).copied()) {
                Some(m) => m,
                None => break,   // Partie nicht nachspielbar — Rest verwerfen
            };
            board = Rules::apply_with_effects(&board, mv);
        }
    }

    (alle, partien, ohne_suche)
}
