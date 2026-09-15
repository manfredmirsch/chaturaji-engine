//! Zugbewertung aus den Partien der Top-Spieler schätzen.
//!
//! # Die Frage
//!
//! Welche Aspekte eines Zuges wiegen wie schwer? Bisher steht darauf nur eine
//! geratene Antwort: `ordering::score_move` sortiert nach MVV-LVA und
//! Promotion, `move_priority` im Self-Play nach Schlagwert. Beide Zahlen hat
//! nie jemand gemessen.
//!
//! Die Daten für eine gemessene Antwort liegen längst da. Jede der 6.366
//! Partien aus `game_data/` enthält rund 120 Entscheidungen eines Spielers ab
//! 2400: eine Stellung, ~30 legale Züge, und welchen davon er gewählt hat. Das
//! sind gut 760.000 Beobachtungen — samt der Züge, die jeweils **nicht**
//! gewählt wurden, und die sind der eigentliche Informationsgehalt.
//!
//! # Das Modell
//!
//! Konditionale Logit-Regression über die legale Zugmenge:
//!
//! ```text
//! P(Zug m | Stellung) = exp(w · f(m)) / Σ_{m' legal} exp(w · f(m'))
//! ```
//!
//! Das ist die formale Entsprechung von „Aspekte gegeneinander aufwiegen": ein
//! Gewicht je Merkmal, und die Wahl fällt auf den Zug mit der höchsten
//! gewichteten Summe. `w` per Maximum Likelihood.
//!
//! Der Gradient der Log-Likelihood einer Entscheidung ist
//! `f(gewählt) − Σ_m P(m)·f(m)` — die Differenz zwischen dem, was der Mensch
//! tat, und dem, was das Modell erwartet hätte. Ist das Modell schon richtig,
//! ist der Gradient null.
//!
//! # Erfolgsgewichtung
//!
//! Jede Entscheidung geht mit dem Platzwert des Ziehenden ein
//! ([`chaturaji_nnue`-Kodierung](crate::move_model::place_weight)): Platz 1
//! zählt am meisten, Platz 4 am wenigsten. Nicht negativ — ein negatives
//! Gewicht würde das Modell in die Gegenrichtung ziehen und „spiele das
//! Gegenteil eines Verlierers" lernen, was etwas anderes ist als „spiele wie
//! ein Gewinner".
//!
//! Die ungewichtete Fassung wird zusätzlich gerechnet. Die Differenz der
//! Gewichtsvektoren zeigt, was Gewinner anders bewerten als der Durchschnitt
//! der starken Spieler.

use std::fs;
use std::path::Path;

use rayon::prelude::*;

use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use chaturaji_engine::move_features::{
    features, MoveFeatureContext, MoveModel, FEATURE_NAMES, N_FEATURES,
};

use crate::opening_book::{move_tokens, parse_meta};

/// Merkmale, die [`chaturaji_engine::move_features::fast_features`] weglässt.
/// Zum Messen, was die Abkürzung in der Suche an Vorhersagekraft kostet.
pub const SLOW_FEATURES: [usize; 2] = [7, 8];
use crate::pgn_import::parse_move_token;
use chaturaji_engine::policy::{PolicyNet, POLICY_HIDDEN};

/// Eine beobachtete Entscheidung: die Merkmale aller legalen Züge, der Index
/// des tatsächlich gespielten, und wie stark sie zählt.
pub struct Decision {
    pub feats:  Vec<[f32; N_FEATURES]>,
    pub chosen: usize,
    pub weight: f32,
    /// Zielverteilung über die Züge, falls es eine gibt.
    ///
    /// Bei menschlichen Partien ist das Ziel ein einzelner Zug, und `chosen`
    /// genügt. Beim Lernen aus der eigenen Suche ist es die **Besuchsverteilung
    /// der Wurzel** — die trägt mehr als nur den besten Zug: dass zwei Züge
    /// fast gleich oft besucht wurden, ist eine Aussage, die ein One-Hot-Ziel
    /// wegwirft.
    pub target: Option<Vec<f32>>,
}

/// Gewicht einer Entscheidung nach der Platzierung des Ziehenden.
///
/// Platz 1 → 1,0; Platz 4 → 0,25. Linear und strikt positiv: die Züge eines
/// Viertplatzierten sollen weniger zählen, aber nicht in die Gegenrichtung
/// ziehen. Sie sind überwiegend normale Züge, keine Fehler.
pub fn place_weight(rank: u32) -> f32 {
    match rank {
        1 => 1.00,
        2 => 0.75,
        3 => 0.50,
        _ => 0.25,
    }
}

// ─── Datensammlung ────────────────────────────────────────────────────────────

/// Liest alle Partien aus `dir` und sammelt die Entscheidungen.
///
/// `limit` begrenzt die Zahl der Partien (0 = alle) — für einen schnellen
/// ersten Durchlauf, bevor man eine Stunde rechnen lässt.
pub fn collect(dir: &str, limit: usize) -> Vec<Decision> {
    let mut paths: Vec<_> = match fs::read_dir(Path::new(dir)) {
        Ok(rd) => rd.flatten().map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
            .collect(),
        Err(e) => { eprintln!("Verzeichnis '{dir}' nicht lesbar: {e}"); return Vec::new(); }
    };
    paths.sort();   // deterministische Reihenfolge, sonst ist der Split zufällig
    if limit > 0 { paths.truncate(limit); }

    println!("{} Partien werden gelesen …", paths.len());

    let alle: Vec<Vec<Decision>> = paths.par_iter()
        .map(|path| decisions_from_file(path).unwrap_or_default())
        .collect();

    alle.into_iter().flatten().collect()
}

fn decisions_from_file(path: &Path) -> Option<Vec<Decision>> {
    let text = fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let pgn4 = json.get("pgn4")?.as_str()?;
    let meta = parse_meta(&json)?;

    let mut out   = Vec::new();
    let mut board = Board::default();

    for token in move_tokens(pgn4) {
        if Rules::is_game_over(&board) { break; }
        let (from_sq, to_sq) = match parse_move_token(&token) {
            Some(x) => x, None => continue,
        };
        let legal = Rules::legal_moves(&board);
        let chosen = match legal.iter().position(|m| m.from == from_sq && m.to == to_sq) {
            Some(i) => i,
            // Stellungsmismatch: ab hier stimmt die Partie nicht mehr, was bis
            // hierher gesammelt wurde, bleibt aber gültig.
            None => break,
        };

        // Mit nur einem legalen Zug ist nichts zu entscheiden und nichts zu
        // lernen — die Wahrscheinlichkeit ist unabhängig von `w` immer 1.
        if legal.len() > 1 {
            let ctx = MoveFeatureContext::new(&board);
            let feats: Vec<[f32; N_FEATURES]> = legal.iter()
                .map(|mv| {
                    let after = Rules::apply_with_effects(&board, *mv);
                    features(&board, mv, &after, &ctx)
                })
                .collect();
            out.push(Decision {
                feats,
                chosen,
                weight: place_weight(meta.ranks[board.to_move.idx()]),
                // Menschliche Partie: das Ziel ist der gespielte Zug.
                target: None,
            });
        }

        board = Rules::apply_with_effects(&board, legal[chosen]);
    }
    Some(out)
}

// ─── Schätzung ────────────────────────────────────────────────────────────────

/// Softmax über die Zugmenge. Der Maximalwert wird abgezogen, damit `exp`
/// nicht überläuft — bei 30 Zügen und großen Gewichten sonst schnell `inf`.
fn probabilities(d: &Decision, w: &[f32]) -> Vec<f32> {
    let scores: Vec<f32> = d.feats.iter()
        .map(|f| f.iter().zip(w).map(|(x, wi)| x * wi).sum())
        .collect();
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let summe: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / summe).collect()
}

pub struct FitStats {
    pub w:            Vec<f32>,
    pub loglik:       f32,
    pub first_loglik: f32,
}

/// Unterhalb dieser Gradientengröße wird nicht aktualisiert.
///
/// Adam normiert den Schritt mit der eigenen Gradientengröße — die Schrittweite
/// hängt also nur vom *Vorzeichen*, nicht vom Betrag ab. Ein Merkmal, das gar
/// keine Information trägt (in jedem Zug derselbe Wert), hat mathematisch
/// Gradient null, numerisch aber einen Rest von etwa 1e-7 aus der
/// Softmax-Summe. Adam macht daraus volle Schritte: nachgerechnet treibt ein
/// Restgradient von 6e-8 das Gewicht in 300 Schritten auf 12,9.
///
/// Aufgefallen ist das an `a_constant_feature_stays_near_zero` — dort bekam
/// `zentrum`, obwohl in beiden Zügen identisch, ein Gewicht von 2,26.
///
/// 1e-6 liegt zwei Größenordnungen über dem Rauschen und mindestens zwei unter
/// dem, was ein schwaches echtes Signal bei ~700.000 gemittelten Beobachtungen
/// liefert.
const GRAD_DEADZONE: f32 = 1e-6;

/// L2-Regularisierung, entkoppelt angewandt (AdamW-Art).
///
/// Hält Gewichte klein, für die es kaum Belege gibt. Klein genug, dass echte
/// Signale davon unberührt bleiben: bei Gleichgewicht gilt `|g| = LAMBDA · |w|`,
/// ein Gewicht von 1 braucht also einen Gradienten von 1e-3.
const LAMBDA: f32 = 1e-3;

/// Maximum-Likelihood-Schätzung per Adam.
///
/// `weighted` schaltet die Erfolgsgewichtung ein. `epochs` sind volle
/// Durchläufe über alle Entscheidungen; der Gradient wird je Durchlauf
/// aufsummiert (Full-Batch), weil 14 Parameter keine Mini-Batches brauchen und
/// Full-Batch reproduzierbar ist.
pub fn fit(data: &[Decision], weighted: bool, epochs: u32, lr: f32) -> FitStats {
    let mut w = vec![0.0f32; N_FEATURES];
    let mut m = vec![0.0f32; N_FEATURES];
    let mut v = vec![0.0f32; N_FEATURES];
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;

    let mut first = 0.0f32;
    let mut last  = 0.0f32;

    for epoch in 1..=epochs.max(1) {
        // Gradient und Log-Likelihood parallel über die Entscheidungen.
        let (grad, loglik, gewicht) = data.par_iter()
            .map(|d| {
                let gewicht = if weighted { d.weight } else { 1.0 };
                let p = probabilities(d, &w);
                let mut g = [0.0f32; N_FEATURES];
                // ∂/∂w log P(gewählt) = f(gewählt) − Σ_m P(m)·f(m)
                for k in 0..N_FEATURES {
                    let erwartet: f32 = p.iter().zip(&d.feats)
                        .map(|(pm, f)| pm * f[k]).sum();
                    g[k] = gewicht * (d.feats[d.chosen][k] - erwartet);
                }
                (g, gewicht * p[d.chosen].max(1e-30).ln(), gewicht)
            })
            .reduce(
                || ([0.0f32; N_FEATURES], 0.0f32, 0.0f32),
                |(mut ga, la, wa), (gb, lb, wb)| {
                    for k in 0..N_FEATURES { ga[k] += gb[k]; }
                    (ga, la + lb, wa + wb)
                },
            );

        let mittel = loglik / gewicht.max(1e-9);
        if epoch == 1 { first = mittel; }
        last = mittel;

        let t = epoch as f32;
        let bc1 = 1.0 - B1.powf(t);
        let bc2 = 1.0 - B2.powf(t);
        for k in 0..N_FEATURES {
            let g = grad[k] / gewicht.max(1e-9);
            // Ohne diese Schranke macht Adam aus Rundungsrauschen echte
            // Gewichte — siehe GRAD_DEADZONE.
            if g.abs() < GRAD_DEADZONE {
                w[k] -= lr * LAMBDA * w[k];
                continue;
            }
            m[k] = B1 * m[k] + (1.0 - B1) * g;
            v[k] = B2 * v[k] + (1.0 - B2) * g * g;
            // Aufstieg: die Log-Likelihood soll größer werden. Der
            // L2-Abzug läuft daneben, nicht über den Gradienten — sonst
            // normierte Adam ihn mit weg.
            w[k] += lr * (m[k] / bc1) / ((v[k] / bc2).sqrt() + EPS);
            w[k] -= lr * LAMBDA * w[k];
        }

        if epoch % 25 == 0 || epoch == 1 || epoch == epochs {
            println!("  Epoche {epoch:>4}/{epochs} | ∅log P(gewählt) {mittel:+.5}");
        }
    }

    FitStats { w, loglik: last, first_loglik: first }
}

// ─── Auswertung ───────────────────────────────────────────────────────────────

pub struct Accuracy {
    pub n:      usize,
    pub top1:   f32,
    pub top3:   f32,
    pub beam6:  f32,
    /// Mittlere Zahl legaler Züge — die Bezugsgröße für den Zufall.
    pub avg_moves: f32,
}

/// Wahrscheinlichkeit, dass der Zug unter den besten `k` landet, wenn
/// Gleichstände zufällig aufgelöst werden.
///
/// `b` = Züge mit echt besserer Bewertung, `t` = gleich bewertete (ohne den
/// Zug selbst). Ohne diese Behandlung wäre das Maß wertlos: eine Heuristik,
/// die fast allen Zügen dieselbe Zahl gibt — `move_priority` setzt jeden
/// ruhigen Zug auf 0 —, hätte nie einen „echt besseren" Konkurrenten und käme
/// auf 100 %. Die Suche sortiert solche Gleichstände aber in beliebiger
/// Reihenfolge; im Mittel trifft sie den richtigen genau mit der
/// Wahrscheinlichkeit unten. So bekommt ein Scorer ohne Information exakt die
/// Zufallsgrundlinie, was er auch verdient.
fn in_top_k(b: usize, t: usize, k: usize) -> f32 {
    if b + t + 1 <= k { return 1.0; }     // die ganze Gleichstandsgruppe passt hinein
    if b >= k         { return 0.0; }     // schon die echt besseren füllen den Beam
    (k - b) as f32 / (t + 1) as f32       // Restplätze auf die Gleichstandsgruppe verteilt
}

/// Trefferquoten eines Scorers: wie oft landet der menschliche Zug auf Platz 1,
/// unter den ersten 3, unter den ersten 6.
///
/// `beam6` ist die Zahl, auf die es ankommt: die Beam-Suche mit `beam_width 6`
/// verwirft alles dahinter, und was sie verwirft, existiert für sie nicht.
pub fn accuracy(data: &[Decision], score: impl Fn(&[f32; N_FEATURES]) -> f32 + Sync) -> Accuracy {
    let (t1, t3, t6, moves) = data.par_iter()
        .map(|d| {
            let s_chosen = score(&d.feats[d.chosen]);
            let mut besser = 0usize;
            let mut gleich = 0usize;
            for (i, f) in d.feats.iter().enumerate() {
                if i == d.chosen { continue; }
                let s = score(f);
                if s > s_chosen { besser += 1; } else if s == s_chosen { gleich += 1; }
            }
            (
                in_top_k(besser, gleich, 1),
                in_top_k(besser, gleich, 3),
                in_top_k(besser, gleich, 6),
                d.feats.len() as f32,
            )
        })
        .reduce(|| (0.0, 0.0, 0.0, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));

    let n = data.len().max(1) as f32;
    Accuracy {
        n: data.len(),
        top1:  t1 / n,
        top3:  t3 / n,
        beam6: t6 / n,
        avg_moves: moves / n,
    }
}

/// Die heutige Self-Play-Heuristik, ausgedrückt über denselben Merkmalsvektor.
///
/// `move_priority` in `crates/nnue/src/selfplay.rs` vergibt Schlagwert (König
/// 100, Boot/Bishop 50, Springer 30, Bauer 10) plus 20 für Promotion. Über die
/// normierten Merkmale nachgebildet: Schlagwert zählt unabhängig davon, ob das
/// Zielfeld gedeckt ist — genau das ist der blinde Fleck, den das gelernte
/// Modell füllen soll.
/// Kopie der Entscheidungen, in der die teuren Merkmale auf null stehen.
///
/// Ein konstant-null gesetztes Merkmal trägt keine Information, sein Gradient
/// ist null, und die Deadzone hält sein Gewicht bei null — das neu geschätzte
/// Modell ist also genau das, was die Suche mit `fast_features` rechnen kann.
pub fn without_slow_features(data: &[Decision]) -> Vec<Decision> {
    data.iter().map(|d| Decision {
        feats: d.feats.iter().map(|f| {
            let mut g = *f;
            for k in SLOW_FEATURES { g[k] = 0.0; }
            g
        }).collect(),
        chosen: d.chosen,
        target: d.target.clone(),
        weight: d.weight,
    }).collect()
}

pub fn heuristik_move_priority(f: &[f32; N_FEATURES]) -> f32 {
    let schlag = (f[4] + f[5]) * 5.0;   // zurück auf die 1..5-Skala
    let wert = if f[6] > 0.0 { 100.0 } else {
        match schlag {
            s if s >= 5.0 => 50.0,
            s if s >= 3.0 => 30.0,
            s if s >= 1.0 => 10.0,
            _             => 0.0,
        }
    };
    wert + if f[2] > 0.0 { 20.0 } else { 0.0 }
}

pub fn print_accuracy(titel: &str, a: &Accuracy) {
    println!(
        "  {titel:<22} Top-1 {:>5.1} %   Top-3 {:>5.1} %   im Beam-6 {:>5.1} %",
        a.top1 * 100.0, a.top3 * 100.0, a.beam6 * 100.0,
    );
}

/// Nebeneinanderstellung zweier Gewichtsvektoren.
pub fn compare_table(a: &MoveModel, b: &MoveModel, name_a: &str, name_b: &str) -> String {
    let mut idx: Vec<usize> = (0..N_FEATURES).collect();
    idx.sort_by(|&i, &j| a.w[j].abs().partial_cmp(&a.w[i].abs()).unwrap());
    let mut s = format!("  {:<20} {:>10} {:>10} {:>10}\n", "Merkmal", name_a, name_b, "Differenz");
    s.push_str(&format!("  {}\n", "─".repeat(52)));
    for i in idx {
        s.push_str(&format!(
            "  {:<20} {:>10.4} {:>10.4} {:>+10.4}\n",
            FEATURE_NAMES[i], a.w[i], b.w[i], a.w[i] - b.w[i],
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entscheidung(feats: Vec<[f32; N_FEATURES]>, chosen: usize) -> Decision {
        Decision { feats, chosen, weight: 1.0 }
    }

    /// Ein Merkmal, das der gewählte Zug hat und alle anderen nicht, muss ein
    /// positives Gewicht bekommen. Das ist die Grundeigenschaft des Verfahrens.
    #[test]
    fn a_feature_that_marks_the_played_move_gets_a_positive_weight() {
        let mut gut  = [0.0f32; N_FEATURES]; gut[2]  = 1.0;   // Umwandlung
        let schlecht = [0.0f32; N_FEATURES];
        let data: Vec<Decision> = (0..50)
            .map(|_| entscheidung(vec![gut, schlecht, schlecht], 0))
            .collect();

        let f = fit(&data, false, 200, 0.05);
        assert!(f.w[2] > 0.5, "umwandlung muss positiv werden, ist {}", f.w[2]);
        assert!(f.loglik > f.first_loglik, "die Log-Likelihood muss steigen");
    }

    /// Und umgekehrt: ein Merkmal, das nur die *nicht* gewählten Züge tragen,
    /// muss negativ werden.
    #[test]
    fn a_feature_that_marks_rejected_moves_gets_a_negative_weight() {
        let neutral = [0.0f32; N_FEATURES];
        let mut schlecht = [0.0f32; N_FEATURES]; schlecht[10] = 1.0;  // eingestellt
        let data: Vec<Decision> = (0..50)
            .map(|_| entscheidung(vec![neutral, schlecht, schlecht], 0))
            .collect();

        let f = fit(&data, false, 200, 0.05);
        assert!(f.w[10] < -0.5, "figur_eingestellt muss negativ werden, ist {}", f.w[10]);
    }

    /// Merkmale, die in jedem Zug gleich aussehen, tragen keine Information —
    /// ihr Gewicht darf nicht davonlaufen.
    #[test]
    fn a_constant_feature_stays_near_zero() {
        let mut f0 = [0.0f32; N_FEATURES]; f0[11] = 1.0; f0[2] = 1.0;
        let mut f1 = [0.0f32; N_FEATURES]; f1[11] = 1.0;
        let data: Vec<Decision> = (0..50).map(|_| entscheidung(vec![f0, f1], 0)).collect();

        let f = fit(&data, false, 200, 0.05);
        assert!(f.w[11].abs() < 0.05, "zentrum ist konstant, muss ~0 bleiben, ist {}", f.w[11]);
        assert!(f.w[2] > 0.5, "das unterscheidende Merkmal muss wirken");
    }

    /// Die Wahrscheinlichkeiten müssen sich zu eins summieren, auch bei
    /// Gewichten, die `exp` ohne Stabilisierung überlaufen ließen.
    #[test]
    fn probabilities_stay_normalised_even_with_huge_weights() {
        let mut a = [0.0f32; N_FEATURES]; a[0] = 1.0;
        let b = [0.0f32; N_FEATURES];
        let d = entscheidung(vec![a, b, b], 0);
        let mut w = vec![0.0f32; N_FEATURES];
        w[0] = 500.0;
        let p = probabilities(&d, &w);
        let summe: f32 = p.iter().sum();
        assert!((summe - 1.0).abs() < 1e-5, "Σp = {summe}");
        assert!(p.iter().all(|x| x.is_finite()), "kein Wert darf inf oder NaN sein");
    }

    /// Ein Scorer ohne Information muss exakt die Zufallsgrundlinie bekommen:
    /// bei 30 gleich bewerteten Zügen 1/30 für Platz 1 und 6/30 für den Beam.
    ///
    /// Vorher zählte ein Gleichstand als Treffer, und `move_priority` — das
    /// jeden ruhigen Zug auf 0 setzt — kam damit auf 99,9 % im Beam-6. Der
    /// Vergleich war dadurch wertlos.
    #[test]
    fn a_scorer_without_information_gets_the_random_baseline() {
        let f = [0.0f32; N_FEATURES];
        let data: Vec<Decision> = (0..10).map(|_| entscheidung(vec![f; 30], 0)).collect();
        let a = accuracy(&data, |_| 0.0);
        assert!((a.top1  - 1.0 / 30.0).abs() < 1e-6, "Top-1 war {}", a.top1);
        assert!((a.beam6 - 6.0 / 30.0).abs() < 1e-6, "Beam-6 war {}", a.beam6);
        assert_eq!(a.avg_moves, 30.0);
    }

    /// Passt die ganze Gleichstandsgruppe in den Beam, ist es ein sicherer
    /// Treffer; füllen die echt besseren ihn schon, ein sicherer Fehlschlag.
    #[test]
    fn tie_handling_covers_both_extremes() {
        assert_eq!(in_top_k(0, 3, 6), 1.0, "4 gleiche Züge passen in 6 Plätze");
        assert_eq!(in_top_k(6, 0, 6), 0.0, "6 echt bessere füllen den Beam");
        assert_eq!(in_top_k(4, 3, 6), 0.5, "2 Restplätze auf 4 Gleiche");
    }

    /// Die Erfolgsgewichtung muss monoton in der Platzierung sein und positiv
    /// bleiben — ein negatives Gewicht würde „spiele das Gegenteil" lernen.
    #[test]
    fn place_weight_is_monotone_and_positive() {
        let w: Vec<f32> = (1..=4).map(place_weight).collect();
        assert!(w.windows(2).all(|p| p[0] > p[1]), "monoton fallend: {w:?}");
        assert!(w.iter().all(|&x| x > 0.0), "strikt positiv: {w:?}");
    }
}

// ─── Policy-Netz ──────────────────────────────────────────────────────────────

/// Trainiert ein [`PolicyNet`] auf derselben Verlustfunktion wie [`fit`].
///
/// Die Struktur bleibt die konditionale Logit-Regression: Score je Zug, Softmax
/// über die legale Zugmenge, Maximum Likelihood auf dem tatsächlich gespielten
/// Zug. Nur die Score-Funktion ist jetzt ein Netz statt einer Linearform.
///
/// Anders als beim linearen Modell wird in **Mini-Batches** gerechnet. Bei 14
/// Parametern ist Full-Batch reproduzierbar und billig; bei einigen hundert
/// lohnt sich das nicht mehr, und ein Netz braucht ohnehin mehr Schritte als
/// Epochen. Die Reihenfolge der Batches ist deterministisch aus dem Seed
/// abgeleitet — ein Trainingsergebnis, das vom Zufall der Durchmischung
/// abhängt, taugt nicht als Messung.
pub fn fit_policy(
    data: &[Decision], weighted: bool, epochs: u32, lr: f32, batch: usize, seed: u64,
) -> (PolicyNet, f32, f32) {
    fit_policy_from(PolicyNet::new(N_FEATURES, seed), data, weighted, epochs, lr, batch, seed)
}

/// Wie [`fit_policy`], aber von einem vorhandenen Netz aus.
///
/// Gebraucht für den zweiten Anlauf des Selbstspiel-Kreislaufs. Der erste
/// begann bei Zufall und **ersetzte** damit, was aus 862.206 menschlichen
/// Entscheidungen gelernt war, durch 71.947 eigene Stellungen — zwölfmal
/// weniger Daten von einem schwächeren Lehrer. Ergebnis: −0,037 Platzwert.
///
/// Von v1 aus weiterzutrainieren behält das menschliche Wissen und legt die
/// Suchkorrektur darauf. Die Lernrate gehört dabei deutlich kleiner gewählt:
/// bei der Rate des Neuanfangs wäre der Startpunkt nach wenigen Schritten
/// vergessen, und man landete wieder bei v2.
pub fn fit_policy_from(
    start: PolicyNet, data: &[Decision], weighted: bool,
    epochs: u32, lr: f32, batch: usize, seed: u64,
) -> (PolicyNet, f32, f32) {
    let inputs = N_FEATURES;
    let mut netz = start;

    // Adam-Momente, in derselben Form wie die Gewichte.
    let mut m1 = vec![vec![0.0f32; inputs]; POLICY_HIDDEN];
    let mut v1 = vec![vec![0.0f32; inputs]; POLICY_HIDDEN];
    let mut mb = vec![0.0f32; POLICY_HIDDEN];
    let mut vb = vec![0.0f32; POLICY_HIDDEN];
    let mut m2 = vec![0.0f32; POLICY_HIDDEN];
    let mut v2 = vec![0.0f32; POLICY_HIDDEN];
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;

    let mut reihenfolge: Vec<usize> = (0..data.len()).collect();
    let mut zustand = seed | 1;
    let mut wuerfel = move || {
        zustand ^= zustand << 13; zustand ^= zustand >> 7; zustand ^= zustand << 17; zustand
    };

    let mut schritt = 0f32;
    let (mut erste, mut letzte) = (0.0f32, 0.0f32);

    for epoch in 1..=epochs.max(1) {
        // Fisher-Yates mit demselben Würfel wie oben.
        for i in (1..reihenfolge.len()).rev() {
            let j = (wuerfel() as usize) % (i + 1);
            reihenfolge.swap(i, j);
        }

        let mut loglik = 0.0f64;
        let mut gewicht_gesamt = 0.0f64;

        for block in reihenfolge.chunks(batch.max(1)) {
            // Gradienten je Batch, parallel über die Entscheidungen.
            let (g1, gb, g2, ll, gw) = block.par_iter()
                .map(|&idx| {
                    let d = &data[idx];
                    let gewicht = if weighted { d.weight } else { 1.0 };

                    // Vorwärts: Score und verborgene Aktivierung je Zug.
                    let mut scores = Vec::with_capacity(d.feats.len());
                    let mut hidden = Vec::with_capacity(d.feats.len());
                    for f in &d.feats {
                        let (s, h) = netz.forward(f);
                        scores.push(s);
                        hidden.push(h);
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                    let summe: f32 = exps.iter().sum::<f32>().max(1e-30);
                    let p: Vec<f32> = exps.iter().map(|e| e / summe).collect();

                    // ∂ log P(gewählt) / ∂ score(m) = [m == gewählt] − P(m)
                    let mut a1 = vec![vec![0.0f32; inputs]; POLICY_HIDDEN];
                    let mut ab = vec![0.0f32; POLICY_HIDDEN];
                    let mut a2 = vec![0.0f32; POLICY_HIDDEN];
                    // ∂/∂score(m) der Kreuzentropie ist π(m) − P(m); das
                    // One-Hot-Ziel ist davon nur der Sonderfall.
                    let ziel = |i: usize| match &d.target {
                        Some(t) => t.get(i).copied().unwrap_or(0.0),
                        None    => (i == d.chosen) as u8 as f32,
                    };
                    for (i, f) in d.feats.iter().enumerate() {
                        let ds = gewicht * (ziel(i) - p[i]);
                        if ds == 0.0 { continue; }
                        for h in 0..POLICY_HIDDEN {
                            let akt = hidden[i][h];
                            a2[h] += ds * akt;
                            if akt <= 0.0 { continue; }   // ReLU sperrt
                            let dh = ds * netz.w2[h];
                            ab[h] += dh;
                            for (k, &x) in f.iter().enumerate() { a1[h][k] += dh * x; }
                        }
                    }
                    let ll = match &d.target {
                        Some(t) => t.iter().zip(&p)
                            .map(|(pi, pm)| pi * pm.max(1e-30).ln()).sum::<f32>(),
                        None => p[d.chosen].max(1e-30).ln(),
                    };
                    (a1, ab, a2, gewicht * ll, gewicht)
                })
                .reduce(
                    || (vec![vec![0.0f32; inputs]; POLICY_HIDDEN],
                        vec![0.0f32; POLICY_HIDDEN], vec![0.0f32; POLICY_HIDDEN], 0.0f32, 0.0f32),
                    |(mut a1, mut ab, mut a2, la, wa), (b1v, bb, b2v, lb, wb)| {
                        for h in 0..POLICY_HIDDEN {
                            for k in 0..inputs { a1[h][k] += b1v[h][k]; }
                            ab[h] += bb[h];
                            a2[h] += b2v[h];
                        }
                        (a1, ab, a2, la + lb, wa + wb)
                    },
                );

            loglik += ll as f64;
            gewicht_gesamt += gw as f64;
            let norm = gw.max(1e-9);

            schritt += 1.0;
            let bc1 = 1.0 - B1.powf(schritt);
            let bc2 = 1.0 - B2.powf(schritt);
            let mut adam = |w: &mut f32, m: &mut f32, v: &mut f32, g: f32| {
                *m = B1 * *m + (1.0 - B1) * g;
                *v = B2 * *v + (1.0 - B2) * g * g;
                // Aufstieg wie beim linearen Modell, L2 daneben.
                *w += lr * (*m / bc1) / ((*v / bc2).sqrt() + EPS);
                *w -= lr * LAMBDA * *w;
            };
            for h in 0..POLICY_HIDDEN {
                for k in 0..inputs {
                    adam(&mut netz.w1[h][k], &mut m1[h][k], &mut v1[h][k], g1[h][k] / norm);
                }
                adam(&mut netz.b1[h], &mut mb[h], &mut vb[h], gb[h] / norm);
                adam(&mut netz.w2[h], &mut m2[h], &mut v2[h], g2[h] / norm);
            }
        }

        let mittel = (loglik / gewicht_gesamt.max(1e-9)) as f32;
        if epoch == 1 { erste = mittel; }
        letzte = mittel;
        if epoch % 2 == 0 || epoch == 1 || epoch == epochs {
            println!("  Epoche {epoch:>3}/{epochs} | ∅log P(gewählt) {mittel:+.5}");
        }
    }

    netz.note = format!("Policy-Netz, {POLICY_HIDDEN} verborgene, {inputs} Eingaben");
    (netz, letzte, erste)
}
