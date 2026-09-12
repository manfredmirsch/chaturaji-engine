//! Zugbewertung aus echten Partien schätzen und auswerten.
//!
//!   cargo run --release -p chaturaji-trainer --bin move_model -- \
//!       --games ~/chaturaji/game_data [--limit 200] [--out move_weights.json]
//!
//! Rechnet zwei Modelle — erfolgsgewichtet und ungewichtet — stellt die
//! Gewichte nebeneinander und misst beide gegen die heutige Handheuristik auf
//! einem Zehntel der Partien, das nicht zum Lernen benutzt wurde.

use chaturaji_engine::move_features::{MoveModel, N_FEATURES};
use chaturaji_trainer::move_model::{
    accuracy, collect, compare_table, fit, heuristik_move_priority, print_accuracy,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut games  = "/home/manfred/chaturaji/game_data".to_string();
    let mut out    = "move_weights.json".to_string();
    let mut limit  = 0usize;
    let mut epochs = 300u32;
    let mut lr     = 0.05f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games"  => { i += 1; if i < args.len() { games = args[i].clone(); } }
            "--out"    => { i += 1; if i < args.len() { out   = args[i].clone(); } }
            "--limit"  => { i += 1; if i < args.len() { limit  = args[i].parse().unwrap_or(0); } }
            "--epochs" => { i += 1; if i < args.len() { epochs = args[i].parse().unwrap_or(epochs); } }
            "--lr"     => { i += 1; if i < args.len() { lr     = args[i].parse().unwrap_or(lr); } }
            "--help" | "-h" => {
                println!("move_model [--games <verz>] [--out <datei>] [--limit n] [--epochs n] [--lr f]");
                return;
            }
            other => eprintln!("unbekanntes Argument: {other}"),
        }
        i += 1;
    }

    let t0 = std::time::Instant::now();
    let mut data = collect(&games, limit);
    if data.is_empty() {
        eprintln!("Keine Entscheidungen gesammelt — stimmt --games?");
        std::process::exit(1);
    }
    let zuege: usize = data.iter().map(|d| d.feats.len()).sum();
    println!(
        "{} Entscheidungen, {} bewertete Züge (∅ {:.1} je Stellung) in {:.1} s\n",
        data.len(), zuege, zuege as f32 / data.len() as f32, t0.elapsed().as_secs_f32(),
    );

    // Aufteilen in Lern- und Testteil. Die Partien wurden in sortierter
    // Reihenfolge gelesen, ein zusammenhängender Block am Ende ist also ein
    // sauberer Schnitt nach Partien — nicht nach Stellungen. Stellungen
    // derselben Partie in beiden Töpfen wären eine verdeckte Überschneidung.
    let test_ab = data.len() * 9 / 10;
    let test: Vec<_> = data.split_off(test_ab);
    let train = data;
    println!("Lernen auf {} Entscheidungen, Test auf {}\n", train.len(), test.len());

    println!("─── erfolgsgewichtet ───");
    let gew = fit(&train, true, epochs, lr);
    println!("\n─── ungewichtet ───");
    let ung = fit(&train, false, epochs, lr);

    let m_gew = MoveModel::new(gew.w.clone(), "erfolgsgewichtet, Platz 1..4 = 1.0..0.25");
    let m_ung = MoveModel::new(ung.w.clone(), "ungewichtet");

    println!("\n═══ Gewichte ═══\n");
    print!("{}", compare_table(&m_gew, &m_ung, "gewichtet", "ungewicht."));

    println!("\n═══ Trefferquoten auf den Testpartien ═══\n");
    // Ein Scorer, der jedem Zug dieselbe Zahl gibt, liefert bei korrekter
    // Gleichstandsbehandlung genau die Zufallsgrundlinie — deshalb reicht
    // `accuracy` mit konstanter Funktion, es braucht keine eigene Formel.
    let zufall = accuracy(&test, |_| 0.0);
    println!("  {} Entscheidungen, ∅ {:.1} legale Züge\n", zufall.n, zufall.avg_moves);
    print_accuracy("Zufall", &zufall);

    print_accuracy("move_priority (heute)", &accuracy(&test, heuristik_move_priority));
    print_accuracy("gelernt, gewichtet",    &accuracy(&test, |f| dot(&gew.w, f)));
    print_accuracy("gelernt, ungewichtet",  &accuracy(&test, |f| dot(&ung.w, f)));

    match m_gew.save(&out) {
        Ok(()) => println!("\nGewichte (erfolgsgewichtet) geschrieben: {out}"),
        Err(e) => eprintln!("\nSchreiben fehlgeschlagen: {e}"),
    }
    let out_ung = out.replace(".json", "-ungewichtet.json");
    if let Err(e) = m_ung.save(&out_ung) { eprintln!("Schreiben fehlgeschlagen: {e}"); }
    else { println!("Gewichte (ungewichtet) geschrieben: {out_ung}"); }

    println!("\nGesamtzeit {:.1} s", t0.elapsed().as_secs_f32());
}

fn dot(w: &[f32], f: &[f32; N_FEATURES]) -> f32 {
    w.iter().zip(f).map(|(a, b)| a * b).sum()
}
