//! Auf welcher Skala arbeitet das Netz wirklich?
//!
//! `c_puct` wiegt in der PUCT-Formel den Erkundungsterm gegen die gelernte
//! Bewertung Q. Der Wert 1,5 wurde seinerzeit daraus hergeleitet, dass die
//! Netzausgabe in [−1, 1] liegt — das ist zwar richtig, aber irreführend: die
//! Werte **spannen** diesen Bereich nicht aus. Was zählt, ist der Abstand
//! zwischen den Zügen einer Stellung, und der ist eine Größenordnung kleiner.
//!
//! Dieses Beispiel misst ihn. Aufruf:
//!
//! ```text
//! cargo run --release -p chaturaji-nnue --example spanne -- pfad/zu/weights.json
//! ```

use chaturaji_core::board::Board;
use chaturaji_core::rules::Rules;
use chaturaji_engine::nnue_network::NnueNetwork;

fn main() {
    let pfad = std::env::args().nth(1).expect("Pfad zum Netz");
    let json = std::fs::read_to_string(&pfad).unwrap();
    let mut netz: NnueNetwork = serde_json::from_str(&json).unwrap();
    netz.ensure_input_size();

    // Über zufällige Partien laufen und die Spanne der Netzwerte je Stellung
    // messen — das ist die Größe, gegen die der PUCT-Erkundungsterm antritt.
    let mut rng_state: u64 = 12345;
    let mut wuerfel = move || { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; rng_state };

    let mut spannen = Vec::new();
    let mut q_luecken = Vec::new();
    for _ in 0..40 {
        let mut b = Board::default();
        for _ in 0..80 {
            if Rules::is_game_over(&b) { break; }
            let zuege = Rules::legal_moves(&b);
            if zuege.is_empty() { break; }
            let sitz = b.to_move.idx();
            // Wert je Folgestellung aus Sicht des Ziehenden — genau das, was
            // MCTS an den Kindern der Wurzel als Q sieht.
            let mut werte: Vec<f32> = zuege.iter()
                .map(|&m| netz.forward(&Rules::apply_with_effects(&b, m))[sitz])
                .collect();
            werte.sort_by(|a, c| a.partial_cmp(c).unwrap());
            if werte.len() >= 2 {
                spannen.push(werte[werte.len()-1] - werte[0]);
                // Abstand zwischen bestem und zweitbestem Zug: die Größe, die
                // die Suche überhaupt auflösen muss.
                q_luecken.push(werte[werte.len()-1] - werte[werte.len()-2]);
            }
            let m = zuege[(wuerfel() as usize) % zuege.len()];
            b = Rules::apply_with_effects(&b, m);
        }
    }
    let mittel = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
    let median = |v: &mut Vec<f32>| { v.sort_by(|a,b| a.partial_cmp(b).unwrap()); v[v.len()/2] };
    println!("Stellungen: {}", spannen.len());
    println!("Spanne der Zugwerte  : Mittel {:.4}  Median {:.4}", mittel(&spannen), median(&mut spannen.clone()));
    println!("Abstand bester–zweiter: Mittel {:.4}  Median {:.4}", mittel(&q_luecken), median(&mut q_luecken.clone()));
}
