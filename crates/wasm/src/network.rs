//! Minimale Netz-Typen für WASM – kompatible Kopie aus dem trainer-Crate.
//! Wird benötigt um trainierte Gewichte (JSON) im Browser laden zu können,
//! ohne rusqlite/SQLite als WASM-Dependency zu ziehen.

use serde::{Deserialize, Serialize};

pub const INPUT_SIZE: usize = 336;
pub const H1: usize = 256;
pub const H2: usize = 128;
pub const OUTPUT: usize = 4;

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
    #[serde(skip)]
    pub dw: Vec<Vec<f32>>,
    #[serde(skip)]
    pub db: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub l1: Layer,
    pub l2: Layer,
    pub l3: Layer,
    pub lr: f32,
    pub momentum: f32,
    pub steps: u64,
}

impl Network {
    pub fn forward(&self, input: &[f32]) -> [f32; 4] {
        let a1 = relu(&linear(&self.l1, input));
        let a2 = relu(&linear(&self.l2, &a1));
        let a3 = tanh_act(&linear(&self.l3, &a2));
        [a3[0], a3[1], a3[2], a3[3]]
    }

    pub fn param_count(&self) -> usize {
        INPUT_SIZE * H1 + H1 + H1 * H2 + H2 + H2 * OUTPUT + OUTPUT
    }
}

fn linear(layer: &Layer, input: &[f32]) -> Vec<f32> {
    layer.w.iter().zip(&layer.b).map(|(row, &bias)| {
        bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()
    }).collect()
}

fn relu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

fn tanh_act(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| x.tanh()).collect()
}

/// Feature-Extraktion (muss identisch zu trainer/src/features.rs sein)
pub fn extract(board: &chaturaji_core::board::Board) -> Vec<f32> {
    use chaturaji_core::board::rank_of;
    use chaturaji_core::piece::{Color, PieceKind};

    let mut f = Vec::with_capacity(INPUT_SIZE);

    // 1. Figuranzahl pro (Spieler, Typ): 4×5 = 20
    for c in Color::ALL {
        for k in PieceKind::ALL {
            let count = board.bb[c.idx()][k.idx()].count_ones() as f32;
            let max_count: f32 = match k {
                PieceKind::Pawn => 4.0, _ => 1.0,
            };
            f.push(count / max_count.max(1.0));
        }
    }
    // 2. Pawn-Fortschritt: 4
    for c in Color::ALL {
        let pawns = board.bb[c.idx()][PieceKind::Pawn.idx()];
        let count = pawns.count_ones();
        if count == 0 { f.push(0.0); continue; }
        let total: u32 = (0u8..64)
            .filter(|&sq| pawns & (1u64 << sq) != 0)
            .map(|sq| {
                let r = rank_of(sq) as u32;
                match c { Color::Red | Color::Green => r, _ => 7 - r }
            }).sum();
        f.push(total as f32 / count as f32 / 7.0);
    }
    // 3. Zentrale Kontrolle: 4
    const CENTER: u64 = {
        let sqs = [18u64,19,20,21,26,27,28,29,34,35,36,37,42,43,44,45];
        let mut m = 0u64; let mut i = 0;
        while i < sqs.len() { m |= 1u64 << sqs[i]; i += 1; } m
    };
    for c in Color::ALL {
        let occ = board.bb[c.idx()].iter().fold(0u64, |a, &b| a | b);
        f.push((occ & CENTER).count_ones() as f32 / 16.0);
    }
    // 4. Mobilität-Approximation: 4
    for c in Color::ALL {
        if !board.active[c.idx()] { f.push(0.0); continue; }
        let b = board.bb[c.idx()][PieceKind::Bishop.idx()].count_ones() as f32;
        let n = board.bb[c.idx()][PieceKind::Knight.idx()].count_ones() as f32;
        let s = board.bb[c.idx()][PieceKind::Boat.idx()].count_ones() as f32;
        let p = board.bb[c.idx()][PieceKind::Pawn.idx()].count_ones() as f32;
        f.push(((b*7.0 + n*4.0 + s*2.0 + p*1.0) / 40.0).min(1.0));
    }
    // 5. Aktiv/Inaktiv: 4
    for c in Color::ALL { f.push(if board.active[c.idx()] { 1.0 } else { 0.0 }); }
    // 6. Normalisierte Punkte: 4
    for c in Color::ALL { f.push((board.scores.get(c) as f32 / 200.0).clamp(-1.0, 1.0)); }
    // 7. Wer zieht (One-Hot): 4
    for c in Color::ALL { f.push(if board.to_move == c { 1.0 } else { 0.0 }); }
    // 8. Spielphase (One-Hot): 3
    let total: u32 = Color::ALL.iter()
        .map(|&c| board.bb[c.idx()].iter().map(|b| b.count_ones()).sum::<u32>()).sum();
    let phase = if total >= 24 { 0 } else if total >= 12 { 1 } else { 2 };
    for i in 0..3 { f.push(if phase == i { 1.0 } else { 0.0 }); }
    // 9. Halbzug: 1
    f.push((board.half_moves as f32 / 200.0).min(1.0));
    // 10. König-Sicherheit: 4
    for c in Color::ALL {
        let kb = board.bb[c.idx()][PieceKind::King.idx()];
        if kb == 0 { f.push(0.0); continue; }
        let sq = kb.trailing_zeros() as u8;
        let file = (sq % 8) as f32;
        let rank = (sq / 8) as f32;
        f.push(((file - 3.5).abs() + (rank - 3.5).abs()) / 7.0);
    }
    // 11. Schiff-Bestand: 4
    for c in Color::ALL {
        f.push(board.bb[c.idx()][PieceKind::Boat.idx()].count_ones() as f32 / 4.0);
    }
    // Padding auf INPUT_SIZE
    while f.len() < INPUT_SIZE { f.push(0.0); }
    f.truncate(INPUT_SIZE);
    f
}
