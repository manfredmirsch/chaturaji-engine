//! NNUE-Netz: 1293 → 256 → 64 → 4
//!
//! Architektur:
//!   L1: 1280 binäre PS-Features + 13 dichte Zustands-Features → 256, ReLU
//!   L2 (dense):  256 → 64, ReLU
//!   L3 (output): 64 → 4, tanh
//!
//! Der binäre Teil von L1 wird sparse berechnet: nur besetzte Felder (max. 32)
//! tragen bei, also O(|aktiv| × H1) statt O(INPUT_SIZE × H1). Der dichte Block
//! ist mit 13 Spalten klein genug, um ihn einfach durchzurechnen.
//!
//! Ausgabe: erwartete Platzierung je Spieler in [−1, 1], dieselbe Einheit wie
//! `chaturaji_engine::utility` (Platz 1 → 1, Platz 4 → −1). Deshalb sind Netz-
//! und Handbewertung an einem Blattknoten direkt vergleichbar.
//!
//! Optimierer: Adam (β1=0.9, β2=0.999, ε=1e-8)
//! Training:   TD(λ) mit Eligibility Traces

use chaturaji_core::board::Board;

use crate::features::{
    dense_features, for_each_feature, DENSE_FEATURES, INPUT_SIZE, PIECE_FEATURES,
};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

pub const H1: usize = 256;
pub const H2: usize = 64;
pub const OUTPUT: usize = 4;

const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

// ─── Layer ───────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub w: Vec<Vec<f32>>,           // [out][in]
    pub b: Vec<f32>,                // [out]
    #[serde(skip)] pub mw: Vec<Vec<f32>>,
    #[serde(skip)] pub mb: Vec<f32>,
    #[serde(skip)] pub vw: Vec<Vec<f32>>,
    #[serde(skip)] pub vb: Vec<f32>,
}

impl Layer {
    pub fn new_he(in_size: usize, out_size: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let std = (2.0 / in_size as f32).sqrt();
        let dist = Normal::new(0.0f32, std).unwrap();
        let w = (0..out_size)
            .map(|_| (0..in_size).map(|_| dist.sample(&mut rng)).collect())
            .collect();
        Self {
            w,
            b: vec![0.0f32; out_size],
            mw: vec![vec![0.0f32; in_size]; out_size],
            mb: vec![0.0f32; out_size],
            vw: vec![vec![0.0f32; in_size]; out_size],
            vb: vec![0.0f32; out_size],
        }
    }
}

// ─── NnueNetwork ─────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct NnueNetwork {
    pub l1: Layer,
    pub l2: Layer,
    pub l3: Layer,
    pub lr: f32,
    #[serde(default = "default_beta1")]
    pub momentum: f32,
    pub steps: u64,
}

fn default_beta1() -> f32 { BETA1 }

impl NnueNetwork {
    pub fn new(lr: f32, _momentum: f32) -> Self {
        Self {
            l1: Layer::new_he(INPUT_SIZE, H1, 0xDEAD_BEEF_1234_5678),
            l2: Layer::new_he(H1, H2,         0xCAFE_BABE_ABCD_EF01),
            l3: Layer::new_he(H2, OUTPUT,      0x1234_5678_9ABC_DEF0),
            lr,
            momentum: BETA1,
            steps: 0,
        }
    }

    /// Forward-Pass aus einer Stellung.
    ///
    /// Nimmt bewusst das ganze `Board` und nicht nur die Bitboards: Punktestand,
    /// Zugrecht und Spielphase gehören zur Eingabe, und eine Signatur, die sie
    /// gar nicht erst entgegennimmt, lädt zum Weglassen ein.
    pub fn forward(&self, board: &Board) -> [f32; 4] {
        let dense = dense_features(board);
        let a1 = self.l1_activations(&board.bb, &dense);
        let a2 = dense_relu(&self.l2, &a1);
        let a3 = dense_tanh(&self.l3, &a2);
        [a3[0], a3[1], a3[2], a3[3]]
    }

    /// Forward-Pass mit Zwischenwerten für Backpropagation.
    pub fn forward_full(&self, board: &Board) -> ForwardCache {
        let dense = dense_features(board);
        let (z1, a1) = self.l1_full(&board.bb, &dense);
        let (z2, a2) = dense_layer_relu(&self.l2, &a1);
        let (z3, a3) = dense_layer_tanh(&self.l3, &a2);
        ForwardCache { bb: board.bb, dense, z1, a1, z2, a2, z3, a3 }
    }

    /// Backpropagation → akkumuliert Eligibility Traces.
    /// L1 wird sparse über die Bitboards aus `cache.bb` aktualisiert.
    pub fn backward_into_traces(&self, cache: &ForwardCache, traces: &mut Traces, lambda: f32) {
        // L3: d/d(tanh) = 1 - tanh²
        let d_tanh: Vec<f32> = cache.a3.iter().map(|&a| 1.0 - a * a).collect();
        for i in 0..OUTPUT {
            let d = d_tanh[i];
            for j in 0..H2 {
                traces.l3w[i][j] = lambda * traces.l3w[i][j] + d * cache.a2[j];
            }
            traces.l3b[i] = lambda * traces.l3b[i] + d;
        }

        // L2
        let mut delta2 = vec![0.0f32; H2];
        for j in 0..H2 {
            let g: f32 = (0..OUTPUT).map(|i| self.l3.w[i][j] * d_tanh[i]).sum();
            delta2[j] = g * if cache.z2[j] > 0.0 { 1.0 } else { 0.0 };
        }
        for i in 0..H2 {
            for j in 0..H1 {
                traces.l2w[i][j] = lambda * traces.l2w[i][j] + delta2[i] * cache.a1[j];
            }
            traces.l2b[i] = lambda * traces.l2b[i] + delta2[i];
        }

        // L1: sparse über Bitboards — kein Vec<usize> nötig
        let mut delta1 = vec![0.0f32; H1];
        for j in 0..H1 {
            let g: f32 = (0..H2).map(|i| self.l2.w[i][j] * delta2[i]).sum();
            delta1[j] = g * if cache.z1[j] > 0.0 { 1.0 } else { 0.0 };
        }
        for_each_feature(&cache.bb, |feat| {
            if !traces.l1_seen[feat] {
                traces.l1_seen[feat] = true;
                traces.l1_seen_list.push(feat);
            }
            for i in 0..H1 {
                traces.l1w[i][feat] = lambda * traces.l1w[i][feat] + delta1[i];
            }
        });
        // Dichter Block: derselbe Trace, nur mit dem Eingabewert skaliert.
        for (k, &x) in cache.dense.iter().enumerate() {
            if x == 0.0 { continue; }
            let col = PIECE_FEATURES + k;
            if !traces.l1_seen[col] {
                traces.l1_seen[col] = true;
                traces.l1_seen_list.push(col);
            }
            for i in 0..H1 {
                traces.l1w[i][col] = lambda * traces.l1w[i][col] + delta1[i] * x;
            }
        }
        for i in 0..H1 {
            traces.l1b[i] = lambda * traces.l1b[i] + delta1[i];
        }
    }

    /// Adam-Update mit TD-Fehler-Skalierung.
    pub fn apply_td_update(&mut self, traces: &Traces, td_error: &[f32; 4]) {
        let lr  = self.lr;
        let t   = (self.steps + 1) as f32;
        let bc1 = 1.0 - BETA1.powf(t);
        let bc2 = 1.0 - BETA2.powf(t);

        macro_rules! adam_step {
            ($mw:expr, $vw:expr, $w:expr, $g:expr) => {{
                $mw = BETA1 * $mw + (1.0 - BETA1) * $g;
                $vw = BETA2 * $vw + (1.0 - BETA2) * $g * $g;
                $w += lr * ($mw / bc1) / (($vw / bc2).sqrt() + ADAM_EPS);
            }};
        }

        // L3 (OUTPUT × H2)
        for i in 0..OUTPUT {
            let err = td_error[i];
            for j in 0..H2 {
                let g = err * traces.l3w[i][j];
                adam_step!(self.l3.mw[i][j], self.l3.vw[i][j], self.l3.w[i][j], g);
            }
            let gb = err * traces.l3b[i];
            adam_step!(self.l3.mb[i], self.l3.vb[i], self.l3.b[i], gb);
        }

        // L2 (H2 × H1)
        for i in 0..H2 {
            let err = td_error[i % OUTPUT];
            for j in 0..H1 {
                let g = err * traces.l2w[i][j];
                adam_step!(self.l2.mw[i][j], self.l2.vw[i][j], self.l2.w[i][j], g);
            }
            let gb = err * traces.l2b[i];
            adam_step!(self.l2.mb[i], self.l2.vb[i], self.l2.b[i], gb);
        }

        // L1 (H1 × INPUT_SIZE) – nur gesehene Features
        for i in 0..H1 {
            let err = td_error[i % OUTPUT];
            for &j in &traces.l1_seen_list {
                let g = err * traces.l1w[i][j];
                adam_step!(self.l1.mw[i][j], self.l1.vw[i][j], self.l1.w[i][j], g);
            }
            let gb = err * traces.l1b[i];
            adam_step!(self.l1.mb[i], self.l1.vb[i], self.l1.b[i], gb);
        }

        self.steps += 1;
    }

    /// Erweitert ein älteres, nur 1280 Spalten breites L1 auf die aktuelle
    /// Eingabebreite und füllt mit Nullen auf.
    ///
    /// Damit lädt ein Netz, das vor Einführung der dichten Features trainiert
    /// wurde, weiterhin und verhält sich exakt wie zuvor — die neuen Eingaben
    /// tragen zunächst nichts bei und werden erst beim Weitertrainieren
    /// gelernt. Ohne das wäre jeder gespeicherte Checkpoint wertlos.
    pub fn ensure_input_size(&mut self) {
        for row in &mut self.l1.w {
            if row.len() < INPUT_SIZE {
                row.resize(INPUT_SIZE, 0.0);
            }
        }
    }

    /// Initialisiert Adam-Momente nach dem Laden aus der DB.
    pub fn init_momentum(&mut self) {
        self.ensure_input_size();
        let init = |layer: &mut Layer, out: usize, inp: usize| {
            if layer.mw.is_empty() { layer.mw = vec![vec![0.0f32; inp]; out]; }
            if layer.mb.is_empty() { layer.mb = vec![0.0f32; out]; }
            if layer.vw.is_empty() { layer.vw = vec![vec![0.0f32; inp]; out]; }
            if layer.vb.is_empty() { layer.vb = vec![0.0f32; out]; }
        };
        init(&mut self.l1, H1, INPUT_SIZE);
        init(&mut self.l2, H2, H1);
        init(&mut self.l3, OUTPUT, H2);
    }

    pub fn param_count(&self) -> usize {
        INPUT_SIZE * H1 + H1 + H1 * H2 + H2 + H2 * OUTPUT + OUTPUT
    }

    /// Standard-Backpropagation für Supervised Learning.
    ///
    /// Korrekte Kettenregel über alle 4 Ausgaben hinweg — kein `i % OUTPUT`-Hack.
    /// Gibt den MSE-Loss zurück.
    pub fn apply_supervised_gradient(&mut self, cache: &ForwardCache, error: &[f32; 4]) -> f32 {
        let loss = error.iter().map(|e| e * e).sum::<f32>() / 4.0;

        let lr  = self.lr;
        let t   = (self.steps + 1) as f32;
        let bc1 = 1.0 - BETA1.powf(t);
        let bc2 = 1.0 - BETA2.powf(t);

        macro_rules! adam_step {
            ($mw:expr, $vw:expr, $w:expr, $g:expr) => {{
                $mw = BETA1 * $mw + (1.0 - BETA1) * $g;
                $vw = BETA2 * $vw + (1.0 - BETA2) * $g * $g;
                $w += lr * ($mw / bc1) / (($vw / bc2).sqrt() + ADAM_EPS);
            }};
        }

        // L3: delta3[o] = error[o] * tanh'(z3[o])
        let delta3: Vec<f32> = (0..OUTPUT)
            .map(|o| error[o] * (1.0 - cache.a3[o] * cache.a3[o]))
            .collect();

        for o in 0..OUTPUT {
            for j in 0..H2 {
                let g = delta3[o] * cache.a2[j];
                adam_step!(self.l3.mw[o][j], self.l3.vw[o][j], self.l3.w[o][j], g);
            }
            adam_step!(self.l3.mb[o], self.l3.vb[o], self.l3.b[o], delta3[o]);
        }

        // L2: delta2[i] = sum_o(l3.w[o][i] * delta3[o]) * relu'(z2[i])
        let delta2: Vec<f32> = (0..H2).map(|i| {
            let g: f32 = (0..OUTPUT).map(|o| self.l3.w[o][i] * delta3[o]).sum();
            g * if cache.z2[i] > 0.0 { 1.0 } else { 0.0 }
        }).collect();

        for i in 0..H2 {
            for j in 0..H1 {
                let g = delta2[i] * cache.a1[j];
                adam_step!(self.l2.mw[i][j], self.l2.vw[i][j], self.l2.w[i][j], g);
            }
            adam_step!(self.l2.mb[i], self.l2.vb[i], self.l2.b[i], delta2[i]);
        }

        // L1: delta1[k] = sum_i(l2.w[i][k] * delta2[i]) * relu'(z1[k])
        let mut delta1 = vec![0.0f32; H1];
        for i in 0..H2 {
            if delta2[i] == 0.0 { continue; }
            for k in 0..H1 {
                delta1[k] += self.l2.w[i][k] * delta2[i];
            }
        }
        for k in 0..H1 {
            if cache.z1[k] <= 0.0 { delta1[k] = 0.0; }
        }

        // L1, binärer Block: grad(l1.w[k][feat]) = delta1[k], weil das Feature 1 ist.
        for_each_feature(&cache.bb, |feat| {
            for k in 0..H1 {
                if delta1[k] == 0.0 { continue; }
                adam_step!(self.l1.mw[k][feat], self.l1.vw[k][feat], self.l1.w[k][feat], delta1[k]);
            }
        });

        // L1, dichter Block: grad = delta1[k] × Eingabewert.
        for (d, &x) in cache.dense.iter().enumerate() {
            if x == 0.0 { continue; }
            let col = PIECE_FEATURES + d;
            for k in 0..H1 {
                if delta1[k] == 0.0 { continue; }
                let g = delta1[k] * x;
                adam_step!(self.l1.mw[k][col], self.l1.vw[k][col], self.l1.w[k][col], g);
            }
        }

        for k in 0..H1 {
            if delta1[k] != 0.0 {
                adam_step!(self.l1.mb[k], self.l1.vb[k], self.l1.b[k], delta1[k]);
            }
        }

        self.steps += 1;
        loss
    }
}

// ─── Hilfsfunktionen Forward ──────────────────────────────────────────────────

fn l1_preactivations(
    layer: &Layer,
    bb: &[[u64; 5]; 4],
    dense: &[f32; DENSE_FEATURES],
) -> Vec<f32> {
    let mut pre = layer.b.clone();
    // Binärer Block: Gewicht zählt einfach, weil das Feature 1 ist.
    for_each_feature(bb, |feat| {
        for i in 0..H1 {
            pre[i] += layer.w[i][feat];
        }
    });
    // Dichter Block: Gewicht × Wert.
    for (k, &x) in dense.iter().enumerate() {
        if x == 0.0 { continue; }
        let col = PIECE_FEATURES + k;
        for i in 0..H1 {
            pre[i] += layer.w[i][col] * x;
        }
    }
    pre
}

impl NnueNetwork {
    fn l1_activations(&self, bb: &[[u64; 5]; 4], dense: &[f32; DENSE_FEATURES]) -> Vec<f32> {
        l1_preactivations(&self.l1, bb, dense)
            .iter()
            .map(|&z| z.max(0.0))
            .collect()
    }

    fn l1_full(
        &self,
        bb: &[[u64; 5]; 4],
        dense: &[f32; DENSE_FEATURES],
    ) -> (Vec<f32>, Vec<f32>) {
        let pre = l1_preactivations(&self.l1, bb, dense);
        let act = pre.iter().map(|&z| z.max(0.0)).collect();
        (pre, act)
    }
}

fn dense_relu(layer: &Layer, input: &[f32]) -> Vec<f32> {
    layer.w.iter().zip(&layer.b).map(|(row, &bias)| {
        (bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()).max(0.0)
    }).collect()
}

fn dense_tanh(layer: &Layer, input: &[f32]) -> Vec<f32> {
    layer.w.iter().zip(&layer.b).map(|(row, &bias)| {
        (bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()).tanh()
    }).collect()
}

fn dense_layer_relu(layer: &Layer, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let pre: Vec<f32> = layer.w.iter().zip(&layer.b).map(|(row, &bias)| {
        bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()
    }).collect();
    let act = pre.iter().map(|&z| z.max(0.0)).collect();
    (pre, act)
}

fn dense_layer_tanh(layer: &Layer, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let pre: Vec<f32> = layer.w.iter().zip(&layer.b).map(|(row, &bias)| {
        bias + row.iter().zip(input).map(|(&w, &x)| w * x).sum::<f32>()
    }).collect();
    let act = pre.iter().map(|&z| z.tanh()).collect();
    (pre, act)
}

// ─── ForwardCache + Traces ────────────────────────────────────────────────────

pub struct ForwardCache {
    pub bb:    [[u64; 5]; 4],            // binäre Eingabe (kein Vec<usize> nötig)
    pub dense: [f32; DENSE_FEATURES],    // Punktestand, Zugrecht, Phase
    pub z1: Vec<f32>, pub a1: Vec<f32>,
    pub z2: Vec<f32>, pub a2: Vec<f32>,
    pub z3: Vec<f32>, pub a3: Vec<f32>,
}

pub struct Traces {
    // L1: sparse – nur gesehene Feature-Spalten werden benutzt
    pub l1w: Vec<Vec<f32>>,      // [H1][INPUT_SIZE]
    pub l1b: Vec<f32>,           // [H1]
    pub l1_seen: Vec<bool>,      // O(1)-Lookup welche Features gesehen wurden
    pub l1_seen_list: Vec<usize>,// sortierte Liste für schnelle Iteration
    // L2 + L3: dense (klein)
    pub l2w: Vec<Vec<f32>>,      // [H2][H1]
    pub l2b: Vec<f32>,
    pub l3w: Vec<Vec<f32>>,      // [OUTPUT][H2]
    pub l3b: Vec<f32>,
}

impl Traces {
    pub fn new() -> Self {
        Self {
            l1w:          vec![vec![0.0; INPUT_SIZE]; H1],
            l1b:          vec![0.0; H1],
            l1_seen:      vec![false; INPUT_SIZE],
            l1_seen_list: Vec::with_capacity(256),
            l2w:          vec![vec![0.0; H1]; H2],
            l2b:          vec![0.0; H2],
            l3w:          vec![vec![0.0; H2]; OUTPUT],
            l3b:          vec![0.0; OUTPUT],
        }
    }

    pub fn reset(&mut self) {
        // L1: nur gesehene Einträge zurücksetzen
        for &j in &self.l1_seen_list {
            for i in 0..H1 { self.l1w[i][j] = 0.0; }
            self.l1_seen[j] = false;
        }
        self.l1_seen_list.clear();
        self.l1b.fill(0.0);
        // L2 + L3
        for row in &mut self.l2w { row.fill(0.0); }
        self.l2b.fill(0.0);
        for row in &mut self.l3w { row.fill(0.0); }
        self.l3b.fill(0.0);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::board::Board;

    #[test]
    fn forward_output_in_tanh_range() {
        let net = NnueNetwork::new(0.001, 0.9);
        let out = net.forward(&Board::default());
        for v in out { assert!((-1.0..=1.0).contains(&v), "output {v} außerhalb tanh-Bereich"); }
    }

    #[test]
    fn param_count_correct() {
        let net = NnueNetwork::new(0.001, 0.9);
        let expected = INPUT_SIZE * H1 + H1 + H1 * H2 + H2 + H2 * OUTPUT + OUTPUT;
        assert_eq!(net.param_count(), expected);
    }

    #[test]
    fn traces_reset_clears_state() {
        let mut t = Traces::new();
        t.l1w[0][5] = 3.14;
        t.l1_seen[5] = true;
        t.l1_seen_list.push(5);
        t.reset();
        assert_eq!(t.l1w[0][5], 0.0);
        assert!(!t.l1_seen[5]);
        assert!(t.l1_seen_list.is_empty());
    }

    #[test]
    fn adam_update_changes_weights() {
        let mut net = NnueNetwork::new(0.001, 0.9);
        net.init_momentum();
        let cache = net.forward_full(&Board::default());
        let mut traces = Traces::new();
        net.backward_into_traces(&cache, &mut traces, 0.7);

        // Nicht an einem einzelnen Gewicht festmachen: hinter `l3.w[0][0]`
        // steht `a2[0]`, und ob dieses ReLU-Neuron bei zufälliger
        // Initialisierung überhaupt lebt, ist Glückssache.
        let before = net.l3.w.clone();
        net.apply_td_update(&traces, &[0.1, -0.1, 0.05, -0.05]);
        let changed = net.l3.w.iter().zip(&before)
            .flat_map(|(a, b)| a.iter().zip(b))
            .filter(|(x, y)| x != y)
            .count();
        assert!(changed > 0, "Adam muss Gewichte ändern");
    }

    /// Der neue dichte Eingabeblock muss auch im Rückwärtspass ankommen —
    /// sonst würden Punktestand, Zugrecht und Phase zwar gelesen, aber nie
    /// gelernt, und der Fehler bliebe still.
    #[test]
    fn dense_input_block_receives_gradient() {
        let mut net = NnueNetwork::new(0.001, 0.9);
        net.init_momentum();

        // Spalte für „Rot am Zug" (dichter Index 8) — in der Startstellung 1.0.
        let col = PIECE_FEATURES + 8;
        let before: Vec<f32> = (0..H1).map(|i| net.l1.w[i][col]).collect();

        let board = Board::default();
        let cache = net.forward_full(&board);
        let error: [f32; 4] = std::array::from_fn(|i| 1.0 - cache.a3[i]);
        net.apply_supervised_gradient(&cache, &error);

        let moved = (0..H1).filter(|&i| net.l1.w[i][col] != before[i]).count();
        assert!(moved > 0, "Gewichte des dichten Blocks müssen sich bewegen");
    }

    /// Eine Spalte, deren Feature in dieser Stellung 0 ist, darf sich nicht
    /// bewegen: der Gradient ist dort exakt null.
    #[test]
    fn inactive_dense_column_gets_no_gradient() {
        let mut net = NnueNetwork::new(0.001, 0.9);
        net.init_momentum();

        // Dichter Index 9 = „Blau am Zug", in der Startstellung 0.0.
        let col = PIECE_FEATURES + 9;
        let before: Vec<f32> = (0..H1).map(|i| net.l1.w[i][col]).collect();

        let board = Board::default();
        let cache = net.forward_full(&board);
        let error: [f32; 4] = std::array::from_fn(|i| 1.0 - cache.a3[i]);
        net.apply_supervised_gradient(&cache, &error);

        for i in 0..H1 {
            assert_eq!(net.l1.w[i][col], before[i],
                "inaktives Feature darf keinen Gradienten bekommen (Neuron {i})");
        }
    }

    /// Ein Checkpoint aus der Zeit vor dem dichten Block muss weiterhin laden
    /// und sich exakt wie vorher verhalten.
    #[test]
    fn old_checkpoints_are_padded_not_rejected() {
        let mut net = NnueNetwork::new(0.001, 0.9);
        for row in &mut net.l1.w {
            row.truncate(PIECE_FEATURES);   // altes Format simulieren
        }
        net.ensure_input_size();

        assert!(net.l1.w.iter().all(|r| r.len() == INPUT_SIZE));
        for row in &net.l1.w {
            for &w in &row[PIECE_FEATURES..] {
                assert_eq!(w, 0.0, "neue Spalten müssen mit Null starten");
            }
        }
        // Und das Netz rechnet weiter.
        let out = net.forward(&Board::default());
        for v in out { assert!((-1.0..=1.0).contains(&v)); }
    }

    #[test]
    fn forward_is_deterministic() {
        let net = NnueNetwork::new(0.001, 0.9);
        let b = Board::default();
        assert_eq!(net.forward(&b), net.forward(&b));
    }

    #[test]
    fn different_boards_give_different_output() {
        let net = NnueNetwork::new(0.001, 0.9);
        let out_start = net.forward(&Board::default());
        let out_empty = net.forward(&Board::empty());
        assert_ne!(out_start, out_empty);
    }
}
