pub mod db;
pub mod dist;
pub mod eval;
pub mod features;
pub mod gen_train;
pub mod network;
pub mod pgn_import;
pub mod selfplay;
pub mod supervised;
pub mod td;

/// Beide Module sind in die Engine gewandert, damit das WASM-Frontend sie
/// benutzen kann — dieser Kiste hängen `rusqlite` (gebündeltes C) und `rayon`
/// an, die für `wasm32-unknown-unknown` nicht bauen. Die Re-Exporte halten die
/// bisherigen Pfade `chaturaji_nnue::mcts` und `::outcome` gültig.
pub use chaturaji_engine::{mcts, outcome};
