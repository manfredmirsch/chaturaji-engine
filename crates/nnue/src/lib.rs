pub mod db;
pub mod dist;
pub mod eval;
pub mod gen_train;
pub mod pgn_import;
pub mod selfplay;
pub mod supervised;
pub mod td;

/// Beide Module sind in die Engine gewandert, damit das WASM-Frontend sie
/// benutzen kann — dieser Kiste hängen `rusqlite` (gebündeltes C) und `rayon`
/// an, die für `wasm32-unknown-unknown` nicht bauen. Die Re-Exporte halten die
/// bisherigen Pfade `chaturaji_nnue::mcts` und `::outcome` gültig.
pub use chaturaji_engine::{mcts, outcome};

/// Merkmalssatz und Netz liegen ebenfalls in der Engine — aus demselben Grund
/// und seit demselben Anlass: das WASM-Frontend hatte eine **eigene** Kopie des
/// Merkmalsextraktors und ein eigenes Netzformat. Die Kopien liefen
/// auseinander, und der Fehler blieb unbemerkt, weil beide Seiten für sich
/// stimmig aussahen. Jetzt gibt es einen Extraktor und einen Forward-Pass für
/// Trainer, Arena und Browser.
pub use chaturaji_engine::{nnue_features as features, nnue_network as network};
