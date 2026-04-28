//! Zobrist hashing for the transposition table.
//!
//! Keys are generated at startup with a deterministic LCG so the engine
//! produces reproducible results without needing `rand`.

use crate::piece::{Color, PieceKind};

/// Pre-computed Zobrist keys.
pub struct ZobristKeys {
    /// `piece[color][kind][square]`
    pub piece: [[[u64; 64]; 5]; 4],
    /// Hash contribution for whose turn it is.
    pub side: [u64; 4],
    /// Hash contribution for each player's active status.
    pub active: [u64; 4],
}

impl ZobristKeys {
    /// Generate keys deterministically using a simple LCG.
    pub fn new() -> Self {
        let mut rng = Lcg::new(0xDEAD_BEEF_CAFE_1234);

        let mut piece = [[[0u64; 64]; 5]; 4];
        for c in 0..4 {
            for k in 0..5 {
                for s in 0..64 {
                    piece[c][k][s] = rng.next();
                }
            }
        }
        let mut side   = [0u64; 4];
        let mut active = [0u64; 4];
        for i in 0..4 {
            side[i]   = rng.next();
            active[i] = rng.next();
        }

        Self { piece, side, active }
    }
}

impl Default for ZobristKeys {
    fn default() -> Self { Self::new() }
}

/// Compute the Zobrist hash for a full board position.
pub fn hash_board(board: &crate::board::Board, keys: &ZobristKeys) -> u64 {
    let mut h: u64 = 0;

    for c in Color::ALL {
        let ci = c.idx();
        for k in PieceKind::ALL {
            let ki = k.idx();
            let mut bb = board.bb[ci][ki];
            while bb != 0 {
                let sq = bb.trailing_zeros() as usize;
                bb &= bb - 1;
                h ^= keys.piece[ci][ki][sq];
            }
        }
        if !board.active[ci] {
            h ^= keys.active[ci];
        }
    }

    h ^= keys.side[board.to_move.idx()];
    h
}

// ─── Deterministic LCG ────────────────────────────────────────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self { Lcg(seed) }
    fn next(&mut self) -> u64 {
        // Numerical Recipes parameters
        self.0 = self.0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn different_positions_different_hash() {
        let keys = ZobristKeys::new();
        let b1 = Board::default();
        let b2 = Board::empty();
        assert_ne!(hash_board(&b1, &keys), hash_board(&b2, &keys));
    }

    #[test]
    fn same_position_same_hash() {
        let keys = ZobristKeys::new();
        let b = Board::default();
        assert_eq!(hash_board(&b, &keys), hash_board(&b, &keys));
    }
}
