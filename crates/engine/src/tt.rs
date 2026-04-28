//! Transposition table (TT) for the Max^n search.
//!
//! Each entry stores:
//!   • Zobrist hash (for collision detection)
//!   • Best move found
//!   • Score vector
//!   • Search depth at which the entry was stored
//!   • Node type: Exact / LowerBound / UpperBound

use chaturaji_core::board::Move;

/// Node type (standard alpha-beta terminology).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Exact,
    LowerBound,  // score is at least this good (fail-high)
    UpperBound,  // score is at most this good (fail-low)
}

/// A single TT entry.
#[derive(Clone, Copy)]
pub struct TtEntry {
    pub hash:   u64,
    pub depth:  u8,
    pub kind:   NodeKind,
    pub scores: [i32; 4],
    pub best:   Option<Move>,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            hash:   0,
            depth:  0,
            kind:   NodeKind::Exact,
            scores: [0; 4],
            best:   None,
        }
    }
}

/// Fixed-size transposition table (power-of-two buckets, always-replace).
pub struct TranspositionTable {
    entries: Vec<TtEntry>,
    mask:    usize,
}

impl TranspositionTable {
    /// Create a TT with `size_mb` megabytes of storage.
    pub fn new(size_mb: usize) -> Self {
        let bytes  = size_mb * 1024 * 1024;
        let entry_size = std::mem::size_of::<TtEntry>();
        // Round down to a power of two
        let mut capacity = bytes / entry_size;
        capacity = capacity.next_power_of_two() >> 1;
        if capacity == 0 { capacity = 1; }

        Self {
            entries: vec![TtEntry::default(); capacity],
            mask:    capacity - 1,
        }
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Store an entry (always-replace policy).
    pub fn store(&mut self, hash: u64, depth: u8, kind: NodeKind, scores: [i32; 4], best: Option<Move>) {
        let idx = self.index(hash);
        self.entries[idx] = TtEntry { hash, depth, kind, scores, best };
    }

    /// Probe the table.  Returns `None` on miss or hash collision.
    pub fn probe(&self, hash: u64) -> Option<&TtEntry> {
        let entry = &self.entries[self.index(hash)];
        if entry.hash == hash && entry.depth > 0 {
            Some(entry)
        } else {
            None
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.iter_mut().for_each(|e| *e = TtEntry::default());
    }
}
