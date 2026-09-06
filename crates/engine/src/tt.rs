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
    pub age:    u8,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            hash:   0,
            depth:  0,
            kind:   NodeKind::Exact,
            scores: [0; 4],
            best:   None,
            age:    0,
        }
    }
}

/// Fixed-size transposition table with depth-preferred + age-based replacement.
///
/// Replacement policy: a new entry at `depth` replaces the existing one when:
///   • `existing.age != current_age` — entry is from a previous search (stale), OR
///   • `new_depth >= existing.depth` — new search is at least as deep.
///
/// This prevents shallow entries from the current search from overwriting deep
/// entries, while still replacing stale entries from earlier positions freely.
pub struct TranspositionTable {
    entries:     Vec<TtEntry>,
    mask:        usize,
    current_age: u8,
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
            entries:     vec![TtEntry::default(); capacity],
            mask:        capacity - 1,
            current_age: 0,
        }
    }

    /// Advance the search age counter. Call once at the start of every
    /// `Engine::search()` so entries from previous positions are considered
    /// stale and replaced freely.
    pub fn new_search(&mut self) {
        self.current_age = self.current_age.wrapping_add(1);
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    /// Store an entry using depth-preferred + age-based replacement.
    pub fn store(&mut self, hash: u64, depth: u8, kind: NodeKind, scores: [i32; 4], best: Option<Move>) {
        let idx      = self.index(hash);
        let existing = &self.entries[idx];
        let stale    = existing.age != self.current_age;
        if stale || depth >= existing.depth {
            self.entries[idx] = TtEntry { hash, depth, kind, scores, best, age: self.current_age };
        }
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

    /// Clear all entries and reset age.
    pub fn clear(&mut self) {
        self.entries.iter_mut().for_each(|e| *e = TtEntry::default());
        self.current_age = 0;
    }
}
