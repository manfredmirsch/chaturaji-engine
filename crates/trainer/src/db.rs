//! SQLite-Datenbank für:
//!   • Netzgewichte (als JSON-Blob, versioniert)
//!   • Trainingsstatistiken (Verlust pro Epoche)
//!   • Gespielte Self-Play-Partien (PGN + Endstand)

use rusqlite::{params, Connection, Result};
use crate::network::Network;

/// Öffnet (oder erstellt) die Datenbank und legt alle Tabellen an.
pub fn open(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    create_tables(&conn)?;
    Ok(conn)
}

fn create_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch("
        -- Netzgewichte (eine Zeile pro gespeichertem Checkpoint)
        CREATE TABLE IF NOT EXISTS network_weights (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at    TEXT    NOT NULL DEFAULT (datetime('now')),
            steps       INTEGER NOT NULL,
            lr          REAL    NOT NULL,
            avg_loss    REAL,
            weights_json TEXT   NOT NULL   -- serialisierte Gewichte als JSON
        );

        -- Trainingsstatistiken pro Epoche
        CREATE TABLE IF NOT EXISTS training_stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch       INTEGER NOT NULL,
            games       INTEGER NOT NULL,
            avg_loss    REAL    NOT NULL,
            avg_game_len REAL   NOT NULL,
            recorded_at TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- Gespielte Self-Play-Partien
        CREATE TABLE IF NOT EXISTS games (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            moves       INTEGER NOT NULL,
            winner      TEXT,                   -- 'Red','Blue','Yellow','Green' oder NULL
            score_red   INTEGER NOT NULL DEFAULT 0,
            score_blue  INTEGER NOT NULL DEFAULT 0,
            score_yellow INTEGER NOT NULL DEFAULT 0,
            score_green  INTEGER NOT NULL DEFAULT 0,
            pgn         TEXT    NOT NULL,
            network_steps INTEGER NOT NULL DEFAULT 0
        );

        -- Index für schnellen Zugriff auf letzte Gewichte
        CREATE INDEX IF NOT EXISTS idx_weights_steps ON network_weights(steps DESC);
        CREATE INDEX IF NOT EXISTS idx_games_played  ON games(played_at DESC);
    ")?;
    Ok(())
}

// ── Netzgewichte ──────────────────────────────────────────────────────────────

/// Speichert den aktuellen Netz-Checkpoint in der DB.
pub fn save_network(conn: &Connection, net: &Network, avg_loss: Option<f32>) -> Result<i64> {
    let json = serde_json::to_string(net)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

    conn.execute(
        "INSERT INTO network_weights (steps, lr, avg_loss, weights_json)
         VALUES (?1, ?2, ?3, ?4)",
        params![net.steps as i64, net.lr as f64, avg_loss.map(|v| v as f64), json],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Lädt die neuesten Netzgewichte aus der DB.
/// Gibt `None` zurück wenn noch kein Checkpoint existiert.
pub fn load_latest_network(conn: &Connection) -> Result<Option<Network>> {
    let result = conn.query_row(
        "SELECT weights_json FROM network_weights ORDER BY steps DESC LIMIT 1",
        [],
        |row| row.get::<_, String>(0),
    );

    match result {
        Ok(json) => {
            let net: Network = serde_json::from_str(&json)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    0, rusqlite::types::Type::Text, Box::new(e)
                ))?;
            Ok(Some(net))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Gibt alle gespeicherten Checkpoints zurück (id, steps, avg_loss, saved_at).
pub fn list_checkpoints(conn: &Connection) -> Result<Vec<CheckpointInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, steps, lr, avg_loss, saved_at FROM network_weights ORDER BY steps DESC"
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(CheckpointInfo {
            id:       row.get(0)?,
            steps:    row.get(1)?,
            lr:       row.get(2)?,
            avg_loss: row.get(3)?,
            saved_at: row.get(4)?,
        })
    })?;
    rows.collect()
}

/// Lädt einen spezifischen Checkpoint anhand der ID.
pub fn load_network_by_id(conn: &Connection, id: i64) -> Result<Network> {
    let json: String = conn.query_row(
        "SELECT weights_json FROM network_weights WHERE id = ?1",
        params![id],
        |row| row.get(0),
    )?;
    serde_json::from_str(&json)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
            0, rusqlite::types::Type::Text, Box::new(e)
        ))
}

// ── Trainingsstatistiken ──────────────────────────────────────────────────────

/// Speichert Statistiken einer Trainingsepoche.
pub fn save_stats(
    conn: &Connection,
    epoch: u32,
    games: u32,
    avg_loss: f32,
    avg_game_len: f32,
) -> Result<()> {
    conn.execute(
        "INSERT INTO training_stats (epoch, games, avg_loss, avg_game_len)
         VALUES (?1, ?2, ?3, ?4)",
        params![epoch, games, avg_loss as f64, avg_game_len as f64],
    )?;
    Ok(())
}

/// Gibt die letzten `n` Trainingsstatistiken zurück.
pub fn load_stats(conn: &Connection, n: usize) -> Result<Vec<TrainingStat>> {
    let mut stmt = conn.prepare(
        "SELECT epoch, games, avg_loss, avg_game_len, recorded_at
         FROM training_stats ORDER BY epoch DESC LIMIT ?1"
    )?;
    let rows = stmt.query_map(params![n as i64], |row| {
        Ok(TrainingStat {
            epoch:        row.get(0)?,
            games:        row.get(1)?,
            avg_loss:     row.get(2)?,
            avg_game_len: row.get(3)?,
            recorded_at:  row.get(4)?,
        })
    })?;
    let mut stats: Vec<_> = rows.collect::<Result<_>>()?;
    stats.reverse();
    Ok(stats)
}

// ── Partien ───────────────────────────────────────────────────────────────────

/// Speichert eine gespielte Partie.
pub fn save_game(conn: &Connection, game: &GameRecord) -> Result<i64> {
    conn.execute(
        "INSERT INTO games
         (moves, winner, score_red, score_blue, score_yellow, score_green, pgn, network_steps)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            game.moves,
            game.winner.as_deref(),
            game.score_red,
            game.score_blue,
            game.score_yellow,
            game.score_green,
            game.pgn,
            game.network_steps as i64,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Gibt die letzten `n` Partien zurück.
pub fn load_recent_games(conn: &Connection, n: usize) -> Result<Vec<GameRecord>> {
    let mut stmt = conn.prepare(
        "SELECT moves, winner, score_red, score_blue, score_yellow, score_green, pgn, network_steps
         FROM games ORDER BY played_at DESC LIMIT ?1"
    )?;
    let rows = stmt.query_map(params![n as i64], |row| {
        Ok(GameRecord {
            moves:         row.get(0)?,
            winner:        row.get(1)?,
            score_red:     row.get(2)?,
            score_blue:    row.get(3)?,
            score_yellow:  row.get(4)?,
            score_green:   row.get(5)?,
            pgn:           row.get(6)?,
            network_steps: row.get::<_, i64>(7)? as u64,
        })
    })?;
    rows.collect()
}

/// Gesamtzahl gespeicherter Partien.
pub fn game_count(conn: &Connection) -> Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM games", [], |r| r.get(0))
}

// ── Datentypen ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct CheckpointInfo {
    pub id:       i64,
    pub steps:    i64,
    pub lr:       f64,
    pub avg_loss: Option<f64>,
    pub saved_at: String,
}

#[derive(Debug)]
pub struct TrainingStat {
    pub epoch:        i64,
    pub games:        i64,
    pub avg_loss:     f64,
    pub avg_game_len: f64,
    pub recorded_at:  String,
}

#[derive(Debug)]
pub struct GameRecord {
    pub moves:         i32,
    pub winner:        Option<String>,
    pub score_red:     i32,
    pub score_blue:    i32,
    pub score_yellow:  i32,
    pub score_green:   i32,
    pub pgn:           String,
    pub network_steps: u64,
}
