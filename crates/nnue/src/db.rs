//! SQLite-Persistenz für NNUE-Gewichte und Trainingsstatistiken.
//! Analog zu `crates/trainer/src/db.rs`.

use rusqlite::{params, Connection, Result};
use crate::network::NnueNetwork;

pub fn open(path: &str) -> Result<Connection> {
    let conn = Connection::open(path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;
    create_tables(&conn)?;
    Ok(conn)
}

fn create_tables(conn: &Connection) -> Result<()> {
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS network_weights (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            steps        INTEGER NOT NULL,
            lr           REAL    NOT NULL,
            avg_loss     REAL,
            weights_json TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_stats (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch        INTEGER NOT NULL,
            games        INTEGER NOT NULL,
            avg_loss     REAL    NOT NULL,
            avg_game_len REAL    NOT NULL,
            recorded_at  TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS games (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at     TEXT    NOT NULL DEFAULT (datetime('now')),
            moves         INTEGER NOT NULL,
            winner        TEXT,
            score_red     INTEGER NOT NULL DEFAULT 0,
            score_blue    INTEGER NOT NULL DEFAULT 0,
            score_yellow  INTEGER NOT NULL DEFAULT 0,
            score_green   INTEGER NOT NULL DEFAULT 0,
            pgn           TEXT    NOT NULL,
            network_steps INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_nnue_weights_steps ON network_weights(steps DESC);
        CREATE INDEX IF NOT EXISTS idx_nnue_games_played  ON games(played_at DESC);
    ")?;
    Ok(())
}

pub fn save_network(conn: &Connection, net: &NnueNetwork, avg_loss: Option<f32>) -> Result<i64> {
    let json = serde_json::to_string(net)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
    conn.execute(
        "INSERT INTO network_weights (steps, lr, avg_loss, weights_json)
         VALUES (?1, ?2, ?3, ?4)",
        params![net.steps as i64, net.lr as f64, avg_loss.map(|v| v as f64), json],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn load_latest_network(conn: &Connection) -> Result<Option<NnueNetwork>> {
    let result = conn.query_row(
        "SELECT weights_json FROM network_weights ORDER BY steps DESC LIMIT 1",
        [],
        |row| row.get::<_, String>(0),
    );
    match result {
        Ok(json) => {
            let mut net: NnueNetwork = serde_json::from_str(&json)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    0, rusqlite::types::Type::Text, Box::new(e),
                ))?;
            // Checkpoints von vor der Eingabe-Erweiterung auf die aktuelle
            // Breite bringen, statt sie abzulehnen.
            net.ensure_input_size();
            Ok(Some(net))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn save_stats(
    conn: &Connection, epoch: u32, games: u32, avg_loss: f32, avg_game_len: f32,
) -> Result<()> {
    conn.execute(
        "INSERT INTO training_stats (epoch, games, avg_loss, avg_game_len)
         VALUES (?1, ?2, ?3, ?4)",
        params![epoch, games, avg_loss as f64, avg_game_len as f64],
    )?;
    Ok(())
}

pub fn save_game(conn: &Connection, game: &GameRecord) -> Result<i64> {
    conn.execute(
        "INSERT INTO games
         (moves, winner, score_red, score_blue, score_yellow, score_green, pgn, network_steps)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            game.moves, game.winner.as_deref(),
            game.score_red, game.score_blue, game.score_yellow, game.score_green,
            game.pgn, game.network_steps as i64,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn game_count(conn: &Connection) -> Result<i64> {
    conn.query_row("SELECT COUNT(*) FROM games", [], |r| r.get(0))
}

pub fn list_checkpoints(conn: &Connection) -> Result<Vec<CheckpointInfo>> {
    let mut stmt = conn.prepare(
        "SELECT id, steps, lr, avg_loss, saved_at FROM network_weights ORDER BY steps DESC"
    )?;
    let rows = stmt.query_map([], |row| Ok(CheckpointInfo {
        id: row.get(0)?, steps: row.get(1)?, lr: row.get(2)?,
        avg_loss: row.get(3)?, saved_at: row.get(4)?,
    }))?;
    rows.collect()
}

pub fn load_stats(conn: &Connection, n: usize) -> Result<Vec<TrainingStat>> {
    let mut stmt = conn.prepare(
        "SELECT epoch, games, avg_loss, avg_game_len
         FROM training_stats ORDER BY epoch DESC LIMIT ?1"
    )?;
    let mut stats: Vec<_> = stmt.query_map(params![n as i64], |row| Ok(TrainingStat {
        epoch: row.get(0)?, games: row.get(1)?,
        avg_loss: row.get(2)?, avg_game_len: row.get(3)?,
    }))?.collect::<Result<_>>()?;
    stats.reverse();
    Ok(stats)
}

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

pub struct CheckpointInfo {
    pub id:       i64,
    pub steps:    i64,
    pub lr:       f64,
    pub avg_loss: Option<f64>,
    pub saved_at: String,
}

pub struct TrainingStat {
    pub epoch:        i64,
    pub games:        i64,
    pub avg_loss:     f64,
    pub avg_game_len: f64,
}
