<?php
/**
 * Speichert eine Aufzeichnung (Recording) aus der Chaturaji-Web-UI.
 *
 * Erwartet POST mit JSON-Body: { "filename": "<name>.json", "game": { … } }
 *   - Schreibt das Spiel-JSON nach  game_analysis/<filename>
 *   - Ergänzt/ersetzt einen Eintrag in  games_index.json
 * damit das Spiel in der App-Liste erscheint und wie jedes andere ladbar ist.
 *
 * Pendant zur lokalen Node-Route POST /save-game.php in server.js.
 */

header('Content-Type: application/json; charset=utf-8');

function fail(int $code, string $msg): void {
    http_response_code($code);
    echo json_encode(['ok' => false, 'error' => $msg], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    fail(405, 'Nur POST erlaubt');
}

$raw = file_get_contents('php://input');
if ($raw === false || $raw === '') {
    fail(400, 'Leerer Request-Body');
}

$payload = json_decode($raw, true);
if (!is_array($payload)) {
    fail(400, 'Ungültiges JSON');
}

$filename = $payload['filename'] ?? '';
$game     = $payload['game'] ?? null;
$baseFile = $payload['baseFile'] ?? null; // gesetzt, wenn ein bestehendes Spiel aufgezeichnet wurde

// Dateiname streng validieren (kein Path-Traversal, nur *.json)
if (!is_string($filename)
    || strpos($filename, '..') !== false
    || !preg_match('/^[\w.-]+\.json$/', $filename)) {
    fail(400, 'Ungültiger Dateiname');
}
$filename = basename($filename); // doppelt absichern
if (!is_array($game)) {
    fail(400, 'Ungültiges Spiel-Objekt');
}

$analysisDir = __DIR__ . '/game_analysis';
$indexFile   = __DIR__ . '/games_index.json';

if (!is_dir($analysisDir) && !@mkdir($analysisDir, 0775, true)) {
    fail(500, 'game_analysis/ nicht anlegbar');
}

// games_index.json laden (oder leer starten)
$index = [];
if (is_file($indexFile)) {
    $decoded = json_decode(@file_get_contents($indexFile), true);
    if (is_array($decoded)) {
        $index = $decoded;
    }
}

// Aufzeichnung eines bestehenden Spiels: Dateiname = alter Name (ohne evtl.
// bestehende "-<Nr>"-Endung) plus fortlaufende Nummer. games_index.json wird
// gelesen, um bereits gespeicherte Varianten zu überspringen.
if (is_string($baseFile)
    && strpos($baseFile, '..') === false
    && preg_match('/^[\w.-]+\.json$/', $baseFile)) {
    $root = preg_replace('/-\d+$/', '', preg_replace('/\.json$/i', '', basename($baseFile)));
    $re   = '/^' . preg_quote($root, '/') . '-(\d+)\.json$/i';
    $max  = 0;
    foreach ($index as $e) {
        if (isset($e['file']) && is_string($e['file']) && preg_match($re, $e['file'], $m)) {
            $max = max($max, (int) $m[1]);
        }
    }
    $filename = $root . '-' . ($max + 1) . '.json';
}

// Spiel-JSON schreiben
$gameJson = json_encode(
    $game,
    JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES
);
if (@file_put_contents($analysisDir . '/' . $filename, $gameJson) === false) {
    fail(500, 'Schreiben von ' . $filename . ' fehlgeschlagen (Rechte?)');
}

// Index-Eintrag im selben Format wie build_games_index.py
$entry = [
    'file'       => $filename,
    'gameNr'     => $game['gameNr'] ?? null,
    'date'       => gmdate('Y-m-d\TH:i:s.000\Z'),
    'result'     => $game['result'] ?? '',
    'players'    => [
        $game['username1'] ?? '', $game['username2'] ?? '',
        $game['username3'] ?? '', $game['username4'] ?? '',
    ],
    'ratings'    => [
        $game['rating1'] ?? null, $game['rating2'] ?? null,
        $game['rating3'] ?? null, $game['rating4'] ?? null,
    ],
    'placements' => [null, null, null, null],
];

// Bestehenden Eintrag mit gleichem Dateinamen ersetzen, sonst anhängen
$replaced = false;
foreach ($index as $i => $e) {
    if (isset($e['file']) && $e['file'] === $filename) {
        $index[$i] = $entry;
        $replaced = true;
        break;
    }
}
if (!$replaced) {
    $index[] = $entry;
}

if (@file_put_contents(
        $indexFile,
        json_encode($index, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES)
    ) === false) {
    fail(500, 'games_index.json nicht schreibbar (Rechte?)');
}

echo json_encode(['ok' => true, 'file' => $filename], JSON_UNESCAPED_UNICODE);
