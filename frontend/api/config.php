<?php
/**
 * Database connection config for XAMPP's default MySQL setup.
 * Default XAMPP MySQL: host=localhost, user=root, password="" (empty).
 * If you set a root password in XAMPP, put it below.
 */

$DB_HOST = 'localhost';
$DB_NAME = 'blueprint_db';
$DB_USER = 'root';
$DB_PASS = '';

header('Content-Type: application/json');
// Allow the front end to be opened from any local origin during development.
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

function json_input() {
    $raw = file_get_contents('php://input');
    $data = json_decode($raw, true);
    return is_array($data) ? $data : [];
}

function send_json($payload, $status = 200) {
    http_response_code($status);
    echo json_encode($payload);
    exit;
}

try {
    $pdo = new PDO(
        "mysql:host=$DB_HOST;dbname=$DB_NAME;charset=utf8mb4",
        $DB_USER,
        $DB_PASS,
        [
            PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        ]
    );
} catch (PDOException $e) {
    send_json(['error' => 'Database connection failed. Is MySQL running in XAMPP, and did you import schema.sql?'], 500);
}
