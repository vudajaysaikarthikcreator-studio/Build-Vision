<?php
require __DIR__ . '/config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') send_json(['error' => 'POST required'], 405);

$in = json_input();
$name = trim($in['name'] ?? '');
$email = strtolower(trim($in['email'] ?? ''));
$password = $in['password'] ?? '';

if ($name === '' || !filter_var($email, FILTER_VALIDATE_EMAIL) || strlen($password) < 6) {
    send_json(['error' => 'Please fill every field correctly (passwords must match, 6+ chars).'], 400);
}

$stmt = $pdo->prepare('SELECT id FROM users WHERE email = ?');
$stmt->execute([$email]);
if ($stmt->fetch()) {
    send_json(['error' => 'An account with that email already exists. Please log in instead.'], 409);
}

$hash = password_hash($password, PASSWORD_BCRYPT);
$stmt = $pdo->prepare('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)');
$stmt->execute([$name, $email, $hash]);

send_json(['user' => ['name' => $name, 'email' => $email]]);
