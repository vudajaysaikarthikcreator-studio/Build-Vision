<?php
require __DIR__ . '/config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') send_json(['error' => 'POST required'], 405);

$in = json_input();
$email = strtolower(trim($in['email'] ?? ''));
$password = $in['password'] ?? '';

if (!filter_var($email, FILTER_VALIDATE_EMAIL) || strlen($password) < 6) {
    send_json(['error' => 'Enter a valid email and password (min 6 characters).'], 400);
}

$stmt = $pdo->prepare('SELECT name, email, password_hash FROM users WHERE email = ?');
$stmt->execute([$email]);
$user = $stmt->fetch();

if (!$user) {
    send_json(['error' => 'No account found for that email. Please register first.'], 404);
}
if (!password_verify($password, $user['password_hash'])) {
    send_json(['error' => 'Incorrect password. Please try again.'], 401);
}

send_json(['user' => ['name' => $user['name'], 'email' => $user['email']]]);
