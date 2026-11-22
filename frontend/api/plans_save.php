<?php
require __DIR__ . '/config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') send_json(['error' => 'POST required'], 405);

$in = json_input();
$email = strtolower(trim($in['email'] ?? ''));
$plan = $in['plan'] ?? null;

if (!filter_var($email, FILTER_VALIDATE_EMAIL) || !$plan || empty($plan['id'])) {
    send_json(['error' => 'email and plan (with id) are required'], 400);
}

$stmt = $pdo->prepare('SELECT id FROM users WHERE email = ?');
$stmt->execute([$email]);
$user = $stmt->fetch();
if (!$user) send_json(['error' => 'Unknown user'], 404);

$stmt = $pdo->prepare(
    'INSERT INTO plans (id, user_id, name, data, created_sort)
     VALUES (?, ?, ?, ?, ?)
     ON DUPLICATE KEY UPDATE name = VALUES(name), data = VALUES(data)'
);
$stmt->execute([
    $plan['id'],
    $user['id'],
    $plan['name'] ?? 'Untitled plan',
    json_encode($plan),
    $plan['createdSort'] ?? (int) round(microtime(true) * 1000),
]);

send_json(['ok' => true]);
