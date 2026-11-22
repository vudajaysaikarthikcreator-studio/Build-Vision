<?php
require __DIR__ . '/config.php';

if ($_SERVER['REQUEST_METHOD'] !== 'POST') send_json(['error' => 'POST required'], 405);

$in = json_input();
$email = strtolower(trim($in['email'] ?? ''));
$id = $in['id'] ?? '';

if (!filter_var($email, FILTER_VALIDATE_EMAIL) || $id === '') {
    send_json(['error' => 'email and id are required'], 400);
}

$stmt = $pdo->prepare('SELECT id FROM users WHERE email = ?');
$stmt->execute([$email]);
$user = $stmt->fetch();
if (!$user) send_json(['ok' => true]); // nothing to delete

$stmt = $pdo->prepare('DELETE FROM plans WHERE id = ? AND user_id = ?');
$stmt->execute([$id, $user['id']]);

send_json(['ok' => true]);
