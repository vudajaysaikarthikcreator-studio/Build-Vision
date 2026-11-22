<?php
require __DIR__ . '/config.php';

$email = strtolower(trim($_GET['email'] ?? ''));
if (!filter_var($email, FILTER_VALIDATE_EMAIL)) send_json(['error' => 'Valid email required'], 400);

$stmt = $pdo->prepare('SELECT id FROM users WHERE email = ?');
$stmt->execute([$email]);
$user = $stmt->fetch();
if (!$user) send_json(['plans' => []]);

$stmt = $pdo->prepare('SELECT data FROM plans WHERE user_id = ? ORDER BY created_sort ASC');
$stmt->execute([$user['id']]);
$rows = $stmt->fetchAll();

$plans = array_map(fn($r) => json_decode($r['data'], true), $rows);
send_json(['plans' => $plans]);
