# Blueprint — running on a real database with XAMPP

This turns the app's storage from the in-browser demo store into a real
**MySQL database**, served through a small **PHP API**, both running under
**XAMPP** (Apache + MySQL + PHP).

## What's in this folder

```
blueprint-app/
├── index.html              ← the app (now talks to the PHP API instead of window.storage)
├── api/
│   ├── config.php           ← MySQL connection settings
│   ├── register.php         ← POST: create account
│   ├── login.php             ← POST: verify login
│   ├── plans_list.php        ← GET:  list a user's saved plans
│   ├── plans_save.php        ← POST: create/update a plan
│   └── plans_delete.php      ← POST: delete a plan
└── database/
    └── schema.sql            ← run this once to create the database + tables
```

## 1. Install & start XAMPP

Download XAMPP from https://www.apachefriends.org if you don't have it, install it,
open the **XAMPP Control Panel**, and click **Start** next to both **Apache** and
**MySQL**.

## 2. Create the database

1. Click **Admin** next to MySQL in the Control Panel (or go to
   `http://localhost/phpmyadmin`).
2. Go to the **Import** tab.
3. Choose the file `database/schema.sql` from this folder.
4. Click **Go**.

This creates a database named `blueprint_db` with two tables: `users` and `plans`.

*(Default XAMPP MySQL has no root password. If you've set one, open
`api/config.php` and put it in `$DB_PASS`.)*

## 3. Put the app in htdocs

Copy this whole `blueprint-app` folder into XAMPP's `htdocs` directory:

- **Windows:** `C:\xampp\htdocs\blueprint-app`
- **macOS:** `/Applications/XAMPP/htdocs/blueprint-app`
- **Linux:** `/opt/lampp/htdocs/blueprint-app`

## 4. Open it in your browser

Go to:

```
http://localhost/blueprint-app/index.html
```

**Important:** you must open it through this `http://localhost/...` URL — not by
double-clicking `index.html` — because the PHP API only runs when Apache serves the
page (a `file://` page can't call PHP scripts).

## 5. Use it

- Register an account — this inserts a row into `users`, with the password hashed
  server-side (bcrypt via PHP's `password_hash()`), never stored in plain text.
- Log in.
- Generate a plan and click **Save** — this inserts/updates a row in `plans`.
- Reload the page, log back in — your plans are still there, loaded from MySQL.
- You can inspect the data any time in phpMyAdmin: `blueprint_db` → `users` /
  `plans` tables.

## How the pieces fit together

- **Apache** serves `index.html` and runs the PHP scripts in `api/`.
- **PHP** (`api/*.php`) talks to MySQL using PDO with prepared statements (safe
  against SQL injection), and returns JSON.
- **MySQL** (`blueprint_db`) is the actual persistent database — data survives
  restarts, and multiple users can each have their own accounts and plans.
- **The front end** (`index.html`) now calls `fetch('api/....php', ...)` instead
  of the old `window.storage` calls — everything else about the app (the planner,
  PDF export, cost estimator) is unchanged.

## Troubleshooting

- **"Database connection failed"** → Make sure MySQL is running in XAMPP and that
  you imported `schema.sql`.
- **Nothing happens on Save/Login, console shows a fetch/CORS error** → You likely
  opened the file directly (`file://...`) instead of through
  `http://localhost/blueprint-app/index.html`.
- **Want a different DB user/password?** → Edit `api/config.php`.
