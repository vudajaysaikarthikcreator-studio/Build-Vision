-- BuildVision AI Hybrid - MySQL/XAMPP database
-- Import this file in phpMyAdmin or run it with mysql.
-- It contains BOTH the frontend tables (users, plans) and the FastAPI AI
-- pipeline tables.

CREATE DATABASE IF NOT EXISTS blueprint_db
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE blueprint_db;

CREATE TABLE IF NOT EXISTS users (
  id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(120) NOT NULL,
  email VARCHAR(190) NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS plans (
  id VARCHAR(64) NOT NULL,
  user_id INT UNSIGNED NOT NULL,
  name VARCHAR(190) NOT NULL,
  data LONGTEXT NOT NULL,
  created_sort BIGINT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id, user_id),
  KEY idx_plans_user (user_id),
  CONSTRAINT fk_plans_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS house_requirements (
  id INT AUTO_INCREMENT PRIMARY KEY,
  plan_id VARCHAR(64) NOT NULL,
  user_id INT NULL,
  plot_length DOUBLE NOT NULL, plot_width DOUBLE NOT NULL,
  bedrooms INT NOT NULL, bathrooms INT NOT NULL, kitchens INT NOT NULL,
  floors INT NOT NULL, budget DOUBLE NOT NULL,
  road_direction VARCHAR(16), main_entrance_direction VARCHAR(16),
  vastu_preference TINYINT(1) DEFAULT 1,
  construction_category VARCHAR(16), raw_json LONGTEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  KEY idx_house_plan (plan_id), KEY idx_house_user (user_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS vastu_arrangements (
  id INT AUTO_INCREMENT PRIMARY KEY, plan_id VARCHAR(64) NOT NULL,
  room_type VARCHAR(64), label VARCHAR(64), preferred_zone VARCHAR(8),
  assigned_zone VARCHAR(8), matched TINYINT(1),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP, KEY idx_vastu_plan (plan_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS rooms (
  id INT AUTO_INCREMENT PRIMARY KEY, plan_id VARCHAR(64) NOT NULL,
  room_type VARCHAR(64), label VARCHAR(64), zone VARCHAR(8),
  x DOUBLE, y DOUBLE, w DOUBLE, h DOUBLE, door_side VARCHAR(8),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP, KEY idx_rooms_plan (plan_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS layouts (
  id INT AUTO_INCREMENT PRIMARY KEY, plan_id VARCHAR(64) NOT NULL UNIQUE,
  plot_length DOUBLE, plot_width DOUBLE, floors INT, planning_type VARCHAR(32),
  collisions INT DEFAULT 0, boundary_violations INT DEFAULT 0, passed_checks TINYINT(1) DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS blueprints (
  id INT AUTO_INCREMENT PRIMARY KEY, plan_id VARCHAR(64) NOT NULL UNIQUE,
  image_path VARCHAR(255), entrance_direction VARCHAR(16),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS cost_estimations (
  id INT AUTO_INCREMENT PRIMARY KEY, plan_id VARCHAR(64) NOT NULL UNIQUE,
  plot_area DOUBLE, builtup_area DOUBLE, rate_per_sqft DOUBLE,
  material_cost DOUBLE, labour_cost DOUBLE, total_cost DOUBLE,
  model_predicted_total DOUBLE, model_predicted_total_challenger DOUBLE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;
