"""BuildVision database configuration and automatic MySQL database setup.

The project is designed for XAMPP MySQL (database: blueprint_db), but can be
forced to SQLite for a quick local test with DB_DRIVER=sqlite. When MySQL is
used, the backend first creates the database if it does not exist and then
SQLAlchemy creates any missing tables.
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "blueprint_db")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DRIVER = os.getenv("DB_DRIVER", "mysql").lower().strip()

Base = declarative_base()

if DB_DRIVER == "sqlite":
    DATABASE_URL = f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'buildvision_local.db'))}"
    DATABASE_SAFE_URL = DATABASE_URL
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, future=True)
else:
    from sqlalchemy.engine import URL

    _url = URL.create(
        "mysql+pymysql",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=DB_NAME,
        query={"charset": "utf8mb4"},
    )
    DATABASE_URL = _url
    DATABASE_SAFE_URL = f"mysql+pymysql://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(
        _url,
        pool_pre_ping=True,
        future=True,
        connect_args={"connect_timeout": 5},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def ensure_mysql_database() -> None:
    """Create blueprint_db when MySQL is running and the database is missing."""
    if DB_DRIVER == "sqlite":
        return

    from sqlalchemy.engine import URL

    server_url = URL.create(
        "mysql+pymysql",
        username=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        query={"charset": "utf8mb4"},
    )
    server_engine = create_engine(
        server_url,
        pool_pre_ping=True,
        future=True,
        connect_args={"connect_timeout": 5},
    )
    try:
        with server_engine.connect() as conn:
            # Identifier is controlled by the .env/config value; backticks keep
            # normal database names safe for MySQL.
            safe_name = DB_NAME.replace("`", "``")
            conn.exec_driver_sql(
                f"CREATE DATABASE IF NOT EXISTS `{safe_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        server_engine.dispose()


def initialize_database() -> None:
    """Create the database (MySQL) and all SQLAlchemy-owned tables."""
    ensure_mysql_database()
    # Import only after Base exists so the ORM classes register themselves.
    from backend import models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
