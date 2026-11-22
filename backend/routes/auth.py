from __future__ import annotations
import hashlib, hmac, os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.schemas import RegisterIn, LoginIn

router = APIRouter(tags=["auth"])


def _hash_password(password: str) -> str:
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except Exception:
        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 180_000)
        return "pbkdf2$" + salt.hex() + "$" + digest.hex()


def _verify_password(password: str, stored: str) -> bool:
    if stored.startswith("$2"):
        try:
            import bcrypt
            return bool(bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8")))
        except Exception:
            return False
    if stored.startswith("pbkdf2$"):
        try:
            _, salt_hex, digest_hex = stored.split("$", 2)
            digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 180_000)
            return hmac.compare_digest(digest.hex(), digest_hex)
        except ValueError:
            return False
    return False


def _register(req: RegisterIn, db: Session):
    password = req.password or ""
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    email = str(req.email).strip().lower()
    try:
        existing = db.execute(text("SELECT id FROM users WHERE email = :email"), {"email": email}).fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    if existing:
        raise HTTPException(status_code=409, detail="Email is already registered.")
    pw_hash = _hash_password(password)
    try:
        result = db.execute(
            text("INSERT INTO users (name, email, password_hash) VALUES (:name, :email, :hash)"),
            {"name": req.name.strip(), "email": email, "hash": pw_hash},
        )
        db.commit()
        new_id = getattr(result, "lastrowid", None)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"Could not save user: {exc}")
    return {"user": {"id": new_id, "name": req.name.strip(), "email": email}, "ok": True, "user_id": new_id, "name": req.name.strip(), "email": email}


def _login(req: LoginIn, db: Session):
    email = str(req.email).strip().lower()
    try:
        row = db.execute(text("SELECT id, name, password_hash FROM users WHERE email = :email"), {"email": email}).fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}")
    if not row or not _verify_password(req.password, row.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"user": {"id": row.id, "name": row.name, "email": email}, "ok": True,
            "user_id": row.id, "name": row.name, "email": email}


@router.post("/register")
@router.post("/auth/register")
def register(req: RegisterIn, db: Session = Depends(get_db)):
    return _register(req, db)


@router.post("/login")
@router.post("/auth/login")
def login(req: LoginIn, db: Session = Depends(get_db)):
    return _login(req, db)
