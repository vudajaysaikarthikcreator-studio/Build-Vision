"""SQLAlchemy models for BuildVision.

The same MySQL database is shared by the PHP frontend (users/plans) and the
FastAPI AI pipeline tables. create_all() only adds missing tables/columns; it
does not remove existing data.
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, Text, DateTime, BigInteger
from sqlalchemy.sql import func
from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    email = Column(String(190), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class Plan(Base):
    __tablename__ = "plans"

    id = Column(String(64), primary_key=True)
    user_id = Column(Integer, primary_key=True, nullable=False, index=True)
    name = Column(String(190), nullable=False)
    data = Column(Text, nullable=False)
    created_sort = Column(BigInteger, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())


class HouseRequirement(Base):
    __tablename__ = "house_requirements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), nullable=False, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    plot_length = Column(Float, nullable=False)
    plot_width = Column(Float, nullable=False)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Integer, nullable=False)
    kitchens = Column(Integer, nullable=False)
    floors = Column(Integer, nullable=False)
    budget = Column(Float, nullable=False)
    road_direction = Column(String(16))
    main_entrance_direction = Column(String(16))
    vastu_preference = Column(Boolean, default=True)
    construction_category = Column(String(16))
    raw_json = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class VastuArrangement(Base):
    __tablename__ = "vastu_arrangements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), nullable=False, index=True)
    room_type = Column(String(64))
    label = Column(String(64))
    preferred_zone = Column(String(8))
    assigned_zone = Column(String(8))
    matched = Column(Boolean, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), nullable=False, index=True)
    room_type = Column(String(64))
    label = Column(String(64))
    zone = Column(String(8))
    x = Column(Float)
    y = Column(Float)
    w = Column(Float)
    h = Column(Float)
    door_side = Column(String(8))
    created_at = Column(DateTime, server_default=func.now())


class Layout(Base):
    __tablename__ = "layouts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), nullable=False, index=True, unique=True)
    plot_length = Column(Float)
    plot_width = Column(Float)
    floors = Column(Integer)
    planning_type = Column(String(32))
    collisions = Column(Integer, default=0)
    boundary_violations = Column(Integer, default=0)
    passed_checks = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())


class Blueprint(Base):
    __tablename__ = "blueprints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), nullable=False, index=True, unique=True)
    image_path = Column(String(255))
    entrance_direction = Column(String(16))
    created_at = Column(DateTime, server_default=func.now())


class CostEstimation(Base):
    __tablename__ = "cost_estimations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(String(64), nullable=False, index=True, unique=True)
    plot_area = Column(Float)
    builtup_area = Column(Float)
    rate_per_sqft = Column(Float)
    material_cost = Column(Float)
    labour_cost = Column(Float)
    total_cost = Column(Float)
    model_predicted_total = Column(Float, nullable=True)
    model_predicted_total_challenger = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
