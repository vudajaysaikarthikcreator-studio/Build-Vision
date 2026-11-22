"""BuildVision ML prediction layer.

Primary models follow the project design: Decision Tree for planning type and
Random Forest for cost. The models train lazily from the included 999-record
CSV datasets. Gradient Boosting challengers are also exposed. If sklearn is
unavailable, deterministic formula fallbacks keep the backend operational.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import csv, math
from typing import Tuple
try:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier
except Exception:
    np=None
    GradientBoostingClassifier=GradientBoostingRegressor=RandomForestRegressor=DecisionTreeClassifier=None

BASE=Path(__file__).resolve().parent
PLANNING_LABELS=("Compact","Balanced","Spacious","Luxury")
CATEGORY_RATE={"basic":1350.0,"standard":1800.0,"premium":2450.0}

def _planning_rule(area,bedrooms,floors,budget):
    score=area/1800.0+bedrooms*0.35+floors*0.15+budget/3_000_000.0
    if score<2.0:return 'Compact'
    if score<3.2:return 'Balanced'
    if score<4.6:return 'Spacious'
    return 'Luxury'

def _load_csv(name):
    path=BASE/'datasets'/name
    if not path.exists(): return []
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

@lru_cache(maxsize=1)
def _planning_models():
    if DecisionTreeClassifier is None or np is None: return None
    rows=[]
    for name in ('requirement_planning_urban_999.csv','requirement_planning_mixed_999.csv','requirement_planning_rural_999.csv',
                 'requirement_planning_urban_2500.csv','requirement_planning_mixed_2500.csv','requirement_planning_rural_2500.csv'):
        rows.extend(_load_csv(name))
    if not rows: return None
    X=np.asarray([[float(r['plot_area']),float(r['plot_length']),float(r['plot_width']),float(r['bedrooms']),float(r['bathrooms']),float(r['floors']),float(r['budget']),float(r['parking_required']),float(r['family_size'])] for r in rows],dtype=float)
    y=np.asarray([r['planning_type'] for r in rows])
    tree=DecisionTreeClassifier(max_depth=10,min_samples_leaf=3,class_weight='balanced',random_state=42).fit(X,y)
    challenger=GradientBoostingClassifier(random_state=42,n_estimators=120,max_depth=3,learning_rate=0.05).fit(X,y)
    return tree,challenger

def _planning_predict(features,challenger=False):
    models=_planning_models()
    if models is None:return _planning_rule(features[0],features[3],features[5],features[6]),0.82
    model=models[1 if challenger else 0]; row=np.asarray([features],dtype=float)
    label=str(model.predict(row)[0]); confidence=float(model.predict_proba(row).max())
    return label,round(confidence,3)

def predict_planning_type(*,plot_area,plot_length,plot_width,bedrooms,bathrooms,floors,budget,parking_required,family_size)->Tuple[str,float]:
    return _planning_predict([plot_area,plot_length,plot_width,bedrooms,bathrooms,floors,budget,int(bool(parking_required)),family_size],False)

def predict_planning_type_ensemble(*,plot_area,plot_length,plot_width,bedrooms,bathrooms,floors,budget,parking_required,family_size)->Tuple[str,float]:
    return _planning_predict([plot_area,plot_length,plot_width,bedrooms,bathrooms,floors,budget,int(bool(parking_required)),family_size],True)

@lru_cache(maxsize=1)
def _cost_models():
    if RandomForestRegressor is None or np is None:return None
    rows=[]
    for name in ('cost_estimation_construction_999.csv','cost_estimation_city_adjusted_999.csv','cost_estimation_budget_mixed_999.csv',
                 'cost_estimation_construction_2500.csv','cost_estimation_city_adjusted_2500.csv','cost_estimation_budget_mixed_2500.csv'):
        rows.extend(_load_csv(name))
    if not rows:return None
    idx={'basic':0,'standard':1,'premium':2}
    X=np.asarray([[float(r['plot_area']),float(r['builtup_area']),float(r['floors']),float(r['num_rooms']),float(r['num_bathrooms']),idx.get(r['category'],1)] for r in rows],dtype=float)
    y=np.asarray([float(r['cost']) for r in rows],dtype=float)
    primary=RandomForestRegressor(n_estimators=180,max_depth=16,min_samples_leaf=2,random_state=42,n_jobs=-1).fit(X,y)
    challenger=GradientBoostingRegressor(random_state=42,n_estimators=180,max_depth=3,learning_rate=0.04,loss='huber').fit(X,y)
    return primary,challenger

def _cost_features(plot_area,builtup_area,floors,num_rooms,num_bathrooms,category):
    return [plot_area,builtup_area,floors,num_rooms,num_bathrooms,{'basic':0,'standard':1,'premium':2}.get(category,1)]

def predict_cost(*,plot_area,builtup_area,floors,num_rooms,num_bathrooms,category)->float:
    models=_cost_models(); rate=CATEGORY_RATE.get(category,CATEGORY_RATE['standard'])
    formula=builtup_area*rate*(1+max(0,num_bathrooms-2)*0.015)
    if models is None:return float(formula)
    return float(models[0].predict(np.asarray([_cost_features(plot_area,builtup_area,floors,num_rooms,num_bathrooms,category)],dtype=float))[0])

def predict_cost_ensemble(*,plot_area,builtup_area,floors,num_rooms,num_bathrooms,category)->float:
    models=_cost_models()
    if models is None:return predict_cost(plot_area=plot_area,builtup_area=builtup_area,floors=floors,num_rooms=num_rooms,num_bathrooms=num_bathrooms,category=category)
    return float(models[1].predict(np.asarray([_cost_features(plot_area,builtup_area,floors,num_rooms,num_bathrooms,category)],dtype=float))[0])
