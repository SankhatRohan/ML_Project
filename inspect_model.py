"""
Diagnostic script — inspect the saved .pkl model.
Run from: loan_default_app/
    python inspect_model.py
"""
import joblib, pickle, os, sys

MODEL_PATH = os.path.join("model", "loan_default_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

print("=" * 60)
print(f"Model type  : {type(model)}")
print(f"Model class : {model.__class__.__name__}")
print()

# Pipeline?
if hasattr(model, "steps"):
    print("Pipeline steps:")
    for name, step in model.steps:
        print(f"  [{name}] → {type(step).__name__}")
    print()
    final = model.steps[-1][1]
    print(f"Final estimator: {type(final).__name__}")
    if hasattr(final, "feature_names_in_"):
        print(f"feature_names_in_: {list(final.feature_names_in_)}")
    if hasattr(final, "n_features_in_"):
        print(f"n_features_in_: {final.n_features_in_}")
else:
    print("NOT a Pipeline — bare model")
    if hasattr(model, "feature_names_in_"):
        print(f"feature_names_in_: {list(model.feature_names_in_)}")
    if hasattr(model, "n_features_in_"):
        print(f"n_features_in_: {model.n_features_in_}")
    if hasattr(model, "estimators_"):
        print(f"n_estimators: {len(model.estimators_)}")
    if hasattr(model, "classes_"):
        print(f"classes_: {model.classes_}")

print("=" * 60)
