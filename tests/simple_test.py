import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import optuna
    print("✅ Optuna version:", optuna.__version__)
except Exception as e:
    print("❌ Optuna error:", e)

try:
    import xgboost
    print("✅ XGBoost version:", xgboost.__version__)
except Exception as e:
    print("❌ XGBoost error:", e)

try:
    import lightgbm
    print("✅ LightGBM version:", lightgbm.__version__)
except Exception as e:
    print("❌ LightGBM error:", e)

print("Test completado")
