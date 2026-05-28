import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# ----------------------- Загрузка и очистка данных -----------------------
df = pd.read_excel("ml_moscow_flats.xlsx").dropna(subset=["price"])

# Числовые колонки: заменяем запятые на точки, конвертируем в float
num_cols = ["floorNumber", "floorsTotal", "totalArea", "kitchenArea",
            "latitude", "longitude"]
for col in num_cols:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Категориальная колонка
cat_col = "wallsMaterial"
df[cat_col] = df[cat_col].fillna('unknown').astype(str)

# Целевая переменная с логарифмированием
y = np.log(df["price"].values + 1e-8)

# Признаки: правильное объединение списка имён колонок
feature_cols = [cat_col] + num_cols
X = df[feature_cols]

# ----------------------- Сценарий 1: все данные (переобучение) -----------------------
print("=" * 60)
print("СЦЕНАРИЙ 1: ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ, ПРОВЕРКА НА НИХ ЖЕ")
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), [cat_col]),
        ("num", StandardScaler(), num_cols)
    ])
X_all = preprocessor.fit_transform(X)

mlp1 = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                    solver='adam', alpha=1e-4, batch_size=64,
                    learning_rate_init=0.001, max_iter=5000,
                    early_stopping=False, random_state=42)
mlp1.fit(X_all, y)
pred1 = mlp1.predict(X_all)
r2_train = r2_score(np.exp(y) - 1e-8, np.exp(pred1) - 1e-8)
print(f"R² на всех (тренировочных) данных: {r2_train:.4f}\n")

# ----------------------- Сценарий 2: обычное разбиение 80/20 -----------------------
print("=" * 60)
print("СЦЕНАРИЙ 2: ОБЫЧНОЕ РАЗБИЕНИЕ 80% train / 20% test")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

preprocessor2 = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), [cat_col]),
        ("num", StandardScaler(), num_cols)
    ])
X_train_prep = preprocessor2.fit_transform(X_train)
X_test_prep = preprocessor2.transform(X_test)

mlp2 = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                    solver='adam', alpha=1e-4, batch_size=64,
                    learning_rate_init=0.001, max_iter=5000,
                    early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=50, random_state=42)
mlp2.fit(X_train_prep, y_train)
pred2 = mlp2.predict(X_test_prep)
r2_holdout = r2_score(np.exp(y_test) - 1e-8, np.exp(pred2) - 1e-8)
print(f"R² на тестовой выборке: {r2_holdout:.4f}\n")

# ----------------------- Сценарий 3: кросс-валидация (5 фолдов) -----------------------
print("=" * 60)
print("СЦЕНАРИЙ 3: КРОСС-ВАЛИДАЦИЯ (5 фолдов)")
pipeline = Pipeline(steps=[
    ("preprocessor", ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), [cat_col]),
            ("num", StandardScaler(), num_cols)
        ])),
    ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', alpha=1e-4, batch_size=64,
                         learning_rate_init=0.001, max_iter=5000,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=50, random_state=42))
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
print(f"R² по фолдам: {scores}")
print(f"Среднее R² (CV): {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print("\n" + "=" * 60)
print(f"Сравнение:")
print(f"  - Переобучение (все данные): R² = {r2_train:.4f}")
print(f"  - Hold-out 80/20:            R² = {r2_holdout:.4f}")
print(f"  - Кросс-валидация:           R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")