import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

# --------------------------------------------
# 1. Загрузка и подготовка данных
# --------------------------------------------
data = pd.read_excel('ml_moscow_flats.xlsx')
data = data.drop_duplicates().reset_index(drop=True)

# Исправление десятичных запятых
for col in ['latitude', 'longitude', 'totalArea', 'kitchenArea']:
    if data[col].dtype == object:
        data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# --------------------------------------------
# 2. Генерация признаков
# --------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

center_lat, center_lon = 55.751244, 37.618423
data['dist_to_center'] = data.apply(
    lambda row: haversine_distance(row['latitude'], row['longitude'], center_lat, center_lon), axis=1)

data['floor_ratio'] = data['floorNumber'] / data['floorsTotal']
data['is_first_floor'] = (data['floorNumber'] == 1).astype(int)
data['is_last_floor'] = (data['floorNumber'] == data['floorsTotal']).astype(int)
data['kitchen_area_ratio'] = data['kitchenArea'] / data['totalArea']
data['total_area_sq'] = data['totalArea'] ** 2
data['log_total_area'] = np.log(data['totalArea'])
data['log_kitchen_area'] = np.log(data['kitchenArea'] + 1)

walls_dummies = pd.get_dummies(data['wallsMaterial'], prefix='walls')
data = pd.concat([data, walls_dummies], axis=1)

features = [
    'floorNumber', 'floorsTotal', 'totalArea', 'kitchenArea',
    'dist_to_center', 'floor_ratio', 'is_first_floor', 'is_last_floor',
    'kitchen_area_ratio', 'total_area_sq', 'log_total_area', 'log_kitchen_area'
] + list(walls_dummies.columns)

X = data[features].values.astype(np.float64)
y = data['price'].values.astype(np.float64)

# Логарифмирование целевой переменной
y_log = np.log(y)

# --------------------------------------------
# 3. Масштабирование признаков
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# 4. Сценарий 1: обучение и оценка на всех данных
# --------------------------------------------
print("=== Сценарий 1: Все данные ===")
# Подбор гиперпараметров на всех данных с помощью кросс-валидации
cv_model_all = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
                            alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                            cv=5, random_state=42, max_iter=5000)
cv_model_all.fit(X_scaled, y_log)

best_alpha = cv_model_all.alpha_
best_l1 = cv_model_all.l1_ratio_
print(f"Лучшие параметры: alpha={best_alpha:.4f}, l1_ratio={best_l1:.2f}")

# Обучение финальной модели на всех данных
model_all = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5000)
model_all.fit(X_scaled, y_log)

# Предсказание и перевод в исходный масштаб
y_pred_log_all = model_all.predict(X_scaled)
y_pred_all = np.exp(y_pred_log_all)

r2_all = r2_score(y, y_pred_all)
mae_all = mean_absolute_error(y, y_pred_all)
print(f"R² на всех данных: {r2_all:.4f}")
print(f"MAE на всех данных: {mae_all:,.0f} руб.\n")

# --------------------------------------------
# 5. Сценарий 2: разделение 80/20
# --------------------------------------------
print("=== Сценарий 2: Разделение 80/20 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_log, test_size=0.2, random_state=42)

# Подбор параметров только на обучающей части
cv_model_8020 = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
                             alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                             cv=5, random_state=42, max_iter=5000)
cv_model_8020.fit(X_train, y_train)

best_alpha_80 = cv_model_8020.alpha_
best_l1_80 = cv_model_8020.l1_ratio_
print(f"Лучшие параметры: alpha={best_alpha_80:.4f}, l1_ratio={best_l1_80:.2f}")

model_8020 = ElasticNet(alpha=best_alpha_80, l1_ratio=best_l1_80, max_iter=5000)
model_8020.fit(X_train, y_train)

y_pred_log_test = model_8020.predict(X_test)
y_pred_test = np.exp(y_pred_log_test)

r2_test = r2_score(np.exp(y_test), y_pred_test)
mae_test = mean_absolute_error(np.exp(y_test), y_pred_test)
print(f"R² на тесте: {r2_test:.4f}")
print(f"MAE на тесте: {mae_test:,.0f} руб.\n")

# --------------------------------------------
# 6. Сценарий 3: 5-кратная кросс-валидация
# --------------------------------------------
print("=== Сценарий 3: Кросс-валидация (5 фолдов) ===")
# Для кросс-валидации используем ElasticNetCV, чтобы внутри каждого фолда
# параметры подбирались заново – это даёт более честную оценку.
cv_scores = cross_val_score(
    ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99],
                 alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                 cv=3, max_iter=5000, random_state=42),
    X_scaled, y_log, cv=5, scoring='r2'
)

print(f"R² по фолдам: {cv_scores}")
print(f"Среднее R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")