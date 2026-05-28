import pandas as pd
import numpy as np

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
# 2. Генерация признаков (такая же, как в исходном решении)
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
y_orig = data['price'].values.astype(np.float64)
y_log = np.log(y_orig)

# --------------------------------------------
# 3. Класс ElasticNetGD и метрика R²
# --------------------------------------------
class ElasticNetGD:
    def __init__(self, alpha=0.1, l1_ratio=0.5, learning_rate=0.01, n_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        prev_loss = float('inf')

        for _ in range(self.n_iter):
            y_pred = X @ self.weights
            error = y_pred - y
            grad_mse = (1 / n_samples) * X.T @ error
            grad_l2 = self.alpha * (1 - self.l1_ratio) * self.weights
            grad_l1 = self.alpha * self.l1_ratio * np.sign(self.weights)
            grad = grad_mse + grad_l2 + grad_l1
            self.weights -= self.lr * grad

            mse = (1 / (2 * n_samples)) * np.sum(error ** 2)
            l2_pen = 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(self.weights ** 2)
            l1_pen = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights))
            loss = mse + l2_pen + l1_pen

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

    def predict(self, X):
        return X @ self.weights

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# --------------------------------------------
# 4. Подбор гиперпараметров (на 80% трейне) для использования во всех сценариях
# --------------------------------------------
np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = idx[:split], idx[split:]

X_train_full, X_test = X[train_idx], X[test_idx]
y_train_full_log, y_test_orig = y_log[train_idx], y_orig[test_idx]
y_train_full_orig = y_orig[train_idx]

# Масштабирование на трейне
train_mean = X_train_full.mean(axis=0)
train_std = X_train_full.std(axis=0) + 1e-8
X_train_full_scaled = (X_train_full - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

y_mean_log = y_train_full_log.mean()
y_train_full_centered = y_train_full_log - y_mean_log

# Внутренняя валидация для подбора гиперпараметров
np.random.seed(123)
val_size = int(0.2 * len(X_train_full_scaled))
sub_idx = np.random.permutation(len(X_train_full_scaled))
train_sub_idx, val_idx = sub_idx[val_size:], sub_idx[:val_size]

X_subtrain = X_train_full_scaled[train_sub_idx]
y_subtrain = y_train_full_centered[train_sub_idx]
X_val = X_train_full_scaled[val_idx]
y_val_orig = y_train_full_orig[val_idx]

alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.99]
best_r2 = -np.inf
best_alpha, best_l1 = alphas[0], l1_ratios[0]

print("Подбор гиперпараметров на валидации...")
for alpha in alphas:
    for l1_ratio in l1_ratios:
        model = ElasticNetGD(alpha=alpha, l1_ratio=l1_ratio,
                             learning_rate=0.01, n_iter=2000, tol=1e-5)
        model.fit(X_subtrain, y_subtrain)
        y_val_pred_centered = model.predict(X_val)
        y_val_pred_log = y_val_pred_centered + y_mean_log
        y_val_pred = np.exp(y_val_pred_log)
        r2 = r2_score(y_val_orig, y_val_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha, best_l1 = alpha, l1_ratio

print(f"Лучшие гиперпараметры: alpha={best_alpha}, l1_ratio={best_l1}\n")

# --------------------------------------------
# 5. Сценарий 1: обучение и оценка на ВСЕХ данных (переобучение)
# --------------------------------------------
print("=== Сценарий 1: Обучение и оценка на всех данных ===")
# Масштабируем все данные
all_mean = X.mean(axis=0)
all_std = X.std(axis=0) + 1e-8
X_all_scaled = (X - all_mean) / all_std

y_all_log = y_log
y_all_centered = y_all_log - y_all_log.mean()

model_all = ElasticNetGD(alpha=best_alpha, l1_ratio=best_l1,
                         learning_rate=0.01, n_iter=3000, tol=1e-6)
model_all.fit(X_all_scaled, y_all_centered)

y_all_pred_centered = model_all.predict(X_all_scaled)
y_all_pred_log = y_all_pred_centered + y_all_log.mean()
y_all_pred = np.exp(y_all_pred_log)

r2_all = r2_score(y_orig, y_all_pred)
mae_all = np.mean(np.abs(y_orig - y_all_pred))
print(f"R² на всех данных (переобучение): {r2_all:.4f}")
print(f"MAE на всех данных: {mae_all:,.0f} руб.\n")

# --------------------------------------------
# 6. Сценарий 2: обычное разделение 80/20
# --------------------------------------------
print("=== Сценарий 2: Обычное разделение 80/20 ===")
final_model = ElasticNetGD(alpha=best_alpha, l1_ratio=best_l1,
                           learning_rate=0.01, n_iter=3000, tol=1e-6)
final_model.fit(X_train_full_scaled, y_train_full_centered)

y_test_pred_centered = final_model.predict(X_test_scaled)
y_test_pred_log = y_test_pred_centered + y_mean_log
y_test_pred = np.exp(y_test_pred_log)

r2_test = r2_score(y_test_orig, y_test_pred)
mae_test = np.mean(np.abs(y_test_orig - y_test_pred))
print(f"R² на тесте (80/20): {r2_test:.4f}")
print(f"MAE на тесте: {mae_test:,.0f} руб.\n")

# --------------------------------------------
# 7. Сценарий 3: кросс-валидация (5 фолдов)
# --------------------------------------------
print("=== Сценарий 3: 5-кратная кросс-валидация ===")
k = 5
np.random.seed(999)
indices = np.random.permutation(len(X))
fold_size = len(X) // k
r2_scores = []

for fold in range(k):
    # Определяем индексы текущего фолда
    start = fold * fold_size
    end = start + fold_size if fold != k-1 else len(X)
    val_idx_cv = indices[start:end]
    train_idx_cv = np.setdiff1d(indices, val_idx_cv)

    X_train_cv, X_val_cv = X[train_idx_cv], X[val_idx_cv]
    y_train_cv_log, y_val_cv_orig = y_log[train_idx_cv], y_orig[val_idx_cv]

    # Масштабирование только по обучающей части фолда
    mu = X_train_cv.mean(axis=0)
    sigma = X_train_cv.std(axis=0) + 1e-8
    X_train_cv_scaled = (X_train_cv - mu) / sigma
    X_val_cv_scaled = (X_val_cv - mu) / sigma

    y_train_cv_centered = y_train_cv_log - y_train_cv_log.mean()

    # Обучение модели с теми же гиперпараметрами
    model_cv = ElasticNetGD(alpha=best_alpha, l1_ratio=best_l1,
                            learning_rate=0.01, n_iter=2000, tol=1e-5)
    model_cv.fit(X_train_cv_scaled, y_train_cv_centered)

    # Предсказание и перевод в исходный масштаб
    y_val_pred_centered = model_cv.predict(X_val_cv_scaled)
    y_val_pred_log = y_val_pred_centered + y_train_cv_log.mean()
    y_val_pred = np.exp(y_val_pred_log)

    r2_fold = r2_score(y_val_cv_orig, y_val_pred)
    r2_scores.append(r2_fold)
    print(f"  Фолд {fold+1}: R² = {r2_fold:.4f}")

r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)
print(f"\nСреднее R² по кросс-валидации: {r2_mean:.4f} ± {r2_std:.4f}")