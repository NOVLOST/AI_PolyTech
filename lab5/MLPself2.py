import numpy as np
import pandas as pd

# ------------------------ Вспомогательные классы ------------------------
class OneHotEncoder:
    def fit(self, col):
        vals = np.array(col).astype(str)
        self.categories_ = np.unique(vals)
        return self
    def transform(self, col):
        vals = np.array(col).astype(str)
        mat = np.zeros((len(vals), len(self.categories_)))
        for i, cat in enumerate(self.categories_):
            mat[:, i] = (vals == cat).astype(int)
        return mat
    def fit_transform(self, col):
        self.fit(col)
        return self.transform(col)

class StandardScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, X):
        return (X - self.mean) / self.std
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def safe_float(series):
    return series.astype(str).str.replace(',', '.').astype(float)

# ------------------------ MLPRegressor (чистый NumPy) ------------------------
class MLPRegressor:
    def __init__(self, hidden_layers=(64,), activation='relu', lr=1e-3,
                 epochs=1000, batch_size=32, val_split=0.1, patience=20,
                 weight_decay=1e-4, verbose=False):
        self.hidden_layers = hidden_layers
        self.activation_name = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.patience = patience
        self.weight_decay = weight_decay
        self.verbose = verbose

    def _init_weights(self, layer_dims):
        self.W, self.b = [], []
        for i in range(len(layer_dims)-1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i+1]
            std = np.sqrt(2./fan_in) if self.activation_name == 'relu' else np.sqrt(1./fan_in)
            self.W.append(np.random.randn(fan_in, fan_out)*std)
            self.b.append(np.zeros((1, fan_out)))

    def _activate(self, Z):
        if self.activation_name == 'relu': return np.maximum(0, Z)
        if self.activation_name == 'tanh': return np.tanh(Z)
        return Z

    def _activate_derivative(self, A):
        if self.activation_name == 'relu': return (A > 0).astype(float)
        if self.activation_name == 'tanh': return 1 - A**2
        return np.ones_like(A)

    def _forward(self, X):
        self.Z, self.A = [], [X]
        for i in range(len(self.W)):
            z = self.A[-1] @ self.W[i] + self.b[i]
            self.Z.append(z)
            a = z if i == len(self.W)-1 else self._activate(z)
            self.A.append(a)
        return self.A[-1]

    def _backward(self, y_true, y_pred):
        m = y_true.shape[0]
        dA = (y_pred - y_true) / m
        grads_W, grads_b = [], []
        for i in reversed(range(len(self.W))):
            dZ = dA if i == len(self.W)-1 else dA * self._activate_derivative(self.A[i+1])
            dW = self.A[i].T @ dZ + self.weight_decay * self.W[i]
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ self.W[i].T
            grads_W.insert(0, dW)
            grads_b.insert(0, db)
        return grads_W, grads_b

    def _adam_init(self):
        self.m_W = [np.zeros_like(w) for w in self.W]
        self.v_W = [np.zeros_like(w) for w in self.W]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def _adam_update(self, grads_W, grads_b):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self.m_W[i] = beta1*self.m_W[i] + (1-beta1)*grads_W[i]
            self.v_W[i] = beta2*self.v_W[i] + (1-beta2)*grads_W[i]**2
            m_hat = self.m_W[i] / (1 - beta1**self.t)
            v_hat = self.v_W[i] / (1 - beta2**self.t)
            self.W[i] -= self.lr * m_hat / (np.sqrt(v_hat)+eps)

            self.m_b[i] = beta1*self.m_b[i] + (1-beta1)*grads_b[i]
            self.v_b[i] = beta2*self.v_b[i] + (1-beta2)*grads_b[i]**2
            m_hat_b = self.m_b[i] / (1 - beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - beta2**self.t)
            self.b[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b)+eps)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Если X_val и y_val не переданы, валидационная часть берётся из X, y
        согласно val_split. Если val_split=0, валидация отсутствует.
        """
        if X_val is None and self.val_split > 0:
            np.random.seed(42)
            idx = np.random.permutation(len(X))
            val_size = int(len(X)*self.val_split)
            train_idx, val_idx = idx[val_size:], idx[:val_size]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
        else:
            X_train, y_train = X, y

        layer_dims = [X.shape[1]] + list(self.hidden_layers) + [1]
        self._init_weights(layer_dims)
        self._adam_init()

        best_loss = np.inf
        wait = 0
        for epoch in range(self.epochs):
            perm = np.random.permutation(len(X_train))
            X_train, y_train = X_train[perm], y_train[perm]
            for i in range(0, len(X_train), self.batch_size):
                Xb, yb = X_train[i:i+self.batch_size], y_train[i:i+self.batch_size]
                y_pred = self._forward(Xb)
                grads_W, grads_b = self._backward(yb, y_pred)
                self._adam_update(grads_W, grads_b)

            # оценка на валидации (если есть)
            if X_val is not None:
                val_pred = self._forward(X_val)
                val_loss = np.mean((val_pred - y_val)**2)
                if self.verbose and epoch%100==0:
                    print(f"  Epoch {epoch}: val_loss={val_loss:.6f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0
                    best_W, best_b = [w.copy() for w in self.W], [b.copy() for b in self.b]
                else:
                    wait += 1
                    if wait >= self.patience:
                        if self.verbose: print(f"  Early stop at epoch {epoch}")
                        break
            else:
                # без валидации просто запоминаем последние веса
                best_W, best_b = [w.copy() for w in self.W], [b.copy() for b in self.b]

        self.W, self.b = best_W, best_b

    def predict(self, X):
        return self._forward(X)

# ------------------------ Загрузка и подготовка сырых данных ------------------------
df = pd.read_excel("ml_moscow_flats.xlsx")
df = df.dropna(subset=["price"])

# Исходные столбцы
walls_raw = df["wallsMaterial"].fillna('unknown').astype(str).values
num_features = {
    "floorNumber": safe_float(df["floorNumber"]).values.reshape(-1,1),
    "floorsTotal": safe_float(df["floorsTotal"]).values.reshape(-1,1),
    "totalArea":   safe_float(df["totalArea"]).values.reshape(-1,1),
    "kitchenArea": safe_float(df["kitchenArea"]).values.reshape(-1,1),
    "latitude":    safe_float(df["latitude"]).values.reshape(-1,1),
    "longitude":   safe_float(df["longitude"]).values.reshape(-1,1)
}
y_log = np.log(df["price"].values.reshape(-1,1).astype(float) + 1e-8)

# ------------------------ Метрика R² ------------------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

# ------------------------ Сценарий 1: обучение = проверка (все данные) ------------------------
print("="*60)
print("СЦЕНАРИЙ 1: ОБУЧЕНИЕ И ПРОВЕРКА НА ВСЕХ ДАННЫХ (переобучение)")
# one-hot и масштабирование на всех данных
ohe_all = OneHotEncoder()
walls_ohe = ohe_all.fit_transform(walls_raw)
num_all = np.hstack([num_features[k] for k in num_features])
scaler_all = StandardScaler()
num_scaled = scaler_all.fit_transform(num_all)
X_all = np.hstack([walls_ohe, num_scaled])
y_all = y_log

mlp_all = MLPRegressor(hidden_layers=[128,64], activation='relu', lr=0.001,
                       epochs=1000, batch_size=64, val_split=0.0,   # валидация отключена
                       patience=500, weight_decay=1e-4, verbose=True)
mlp_all.fit(X_all, y_all)                     # нет валидации
pred_all = mlp_all.predict(X_all)
r2_train = r2_score(np.exp(y_all)-1e-8, np.exp(pred_all)-1e-8)
print(f"R² на всех (тренировочных) данных: {r2_train:.4f}\n")

# ------------------------ Сценарий 2: Обычное разбиение 80/20 ------------------------
print("="*60)
print("СЦЕНАРИЙ 2: ОБЫЧНОЕ РАЗБИЕНИЕ 80% train / 20% test")
np.random.seed(42)
idx = np.random.permutation(len(df))
split = int(0.8 * len(df))
train_idx, test_idx = idx[:split], idx[split:]

ohe = OneHotEncoder()
X_train_walls = ohe.fit_transform(walls_raw[train_idx])
X_test_walls = ohe.transform(walls_raw[test_idx])

num_train = np.hstack([num_features[k][train_idx] for k in num_features])
num_test = np.hstack([num_features[k][test_idx] for k in num_features])

scaler = StandardScaler()
X_train_num = scaler.fit_transform(num_train)
X_test_num = scaler.transform(num_test)

X_train = np.hstack([X_train_walls, X_train_num])
X_test = np.hstack([X_test_walls, X_test_num])
y_train, y_test = y_log[train_idx], y_log[test_idx]

mlp2 = MLPRegressor(hidden_layers=[128,64], activation='relu', lr=0.001,
                    epochs=5000, batch_size=64, val_split=0.1, patience=50,
                    weight_decay=1e-4, verbose=True)
mlp2.fit(X_train, y_train)
pred_test = mlp2.predict(X_test)
r2_holdout = r2_score(np.exp(y_test)-1e-8, np.exp(pred_test)-1e-8)
print(f"R² на тестовой выборке: {r2_holdout:.4f}\n")

# ------------------------ Сценарий 3: Кросс-валидация (5 фолдов) ------------------------
print("="*60)
print("СЦЕНАРИЙ 3: КРОСС-ВАЛИДАЦИЯ (5 фолдов)")
k = 5
np.random.seed(42)
indices = np.random.permutation(len(df))
fold_size = len(df) // k
folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(k-1)]
folds.append(indices[(k-1)*fold_size:])

r2_cv = []
for fold in range(k):
    test_idx = folds[fold]
    train_idx = np.concatenate([folds[i] for i in range(k) if i != fold])

    ohe_cv = OneHotEncoder()
    X_tr_w = ohe_cv.fit_transform(walls_raw[train_idx])
    X_te_w = ohe_cv.transform(walls_raw[test_idx])

    num_tr = np.hstack([num_features[k][train_idx] for k in num_features])
    num_te = np.hstack([num_features[k][test_idx] for k in num_features])

    scaler_cv = StandardScaler()
    X_tr_num = scaler_cv.fit_transform(num_tr)
    X_te_num = scaler_cv.transform(num_te)

    X_tr = np.hstack([X_tr_w, X_tr_num])
    X_te = np.hstack([X_te_w, X_te_num])
    y_tr, y_te = y_log[train_idx], y_log[test_idx]

    print(f"\nFold {fold+1}/{k}")
    mlp_cv = MLPRegressor(hidden_layers=[128,64], activation='relu', lr=0.001,
                          epochs=5000, batch_size=64, val_split=0.1, patience=50,
                          weight_decay=1e-4, verbose=False)
    mlp_cv.fit(X_tr, y_tr)
    pred = mlp_cv.predict(X_te)
    r2_fold = r2_score(np.exp(y_te)-1e-8, np.exp(pred)-1e-8)
    r2_cv.append(r2_fold)
    print(f"R² fold {fold+1}: {r2_fold:.4f}")

print("\n" + "="*60)
print(f"Среднее R² (кросс-валидация): {np.mean(r2_cv):.4f} ± {np.std(r2_cv):.4f}")
print(f"Сравнение:")
print(f"  - R² на всех данных (переобучение): {r2_train:.4f}")
print(f"  - R² на тесте (80/20):              {r2_holdout:.4f}")
print(f"  - R² среднее по CV:                 {np.mean(r2_cv):.4f}")