import os
import pickle
import numpy as np
import csv


# ==========================================================
# НЕОБХОДИМЫЕ КЛАССЫ (должны точно совпадать с теми, что были при сохранении)
# ==========================================================

class Node:
    __slots__ = ['feature_idx', 'threshold', 'left', 'right', 'prediction']

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


class ExtraTree:
    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 n_random_splits=10, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_random_splits = n_random_splits
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) if random_state else np.random.RandomState()
        self.root = None

    def _gini(self, y):
        if len(y) == 0: return 0.0
        y_int = y.astype(int) if hasattr(y, 'astype') else np.array(y, dtype=int)
        max_label = int(np.max(y_int)) + 1
        counts = np.bincount(y_int, minlength=max_label)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _get_n_features(self, n_features):
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                return max(1, int(np.log2(n_features)))
            return n_features
        elif isinstance(self.max_features, (int, float)):
            return max(1, int(self.max_features * n_features))
        return n_features

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split: return None
        n_feats = self._get_n_features(n_features)
        feature_indices = self.rng.choice(n_features, size=n_feats, replace=False)
        best_gini, best_feat, best_thr = float('inf'), None, None
        for feat_idx in feature_indices:
            col = X[:, feat_idx].copy()
            if col.min() == col.max(): continue
            thresholds = self.rng.uniform(col.min(), col.max(), size=self.n_random_splits)
            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask
                l_size, r_size = left_mask.sum(), right_mask.sum()
                if l_size == 0 or r_size == 0: continue
                gini = (l_size / n_samples) * self._gini(y[left_mask]) + (r_size / n_samples) * self._gini(
                    y[right_mask])
                if gini < best_gini:
                    best_gini, best_feat, best_thr = gini, feat_idx, thr
        return (best_feat, best_thr) if best_feat is not None else None

    def _build_tree(self, X, y, depth):
        y = np.asarray(y).flatten()
        unique = np.unique(y)
        if len(unique) == 1: return Node(prediction=int(unique[0]))
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            y_int = y.astype(int)
            return Node(prediction=int(np.bincount(y_int, minlength=int(np.max(y_int)) + 1).argmax()))
        split = self._best_split(X, y)
        if split is None:
            y_int = y.astype(int)
            return Node(prediction=int(np.bincount(y_int, minlength=int(np.max(y_int)) + 1).argmax()))
        feat_idx, threshold = split
        l_mask, r_mask = X[:, feat_idx] <= threshold, ~(X[:, feat_idx] <= threshold)
        return Node(feature_idx=feat_idx, threshold=threshold,
                    left=self._build_tree(X[l_mask], y[l_mask], depth + 1),
                    right=self._build_tree(X[r_mask], y[r_mask], depth + 1))

    def fit(self, X, y):
        self.root = self._build_tree(np.asarray(X), np.asarray(y).flatten(), depth=0)
        return self

    def _predict_one(self, x, node):
        if node.prediction is not None: return node.prediction
        return self._predict_one(x, node.left if x[node.feature_idx] <= node.threshold else node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.asarray(X)])


class ExtraTreeClassifier:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 max_features='sqrt', n_random_splits=10, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_random_splits = n_random_splits
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) if random_state else np.random.RandomState()
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            seed = self.random_state + i if self.random_state is not None else None
            tree = ExtraTree(self.max_depth, self.min_samples_split, self.max_features, self.n_random_splits, seed)
            tree.fit(X, y)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        preds = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i].astype(int)
            preds.append(int(np.bincount(votes, minlength=int(np.max(votes)) + 1).argmax()))
        return np.array(preds)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y).flatten())


# ==========================================================
# ЗАГРУЗКА ДАННЫХ И ОЦЕНКА
# ==========================================================

def load_features(filepath):
    """Загружает только признаки (X)"""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)
    if data.ndim == 1: data = data.reshape(1, -1)
    return data


def load_ground_truth(filepath):
    """Загружает признаки и истинные метки (X, y)"""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)
    if data.ndim == 1: data = data.reshape(1, -1)
    return data[:, :-1], data[:, -1].astype(int)


def main():
    print("=" * 70)
    print("🔍 ОЦЕНКА СОХРАНЁННЫХ МОДЕЛЕЙ НА ТЕСТОВОМ ДАТАСЕТЕ")
    print("=" * 70)

    models_dir = "saved_models"
    if not os.path.exists(models_dir):
        print("❌ Папка 'saved_models' не найдена. Сначала запустите обучение и сохранение.")
        return

    # 1. Загрузка моделей
    models = {}
    model_files = {
        "raw": "extra_tree_raw.pkl",
        "random": "extra_tree_random.pkl",
        "stratified": "extra_tree_stratified.pkl"
    }
    for name, fname in model_files.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"✅ Загружена модель: {fname}")
        else:
            print(f"⚠️  Модель не найдена: {fname} (пропускаем)")

    if not models:
        print("❌ Не загружено ни одной модели. Прерываем.")
        return

    # 2. Загрузка тестовых данных и эталонов
    print("\n📂 Загрузка данных...")
    test_X = load_features("disease_public_test.csv")
    truth_X, truth_y = load_ground_truth("disease_sample_submission.csv")

    if len(test_X) != len(truth_y):
        print(f"⚠️  Внимание: количество строк в тесте ({len(test_X)}) и в эталоне ({len(truth_y)}) не совпадает.")
        print("   Используем эталонные X для предсказания, чтобы индексы совпадали.")
        test_X = truth_X  # fallback на совпадающий датасет

    print(f"   Тестовых объектов: {len(test_X)}")
    print(f"   Истинных меток (Y): {len(truth_y)}")

    # 3. Предсказания и оценка
    print("\n📊 Оценка точности:")
    predictions = {}
    for name, model in models.items():
        preds = model.predict(test_X)
        acc = np.mean(preds == truth_y)
        predictions[name] = preds
        correct = np.sum(preds == truth_y)
        print(f"   {name:12} | Accuracy: {acc:.4f} | Верно: {correct}/{len(truth_y)}")

    # 4. Подробная таблица сравнения
    print("\n📋 Детальное сравнение (первые 10 объектов):")
    print(f"{'№':<3} | {'True Y':<6} | {'Raw':<5} | {'Random':<8} | {'Stratified':<10} | {'Статус'}")
    print("-" * 65)
    n_show = min(10, len(truth_y))
    for i in range(n_show):
        t = truth_y[i]
        p_raw = predictions.get('raw', [])[i] if 'raw' in predictions else '-'
        p_rand = predictions.get('random', [])[i] if 'random' in predictions else '-'
        p_strat = predictions.get('stratified', [])[i] if 'stratified' in predictions else '-'

        match = []
        if 'raw' in predictions and predictions['raw'][i] == t: match.append('R')
        if 'random' in predictions and predictions['random'][i] == t: match.append('Ra')
        if 'stratified' in predictions and predictions['stratified'][i] == t: match.append('S')
        status = "✅ " + ",".join(match) if match else "❌ None"

        print(f"{i + 1:<3} | {t:<6} | {p_raw:<5} | {p_rand:<8} | {p_strat:<10} | {status}")

    # 5. Сохранение результатов
    output_path = "model_predictions.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['ID', 'True_Y']
        for name in models: header.append(f'Pred_{name.capitalize()}')
        header.append('All_Correct')
        writer.writerow(header)

        for i in range(len(truth_y)):
            row = [i + 1, int(truth_y[i])]
            correct_flags = []
            for name in models:
                p = int(predictions[name][i])
                row.append(p)
                correct_flags.append(p == truth_y[i])
            row.append('Yes' if all(correct_flags) else 'No')
            writer.writerow(row)

    print(f"\n💾 Полные предсказания сохранены в: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
