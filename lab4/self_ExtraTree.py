import sys
import numpy as np
import random
import os
import pickle
import json
from datetime import datetime


def check_training_data(filepath="disease_train.csv"):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)
    X, y = data[:, :-1], data[:, -1].astype(int)

    print("📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ В ТРЕНИРОВКЕ:")
    print(f"   Класс 0: {np.sum(y == 0)} ({np.mean(y == 0) * 100:.1f}%)")
    print(f"   Класс 1: {np.sum(y == 1)} ({np.mean(y == 1) * 100:.1f}%)")
    print()

    print("📊 КОРРЕЛЯЦИЯ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (точечно-бисериальная):")
    for i in range(X.shape[1]):
        # Простая аппроксимация: корреляция Пирсона для бинарного Y
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"   X{i + 1}: {corr:+.3f} {'✅ Информативен' if abs(corr) > 0.15 else '⚠️ Слабый/Шум'}")
    print()


# Вызовите в main() после загрузки данных:
# check_training_data(dataset_path)

# ==========================================================
# РЕАЛИЗАЦИЯ EXTRA TREE ALGORITHM (ВАШ АЛГОРИТМ)
# ==========================================================
class Node:
    """Узел дерева решений"""
    __slots__ = ['feature_idx', 'threshold', 'left', 'right', 'prediction']

    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


class ExtraTree:
    """Одно дерево Extra Tree"""

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
        """Расчёт индекса Джини"""
        if len(y) == 0:
            return 0.0
        y_int = y.astype(int) if hasattr(y, 'astype') else np.array(y, dtype=int)
        max_label = int(np.max(y_int)) + 1
        counts = np.bincount(y_int, minlength=max_label)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _get_n_features(self, n_features):
        """Определяет количество признаков для рассмотрения"""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == 'log2':
                return max(1, int(np.log2(n_features)))
            else:
                return n_features
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features

    def _best_split(self, X, y):
        """Поиск лучшего случайного разделения"""
        n_samples, n_features = X.shape

        if n_samples < self.min_samples_split:
            return None

        n_feats = self._get_n_features(n_features)
        feature_indices = self.rng.choice(n_features, size=n_feats, replace=False)

        best_gini = float('inf')
        best_feat = None
        best_thr = None

        for feat_idx in feature_indices:
            col = X[:, feat_idx].copy()
            min_val = col.min()
            max_val = col.max()

            if min_val == max_val:
                continue

            thresholds = self.rng.uniform(min_val, max_val, size=self.n_random_splits)

            for thr in thresholds:
                left_mask = col <= thr
                right_mask = ~left_mask

                left_size = left_mask.sum()
                right_size = right_mask.sum()

                if left_size == 0 or right_size == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])

                gini = (left_size / n_samples) * left_gini + (right_size / n_samples) * right_gini

                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat_idx
                    best_thr = thr

        return (best_feat, best_thr) if best_feat is not None else None

    def _build_tree(self, X, y, depth):
        """Рекурсивное построение дерева"""
        y = np.asarray(y).flatten()
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            return Node(prediction=int(unique_classes[0]))

        if depth >= self.max_depth or len(y) < self.min_samples_split:
            y_int = y.astype(int)
            max_label = int(np.max(y_int)) + 1
            most_common = np.bincount(y_int, minlength=max_label).argmax()
            return Node(prediction=int(most_common))

        split = self._best_split(X, y)
        if split is None:
            y_int = y.astype(int)
            max_label = int(np.max(y_int)) + 1
            most_common = np.bincount(y_int, minlength=max_label).argmax()
            return Node(prediction=int(most_common))

        feat_idx, threshold = split
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx=feat_idx, threshold=threshold,
                    left=left_child, right=right_child)

    def fit(self, X, y):
        """Обучение дерева"""
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        """Предсказание для одного объекта"""
        if node.prediction is not None:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        """Предсказание для массива объектов"""
        X = np.asarray(X)
        return np.array([self._predict_one(x, self.root) for x in X])


class ExtraTreeClassifier:
    """Ансамбль Extra Trees"""

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
        """Обучение ансамбля"""
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        self.trees = []
        for i in range(self.n_estimators):
            tree_seed = self.random_state + i if self.random_state is not None else None
            tree = ExtraTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                n_random_splits=self.n_random_splits,
                random_state=tree_seed
            )
            tree.fit(X, y)
            self.trees.append(tree)
        return self

    def predict(self, X):
        """Предсказание с голосованием"""
        X = np.asarray(X)
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        predictions = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            max_label = int(np.max(votes)) + 1 if len(votes) > 0 else 2
            most_common = np.bincount(votes.astype(int), minlength=max_label).argmax()
            predictions.append(int(most_common))
        return np.array(predictions)

    def score(self, X, y):
        """Точность модели"""
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        pred = self.predict(X)
        return np.mean(pred == y)


# ==========================================================
# ФУНКЦИИ ДЛЯ РАЗБИЕНИЯ ДАННЫХ
# ==========================================================

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def stratified_shuffle_split_manual(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    y = np.asarray(y).flatten()
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0].tolist()
        random.shuffle(cls_indices)

        n_cls_test = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_cls_test])
        train_indices.extend(cls_indices[n_cls_test:])

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ==========================================================
# ЗАГРУЗКА КАСТОМНОГО CSV ДАТАСЕТА
# ==========================================================

def load_custom_dataset(filepath):
    """
    Загружает датасет из CSV файла формата:
    X1,X2,X3,X4,X5,X6,X7,Y
    (первая строка считается заголовком и пропускается)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл '{filepath}' не найден.")

    data = np.genfromtxt(filepath, delimiter=',', skip_header=1, dtype=float)

    if data.ndim == 1:
        raise ValueError("Датасет содержит меньше 2 строк или имеет неверный формат.")

    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


# ==========================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ==========================================================

def run_experiment(X, y, n_runs=30, n_estimators=20, max_depth=5):
    """
    Запуск эксперимента. Возвращает:
    1. results: списки accuracy для каждого подхода
    2. best_models: словарь с лучшими моделями для каждого подхода
    3. best_acc: словарь с лучшими accuracy
    """
    results = {'raw_data': [], 'random_split': [], 'stratified_split': []}
    best_models = {'raw_data': None, 'random_split': None, 'stratified_split': None}
    best_acc = {'raw_data': -1.0, 'random_split': -1.0, 'stratified_split': -1.0}

    for run in range(n_runs):
        print(f"  Итерация {run + 1}/{n_runs}", end='\r')
        sys.stdout.flush()

        # 1. Сырые данные
        model_raw = ExtraTreeClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        max_features='sqrt', n_random_splits=10, random_state=run)
        model_raw.fit(X, y)
        acc_raw = model_raw.score(X, y)
        results['raw_data'].append(acc_raw)
        if acc_raw > best_acc['raw_data']:
            best_acc['raw_data'] = acc_raw
            best_models['raw_data'] = model_raw

        # 2. Обычное разбиение
        X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2, random_state=run)
        model_random = ExtraTreeClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                           max_features='sqrt', n_random_splits=10, random_state=run)
        model_random.fit(X_train, y_train)
        acc_random = model_random.score(X_test, y_test)
        results['random_split'].append(acc_random)
        if acc_random > best_acc['random_split']:
            best_acc['random_split'] = acc_random
            best_models['random_split'] = model_random

        # 3. Стратифицированное разбиение
        X_train_s, X_test_s, y_train_s, y_test_s = stratified_shuffle_split_manual(X, y, test_size=0.2,
                                                                                   random_state=run)
        model_strat = ExtraTreeClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          max_features='sqrt', n_random_splits=10, random_state=run)
        model_strat.fit(X_train_s, y_train_s)
        acc_strat = model_strat.score(X_test_s, y_test_s)
        results['stratified_split'].append(acc_strat)
        if acc_strat > best_acc['stratified_split']:
            best_acc['stratified_split'] = acc_strat
            best_models['stratified_split'] = model_strat

    print()
    return results, best_models, best_acc


def calculate_statistics(results):
    stats = {}
    for key, values in results.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    return stats


def plot_results(results, stats):
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [results['raw_data'], results['random_split'], results['stratified_split']]
        labels = [
            f'Сырые данные\nμ={stats["raw_data"]["mean"]:.4f}±{stats["raw_data"]["std"]:.4f}',
            f'Обычное разбиение\nμ={stats["random_split"]["mean"]:.4f}±{stats["random_split"]["std"]:.4f}',
            f'Стратифицированное\nμ={stats["stratified_split"]["mean"]:.4f}±{stats["stratified_split"]["std"]:.4f}'
        ]

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['#FF9999', '#99FF99', '#9999FF']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        for i, key in enumerate(results.keys(), 1):
            ax.scatter(i, stats[key]['mean'], color='red', s=100, zorder=3, marker='D')

        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Сравнение точности Extra Tree\n(разные подходы к разбиению)', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\n⚠️ Matplotlib не установлен. Пропускаем построение графика.")


def save_models(best_models, best_acc, params, stats, dataset_path):
    """Сохраняет лучшие модели и метаданные"""
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    files_to_save = [
        ("extra_tree_raw.pkl", "raw_data", "Сырые данные"),
        ("extra_tree_random.pkl", "random_split", "Обычное разбиение"),
        ("extra_tree_stratified.pkl", "stratified_split", "Стратифицированное разбиение")
    ]

    for filename, key, desc in files_to_save:
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(best_models[key], f)
        print(f"   ✅ {desc:25} -> {filename} (Accuracy: {best_acc[key]:.4f})")

    # Сохраняем метаданные для воспроизводимости
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_path,
        "parameters": params,
        "best_accuracies": best_acc,
        "full_statistics": stats
    }
    with open(os.path.join(save_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"   ✅ metadata.json сохранён (параметры и статистика)")


# ==========================================================
# ЗАПУСК
# ==========================================================

def main():
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ: Сравнение подходов к обучению Extra Tree")
    print("=" * 70)

    dataset_path = "disease_train.csv"

    print(f"\n1. Загрузка датасета из '{dataset_path}'...")
    try:
        X, y = load_custom_dataset(dataset_path)
        n_classes = len(np.unique(y))
        print(f"   ✅ Загружено: {X.shape[0]} объектов, {X.shape[1]} признаков, {n_classes} классов")
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return

    n_runs = 30
    n_estimators = 20
    max_depth = 5
    params = {"n_runs": n_runs, "n_estimators": n_estimators, "max_depth": max_depth, "max_features": "sqrt"}

    print(f"\n2. Параметры эксперимента:")
    for k, v in params.items():
        print(f"   - {k}: {v}")

    print("\n3. Запуск экспериментов...")
    results, best_models, best_acc = run_experiment(
        X, y,
        n_runs=n_runs,
        n_estimators=n_estimators,
        max_depth=max_depth
    )

    print("\n4. Расчёт статистики...")
    stats = calculate_statistics(results)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)

    labels_names = {
        'raw_data': 'Сырые данные (обучение+тест на всех)',
        'random_split': 'Обычное разбиение (80/20)',
        'stratified_split': 'Стратифицированное разбиение (80/20)'
    }

    for key in results.keys():
        print(f"   {labels_names[key]:35}: {stats[key]['mean']:.4f} ± {stats[key]['std']:.4f}")
        print(f"                                      (min: {stats[key]['min']:.4f}, max: {stats[key]['max']:.4f})")
        print()

    print("\n5. Построение графика...")
    plot_results(results, stats)

    # Сохранение моделей
    print("\n6. Сохранение лучших моделей...")
    save_models(best_models, best_acc, params, stats, dataset_path)

    # Выводы
    print("\n" + "=" * 70)
    print("ВЫВОДЫ")
    print("=" * 70)

    raw_mean, random_mean, strat_mean = stats['raw_data']['mean'], stats['random_split']['mean'], stats['stratified_split']['mean']
    random_std, strat_std = stats['random_split']['std'], stats['stratified_split']['std']

    print(f"""
    1. Сырые данные: {raw_mean:.4f} ± {stats['raw_data']['std']:.4f} (завышенная оценка)
    2. Обычное разбиение: {random_mean:.4f} ± {random_std:.4f}
    3. Стратифицированное разбиение: {strat_mean:.4f} ± {strat_std:.4f}
    """)

    if strat_std < random_std:
        print(f"   ✅ Стратификация дала меньшую дисперсию: ↓{random_std - strat_std:.4f}")
    else:
        print(f"   ⚠️ Обычное разбиение дало меньшую дисперсию: ↓{strat_std - random_std:.4f}")

    if strat_mean > random_mean:
        print(f"   ✅ Стратификация показала лучшую точность: +{strat_mean - random_mean:.4f}")
    elif random_mean > strat_mean:
        print(f"   ⚠️ Обычное разбиение показало лучшую точность: +{random_mean - strat_mean:.4f}")
    else:
        print(f"   📊 Оба метода показали одинаковую точность")

    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    check_training_data()
    main()
