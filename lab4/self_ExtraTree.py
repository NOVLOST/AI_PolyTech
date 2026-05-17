import sys
from typing import List, Tuple, Any
import numpy as np
import random
import math
from collections import Counter


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
        # Используем int для индексов
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

        # Определяем количество признаков для рассмотрения
        n_feats = self._get_n_features(n_features)

        # Случайный выбор признаков
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

            # Генерация случайных порогов
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
        # Преобразуем y в плоский массив int
        y = np.asarray(y).flatten()

        unique_classes = np.unique(y)

        # Условия остановки
        if len(unique_classes) == 1:
            return Node(prediction=int(unique_classes[0]))

        if depth >= self.max_depth:
            y_int = y.astype(int)
            max_label = int(np.max(y_int)) + 1
            most_common = np.bincount(y_int, minlength=max_label).argmax()
            return Node(prediction=int(most_common))

        if len(y) < self.min_samples_split:
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
        # Преобразуем входные данные в numpy массивы
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
# ФУНКЦИИ ДЛЯ РАЗБИЕНИЯ ДАННЫХ (ТОЛЬКО СТАНДАРТНАЯ БИБЛИОТЕКА)
# ==========================================================

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    """
    Ручная реализация train_test_split без sklearn
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indices = list(range(n_samples))
    random.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def stratified_shuffle_split_manual(X, y, test_size=0.2, random_state=None):
    """
    Ручная реализация стратифицированного разбиения без sklearn
    Сохраняет пропорции классов в train и test
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    # Получаем уникальные классы и их индексы
    y = np.asarray(y).flatten()
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []

    for cls in unique_classes:
        # Индексы объектов данного класса
        cls_indices = np.where(y == cls)[0].tolist()
        random.shuffle(cls_indices)

        # Сколько объектов этого класса должно попасть в тест
        n_cls = len(cls_indices)
        n_cls_test = int(n_cls * test_size)

        # Добавляем в test и train
        test_indices.extend(cls_indices[:n_cls_test])
        train_indices.extend(cls_indices[n_cls_test:])

    # Перемешиваем финальные списки
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


# ==========================================================
# ОСНОВНОЙ ЭКСПЕРИМЕНТ
# ==========================================================

def run_experiment(X, y, n_runs=30, n_estimators=20, max_depth=5):
    """
    Запуск эксперимента с тремя вариациями обучения

    Возвращает словарь с результатами accuracy для каждого подхода
    """
    results = {
        'raw_data': [],
        'random_split': [],
        'stratified_split': []
    }

    for run in range(n_runs):
        print(f"  Итерация {run + 1}/{n_runs}", end='\r')
        sys.stdout.flush()

        # ===== 1. Сырые данные (обучение и тест на всех данных) =====
        model_raw = ExtraTreeClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            n_random_splits=10,
            random_state=run
        )
        model_raw.fit(X, y)
        accuracy_raw = model_raw.score(X, y)
        results['raw_data'].append(accuracy_raw)

        # ===== 2. Метод A: Обычное случайное разбиение =====
        X_train, X_test, y_train, y_test = train_test_split_manual(
            X, y, test_size=0.2, random_state=run
        )
        model_random = ExtraTreeClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            n_random_splits=10,
            random_state=run
        )
        model_random.fit(X_train, y_train)
        accuracy_random = model_random.score(X_test, y_test)
        results['random_split'].append(accuracy_random)

        # ===== 3. Метод B: Стратифицированное разбиение =====
        X_train_strat, X_test_strat, y_train_strat, y_test_strat = stratified_shuffle_split_manual(
            X, y, test_size=0.2, random_state=run
        )
        model_strat = ExtraTreeClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            n_random_splits=10,
            random_state=run
        )
        model_strat.fit(X_train_strat, y_train_strat)
        accuracy_strat = model_strat.score(X_test_strat, y_test_strat)
        results['stratified_split'].append(accuracy_strat)

    print()  # переводим строку после прогресс-бара
    return results


def calculate_statistics(results):
    """Вычисляет среднее и стандартное отклонение для каждого подхода"""
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
    """Построение boxplot с результатами"""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Подготовка данных для boxplot
        data_to_plot = [
            results['raw_data'],
            results['random_split'],
            results['stratified_split']
        ]

        labels = [
            f'Сырые данные\n(обучение+тест)\nμ={stats["raw_data"]["mean"]:.4f}±{stats["raw_data"]["std"]:.4f}',
            f'Обычное разбиение\n(80/20)\nμ={stats["random_split"]["mean"]:.4f}±{stats["random_split"]["std"]:.4f}',
            f'Стратифицированное\nразбиение\nμ={stats["stratified_split"]["mean"]:.4f}±{stats["stratified_split"]["std"]:.4f}'
        ]

        # Создаём boxplot
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        # Настройка цветов
        colors = ['#FF9999', '#99FF99', '#9999FF']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Добавляем средние значения точками
        for i, (key, data) in enumerate(results.items(), 1):
            means = stats[key]['mean']
            ax.scatter(i, means, color='red', s=100, zorder=3,
                       marker='D', label='Среднее' if i == 1 else "")

        ax.set_ylabel('Accuracy (Точность)', fontsize=12)
        ax.set_title('Сравнение точности Extra Tree при разных подходах к обучению\n'
                     f'({len(results["raw_data"])} запусков)',
                     fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

        # Добавляем легенду
        ax.legend(loc='lower right')

        # Добавляем аннотацию с интерпретацией
        ax.text(0.02, 0.02,
                'Интерпретация:\n'
                '• Сырые данные завышают оценку (переобучение)\n'
                '• Разбиение даёт более честную оценку\n'
                '• Стратификация снижает дисперсию',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\n⚠️ Matplotlib не установлен. Пропускаем построение графика.")
        print("Установите matplotlib: pip install matplotlib")


# ==========================================================
# ЗАПУСК ЭКСПЕРИМЕНТА
# ==========================================================

def main():
    print("=" * 70)
    print("ЭКСПЕРИМЕНТ: Сравнение подходов к обучению Extra Tree")
    print("=" * 70)

    # Загрузка датасета Iris
    print("\n1. Загрузка датасета Iris...")
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"   ✅ Загружено: {X.shape[0]} объектов, {X.shape[1]} признаков, {len(np.unique(y))} классов")

    # Параметры эксперимента
    n_runs = 30
    n_estimators = 20
    max_depth = 5

    print(f"\n2. Параметры эксперимента:")
    print(f"   - Количество запусков: {n_runs}")
    print(f"   - Количество деревьев: {n_estimators}")
    print(f"   - Максимальная глубина: {max_depth}")
    print(f"   - max_features: 'sqrt' (√{X.shape[1]} = {int(np.sqrt(X.shape[1]))})")

    # Запуск эксперимента
    print("\n3. Запуск экспериментов...")
    results = run_experiment(X, y, n_runs=n_runs, n_estimators=n_estimators, max_depth=max_depth)

    # Расчёт статистики
    print("\n4. Расчёт статистики...")
    stats = calculate_statistics(results)

    # Вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 70)

    print("\n📊 Средняя точность ± стандартное отклонение (по 30 запускам):\n")

    labels_names = {
        'raw_data': 'Сырые данные (обучение+тест на всех)',
        'random_split': 'Обычное разбиение (80/20)',
        'stratified_split': 'Стратифицированное разбиение (80/20)'
    }

    for key in results.keys():
        print(f"   {labels_names[key]:35}: {stats[key]['mean']:.4f} ± {stats[key]['std']:.4f}")
        print(f"                                      (min: {stats[key]['min']:.4f}, max: {stats[key]['max']:.4f})")
        print()

    # Построение графика
    print("\n5. Построение графика...")
    plot_results(results, stats)

    # Итоговые выводы
    print("\n" + "=" * 70)
    print("ВЫВОДЫ")
    print("=" * 70)

    raw_mean = stats['raw_data']['mean']
    random_mean = stats['random_split']['mean']
    strat_mean = stats['stratified_split']['mean']
    random_std = stats['random_split']['std']
    strat_std = stats['stratified_split']['std']

    print(f"""
    1. Сырые данные (обучение на всех данных):
       - Точность: {raw_mean:.4f} ± {stats['raw_data']['std']:.4f}
       - Это ЗАВЫШЕННАЯ оценка (модель просто запомнила данные)
       - Не подходит для реальной оценки качества

    2. Обычное разбиение (train_test_split):
       - Точность на отложенной выборке: {random_mean:.4f} ± {random_std:.4f}
       - Более честная оценка обобщающей способности
       - Дисперсия: {random_std:.4f}

    3. Стратифицированное разбиение (StratifiedShuffleSplit):
       - Точность на отложенной выборке: {strat_mean:.4f} ± {strat_std:.4f}
       - Сохраняет пропорции классов
       - Дисперсия: {strat_std:.4f}
    """)

    if strat_std < random_std:
        print(f"   ✅ Стратифицированное разбиение дало МЕНЬШУЮ дисперсию:")
        print(f"      Уменьшение std на {random_std - strat_std:.4f}")
    else:
        print(f"   ⚠️ Обычное разбиение дало меньшую дисперсию:")
        print(f"      Разница std: {strat_std - random_std:.4f}")

    if strat_mean > random_mean:
        print(f"\n   ✅ Стратифицированное разбиение показало ЛУЧШУЮ среднюю точность:")
        print(f"      +{strat_mean - random_mean:.4f} к accuracy")
    elif random_mean > strat_mean:
        print(f"\n   ⚠️ Обычное разбиение показало лучшую точность:")
        print(f"      +{random_mean - strat_mean:.4f} к accuracy")
    else:
        print(f"\n   📊 Оба метода показали одинаковую точность")

    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 70)


if __name__ == "__main__":
    main()


