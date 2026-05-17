import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


# ==========================================================
# ФУНКЦИИ ДЛЯ ЭКСПЕРИМЕНТОВ
# ==========================================================

def run_experiment_sklearn(X, y, n_runs=30, n_estimators=20, max_depth=5, random_state=42):
    """
    Запуск эксперимента с тремя вариациями обучения используя sklearn

    Параметры:
        X: признаки
        y: целевая переменная
        n_runs: количество запусков
        n_estimators: количество деревьев в ансамбле
        max_depth: максимальная глубина деревьев
        random_state: базовое значение для воспроизводимости

    Возвращает:
        results: словарь с результатами accuracy для каждого подхода
    """

    results = {
        'raw_data': [],  # обучение и тест на всех данных
        'random_split': [],  # обычное случайное разбиение
        'stratified_split': []  # стратифицированное разбиение
    }

    print("Запуск экспериментов:")
    for run in range(n_runs):
        print(f"  Итерация {run + 1}/{n_runs}", end='\r')

        # Устанавливаем seed для воспроизводимости
        current_seed = random_state + run if random_state else None

        # ===== 1. Сырые данные (обучение и тест на всех данных) =====
        model_raw = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            random_state=current_seed,
            n_jobs=-1  # используем все ядра процессора
        )
        model_raw.fit(X, y)
        accuracy_raw = accuracy_score(y, model_raw.predict(X))
        results['raw_data'].append(accuracy_raw)

        # ===== 2. Метод A: Обычное случайное разбиение =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=current_seed,
            shuffle=True  # перемешиваем данные
        )

        model_random = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            random_state=current_seed,
            n_jobs=-1
        )
        model_random.fit(X_train, y_train)
        accuracy_random = accuracy_score(y_test, model_random.predict(X_test))
        results['random_split'].append(accuracy_random)

        # ===== 3. Метод B: Стратифицированное разбиение =====
        sss = StratifiedShuffleSplit(
            n_splits=1,  # одно разбиение
            test_size=0.2,  # 20% на тест
            random_state=current_seed
        )

        # Получаем единственное разбиение
        train_idx, test_idx = next(sss.split(X, y))
        X_train_strat, X_test_strat = X[train_idx], X[test_idx]
        y_train_strat, y_test_strat = y[train_idx], y[test_idx]

        model_strat = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            random_state=current_seed,
            n_jobs=-1
        )
        model_strat.fit(X_train_strat, y_train_strat)
        accuracy_strat = accuracy_score(y_test_strat, model_strat.predict(X_test_strat))
        results['stratified_split'].append(accuracy_strat)

    print()  # переводим строку после прогресс-бара
    return results


def calculate_statistics(results):
    """Вычисляет статистические метрики для каждого подхода"""
    stats = {}
    for key, values in results.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q1': np.percentile(values, 25),  # первый квартиль
            'q3': np.percentile(values, 75)  # третий квартиль
        }
    return stats


def print_results_table(stats):
    """Выводит результаты в виде красивой таблицы"""
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ".center(80))
    print("=" * 80)

    # Заголовки
    print(f"\n{'Метод':<35} {'Среднее':<12} {'Стд. откл.':<12} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    # Названия методов
    method_names = {
        'raw_data': 'Сырые данные (обучение+тест)',
        'random_split': 'Обычное разбиение (80/20)',
        'stratified_split': 'Стратифицированное разбиение (80/20)'
    }

    for key, name in method_names.items():
        s = stats[key]
        print(f"{name:<35} {s['mean']:.4f} ± {s['std']:.4f}    "
              f"{s['min']:.4f}      {s['max']:.4f}")

    print("-" * 80)


def plot_results_comparison(results, stats):
    """Строит улучшенный график сравнения результатов"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Boxplot (слева)
    ax1 = axes[0]
    data_to_plot = [results['raw_data'], results['random_split'], results['stratified_split']]

    labels = [
        f'Сырые данные\n(обучение+тест)',
        f'Обычное разбиение\n(80/20)',
        f'Стратифицированное\nразбиение (80/20)'
    ]

    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Настройка цветов
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Добавляем средние значения
    means = [stats['raw_data']['mean'], stats['random_split']['mean'], stats['stratified_split']['mean']]
    for i, mean in enumerate(means, 1):
        ax1.scatter(i, mean, color='darkred', s=100, zorder=3,
                    marker='D', edgecolors='black', linewidth=1.5)
        ax1.annotate(f'{mean:.3f}', xy=(i, mean), xytext=(i + 0.05, mean),
                     fontsize=9, fontweight='bold')

    ax1.set_ylabel('Accuracy (Точность)', fontsize=12)
    ax1.set_title('Распределение точности при разных подходах\n'
                  f'({len(results["raw_data"])} запусков)', fontsize=12)
    ax1.set_ylim(0.7, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Добавляем аннотацию
    ax1.text(0.02, 0.02,
             'Интерпретация:\n'
             '• Сырые данные завышают оценку\n'
             '• Разбиение даёт честную оценку\n'
             '• Стратификация ↓ дисперсию',
             transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Линейный график с доверительными интервалами (справа)
    ax2 = axes[1]

    x_pos = np.arange(3)
    means_plot = [stats['raw_data']['mean'], stats['random_split']['mean'], stats['stratified_split']['mean']]
    stds_plot = [stats['raw_data']['std'], stats['random_split']['std'], stats['stratified_split']['std']]

    bars = ax2.bar(x_pos, means_plot, yerr=stds_plot, capsize=10,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Добавляем значения на столбцы
    for i, (bar, mean, std) in enumerate(zip(bars, means_plot, stds_plot)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Сырые данные\n(обучение+тест)',
                         'Обычное\nразбиение',
                         'Стратифицированное\nразбиение'], fontsize=9)
    ax2.set_ylabel('Accuracy (Точность)', fontsize=12)
    ax2.set_title('Средняя точность ± доверительный интервал (95%)', fontsize=12)
    ax2.set_ylim(0.7, 1.05)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_class_distribution(y):
    """Анализирует распределение классов в данных"""
    unique, counts = np.unique(y, return_counts=True)
    print("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
    print("=" * 50)
    for cls, count in zip(unique, counts):
        percentage = count / len(y) * 100
        print(f"   Класс {cls}: {count} объектов ({percentage:.1f}%)")
    return dict(zip(unique, counts))


def compare_splitting_methods_demo(X, y):
    """Демонстрация разницы между обычным и стратифицированным разбиением"""
    print("\n" + "=" * 80)
    print("ДЕМОНСТРАЦИЯ РАЗНИЦЫ МЕТОДОВ РАЗБИЕНИЯ".center(80))
    print("=" * 80)

    # 5 запусков для демонстрации
    n_demo = 5
    random_props = []
    strat_props = []

    for i in range(n_demo):
        # Обычное разбиение
        _, y_test_rand, _, _ = train_test_split(X, y, test_size=0.2, random_state=i)
        rand_class0_prop = np.sum(y_test_rand == 0) / len(y_test_rand)
        random_props.append(rand_class0_prop)

        # Стратифицированное разбиение
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        train_idx, test_idx = next(sss.split(X, y))
        y_test_strat = y[test_idx]
        strat_class0_prop = np.sum(y_test_strat == 0) / len(y_test_strat)
        strat_props.append(strat_class0_prop)

    true_prop = np.sum(y == 0) / len(y)

    print(f"\nИсходная доля класса 0: {true_prop:.3f} ({true_prop * 100:.1f}%)\n")
    print("Запуск | Обычное разбиение | Стратифицированное")
    print("-" * 50)
    for i in range(n_demo):
        print(f"  {i + 1:2d}   |     {random_props[i]:.3f}        |        {strat_props[i]:.3f}")

    print(f"\nСреднее отклонение от исходной доли:")
    print(f"  Обычное разбиение:      {np.mean(np.abs(np.array(random_props) - true_prop)):.4f}")
    print(f"  Стратифицированное:     {np.mean(np.abs(np.array(strat_props) - true_prop)):.4f}")


# ==========================================================
# ОСНОВНАЯ ПРОГРАММА
# ==========================================================

def main():
    print("=" * 80)
    print("ЭКСПЕРИМЕНТ: Сравнение подходов к обучению Extra Tree (scikit-learn)")
    print("=" * 80)

    # 1. Загрузка датасета Iris
    print("\n1. Загрузка датасета Iris...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"   ✅ Загружено: {X.shape[0]} объектов")
    print(f"   ✅ Признаков: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"   ✅ Классов: {len(target_names)} ({', '.join(target_names)})")

    # 2. Анализ распределения классов
    class_dist = analyze_class_distribution(y)

    # 3. Демонстрация разницы методов разбиения
    compare_splitting_methods_demo(X, y)

    # 4. Параметры эксперимента
    n_runs = 30
    n_estimators = 20
    max_depth = 5

    print(f"\n2. Параметры эксперимента:")
    print(f"   - Количество запусков: {n_runs}")
    print(f"   - Количество деревьев: {n_estimators}")
    print(f"   - Максимальная глубина: {max_depth}")
    print(f"   - max_features: 'sqrt' (√{X.shape[1]} = {int(np.sqrt(X.shape[1]))})")

    # 5. Запуск эксперимента
    print("\n3. Запуск экспериментов...")
    results = run_experiment_sklearn(X, y, n_runs=n_runs,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)

    # 6. Расчёт статистики
    stats = calculate_statistics(results)

    # 7. Вывод результатов
    print_results_table(stats)

    # 8. Детальный анализ результатов
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ".center(80))
    print("=" * 80)

    print("\n📊 Статистика по каждому методу:")
    for key in stats.keys():
        method_name = {
            'raw_data': 'Сырые данные',
            'random_split': 'Обычное разбиение',
            'stratified_split': 'Стратифицированное разбиение'
        }[key]

        print(f"\n   {method_name}:")
        print(f"      Медиана: {stats[key]['median']:.4f}")
        print(f"      Q1 (25-й перцентиль): {stats[key]['q1']:.4f}")
        print(f"      Q3 (75-й перцентиль): {stats[key]['q3']:.4f}")
        print(f"      IQR (межквартильный размах): {stats[key]['q3'] - stats[key]['q1']:.4f}")

    # 9. Сравнительный анализ
    print("\n" + "=" * 80)
    print("ВЫВОДЫ".center(80))
    print("=" * 80)

    raw_mean = stats['raw_data']['mean']
    random_mean = stats['random_split']['mean']
    strat_mean = stats['stratified_split']['mean']
    random_std = stats['random_split']['std']
    strat_std = stats['stratified_split']['std']

    print(f"""
    1. Сырые данные (обучение на всех данных):
       - Точность: {raw_mean:.4f} ± {stats['raw_data']['std']:.4f}
       - ⚠️  ЗАВЫШЕННАЯ оценка (модель запомнила данные)
       - Не подходит для оценки реального качества

    2. Обычное разбиение (train_test_split):
       - Точность на отложенной выборке: {random_mean:.4f} ± {random_std:.4f}
       - Честная оценка обобщающей способности
       - Дисперсия: {random_std:.4f}

    3. Стратифицированное разбиение (StratifiedShuffleSplit):
       - Точность на отложенной выборке: {strat_mean:.4f} ± {strat_std:.4f}
       - Сохраняет пропорции классов
       - Дисперсия: {strat_std:.4f}
    """)

    # Сравнение методов
    print("📈 СРАВНЕНИЕ МЕТОДОВ:")
    print("-" * 50)

    if strat_std < random_std:
        improvement = random_std - strat_std
        print(f"   ✅ Стратифицированное разбиение уменьшило дисперсию на {improvement:.4f}")
    else:
        print(f"   ⚠️  Обычное разбиение дало меньшую дисперсию")

    if strat_mean > random_mean:
        improvement = strat_mean - random_mean
        print(f"   ✅ Стратифицированное разбиение повысило точность на {improvement:.4f}")

    # Эффект от стратификации
    print(f"\n   💡 Эффект стратификации:")
    print(f"      - Улучшение точности: {(strat_mean - random_mean) * 100:.2f}%")
    print(f"      - Снижение дисперсии: {(random_std - strat_std) * 100:.2f}%")

    # 10. Построение графика
    print("\n4. Построение графика...")
    plot_results_comparison(results, stats)

    print("\n" + "=" * 80)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
