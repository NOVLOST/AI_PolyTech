import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
import warnings
import os

warnings.filterwarnings('ignore')



def run_experiment_sklearn(X, y, n_runs=30, n_estimators=20, max_depth=5, random_state=42):
    results = {
        'raw_data': [],
        'random_split': [],
        'stratified_split': []
    }

    print("Запуск экспериментов:")
    for run in range(n_runs):
        print(f"  Итерация {run + 1}/{n_runs}", end='\r')
        current_seed = random_state + run if random_state else None

        # ===== 1. Сырые данные =====
        model_raw = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features='sqrt', random_state=current_seed, n_jobs=-1
        )
        model_raw.fit(X, y)
        results['raw_data'].append(accuracy_score(y, model_raw.predict(X)))

        # ===== 2. Обычное случайное разбиение =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=current_seed, shuffle=True
        )
        model_random = ExtraTreesClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features='sqrt', random_state=current_seed, n_jobs=-1
        )
        model_random.fit(X_train, y_train)
        results['random_split'].append(accuracy_score(y_test, model_random.predict(X_test)))

        # ===== 3. Стратифицированное разбиение =====
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=current_seed)
            train_idx, test_idx = next(sss.split(X, y))
            X_train_strat, X_test_strat = X[train_idx], X[test_idx]
            y_train_strat, y_test_strat = y[train_idx], y[test_idx]

            model_strat = ExtraTreesClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                max_features='sqrt', random_state=current_seed, n_jobs=-1
            )
            model_strat.fit(X_train_strat, y_train_strat)
            results['stratified_split'].append(accuracy_score(y_test_strat, model_strat.predict(X_test_strat)))
        except ValueError:
            results['stratified_split'].append(np.nan)

    print()
    return results


def calculate_statistics(results):
    stats = {}
    for key, values in results.items():
        valid_values = [v for v in values if not np.isnan(v)]
        if valid_values:
            stats[key] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values),
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'median': np.median(valid_values),
                'q1': np.percentile(valid_values, 25),
                'q3': np.percentile(valid_values, 75)
            }
        else:
            stats[key] = {m: np.nan for m in ['mean', 'std', 'min', 'max', 'median', 'q1', 'q3']}
    return stats


def print_results_table(stats):
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ".center(80))
    print("=" * 80)

    print(f"\n{'Метод':<35} {'Среднее':<12} {'Стд. откл.':<12} {'Min':<10} {'Max':<10}")
    print("-" * 80)

    method_names = {
        'raw_data': 'Сырые данные (обучение+тест)',
        'random_split': 'Обычное разбиение (80/20)',
        'stratified_split': 'Стратифицированное разбиение (80/20)'
    }

    for key, name in method_names.items():
        s = stats[key]
        if np.isnan(s['mean']):
            print(f"{name:<35} {'N/A':<12} {'N/A':<12}    {'N/A':<10}      {'N/A':<10}")
        else:
            print(f"{name:<35} {s['mean']:.4f} ± {s['std']:.4f}    "
                  f"{s['min']:.4f}      {s['max']:.4f}")

    print("-" * 80)


def plot_results_comparison(results, stats):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    data_to_plot = [results['raw_data'], results['random_split'], results['stratified_split']]
    labels = ['Сырые данные\n(обучение+тест)', 'Обычное разбиение\n(80/20)', 'Стратифицированное\nразбиение (80/20)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    ax1 = axes[0]
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    means = [stats[k]['mean'] if not np.isnan(stats[k]['mean']) else 0 for k in
             ['raw_data', 'random_split', 'stratified_split']]
    for i, mean in enumerate(means, 1):
        if mean > 0:
            ax1.scatter(i, mean, color='darkred', s=100, zorder=3, marker='D', edgecolors='black', linewidth=1.5)
            ax1.annotate(f'{mean:.3f}', xy=(i, mean), xytext=(i + 0.05, mean), fontsize=9, fontweight='bold')

    ax1.set_ylabel('Accuracy (Точность)', fontsize=12)
    ax1.set_title('Распределение точности при разных подходах', fontsize=12)
    ax1.set_ylim(0.7, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    ax2 = axes[1]
    x_pos = np.arange(3)
    stds_plot = [stats[k]['std'] if not np.isnan(stats[k]['std']) else 0 for k in
                 ['raw_data', 'random_split', 'stratified_split']]
    bars = ax2.bar(x_pos, means, yerr=stds_plot, capsize=10, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds_plot)):
        if mean > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('Accuracy (Точность)', fontsize=12)
    ax2.set_title('Средняя точность ± доверительный интервал', fontsize=12)
    ax2.set_ylim(0.7, 1.05)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    print("\n📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
    print("=" * 50)
    for cls, count in zip(unique, counts):
        percentage = count / len(y) * 100
        print(f"   Класс {cls}: {count} объектов ({percentage:.1f}%)")
    return dict(zip(unique, counts))


# ==========================================================
# ОСНОВНАЯ ПРОГРАММА
# ==========================================================

def main():
    print("=" * 80)
    print("ЭКСПЕРИМЕНТ: Сравнение подходов к обучению Extra Tree (scikit-learn)")
    print("=" * 80)

    # 1. Загрузка датасета из CSV файла
    print("\n1. Загрузка датасета из CSV файла...")

    # 🔧 УКАЖИТЕ ПУТЬ К ВАШЕМУ ФАЙЛУ
    csv_file_path = "disease_train.csv"

    if not os.path.exists(csv_file_path):
        print(f"❌ Ошибка: Файл '{csv_file_path}' не найден.")
        print("   Положите файл в папку со скриптом или укажите полный путь.")
        return

    try:
        # skiprows=1 пропускает строку заголовков. Если заголовков нет, поставьте skiprows=0
        data = np.loadtxt(csv_file_path, delimiter=',', skiprows=1)

        # Предполагаем, что последний столбец - это целевая переменная (y)
        X = data[:, :-1]
        y = data[:, -1]

    except Exception as e:
        print(f"❌ Ошибка при чтении CSV: {e}")
        print("   Убедитесь, что файл содержит только числа и разделён запятыми.")
        return

   
    y = (y > np.median(y)).astype(int)

    feature_names = [f"X{i}" for i in range(1, X.shape[1] + 1)]
    target_names = [f"Класс {int(cls)}" for cls in np.unique(y)]

    print(f"   ✅ Загружено: {X.shape[0]} объектов")
    print(f"   ✅ Признаков: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"   ✅ Классов: {len(target_names)} ({', '.join(target_names)})")

    # 2. Анализ распределения классов
    analyze_class_distribution(y)

    # 3. Параметры эксперимента
    n_runs = 30
    n_estimators = 20
    max_depth = 5

    print(f"\n2. Параметры эксперимента:")
    print(f"   - Количество запусков: {n_runs}")
    print(f"   - Количество деревьев: {n_estimators}")
    print(f"   - Максимальная глубина: {max_depth}")
    print(f"   - max_features: 'sqrt' (√{X.shape[1]} = {int(np.sqrt(X.shape[1]))})")

    # 4. Запуск эксперимента
    print("\n3. Запуск экспериментов...")
    results = run_experiment_sklearn(X, y, n_runs=n_runs,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth)

    # 5. Расчёт статистики
    stats = calculate_statistics(results)

    # 6. Вывод результатов
    print_results_table(stats)

    raw_mean = stats['raw_data']['mean']
    random_mean = stats['random_split']['mean']
    strat_mean = stats['stratified_split']['mean']
    random_std = stats['random_split']['std']
    strat_std = stats['stratified_split']['std']

    print(f"""
    1. Сырые данные (обучение на всех данных):
       - Точность: {raw_mean:.4f} ± {stats['raw_data']['std']:.4f}
       - ⚠️  ЗАВЫШЕННАЯ оценка (модель запомнила данные)

    2. Обычное разбиение (train_test_split):
       - Точность на отложенной выборке: {random_mean:.4f} ± {random_std:.4f}
       - Честная оценка обобщающей способности

    3. Стратифицированное разбиение (StratifiedShuffleSplit):
       - Точность на отложенной выборке: {strat_mean:.4f} ± {strat_std:.4f}
       - Сохраняет пропорции классов
    """)

   
    # plot_results_comparison(results, stats)


if __name__ == "__main__":
    main()
