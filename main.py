import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analytics.distributions import GenerateFunctions
from analytics.plotting import PlottingFunctions
from analytics.censoring import CensoringFunctions
from analytics.kaplan_meyer import KaplanMeyerFunctions
from analytics.kde import KernelDensityEstimation
from analytics.utils import DataUtils

from typing import Dict


def main(
        n: int,
        distribution_name: str,
        threshold: float = 100.0,
        deep_coeff: float = 2 / 3,
        l_share: float = 0.1,
        i_share: float = 0.1,
        r_share: float = 0.1,
        **kwargs: Dict[str, float]
) -> None:
    """
    Функция выполняет 4 последовательных этапа:
    1. Генерирует выборку
    2. Цензурирует данные
    3. Строит оценку Каплана-Мейера
    4. Строит ядерную оценку

    :param n: Количество элементов в выборке
    :param distribution_name:
           'lognormal' - Логнормальное распределение
           'weibull' - Распределение Вейбулла
           'gamma' - Гамма-распределение
    :param kwargs: Параметры распределения (например, shape, scale, loc).
    :param r_share: Доля правостороннего цензурирования
    :param i_share: Доля интервального цензурирования
    :param l_share: Доля левостороннего цензурирования
    :param threshold: Порог цензурирования
    :param deep_coeff: Глубина цензурирования
    """
    # Устанавливаем опции печати
    np.set_printoptions(suppress=True)
    # Устанавливаем опции для отображения всех строк и столбцов
    pd.set_option('display.max_columns', None)

    # --------- КОНСТАНТЫ ----------
    ALPHA_FOR_GREENWOOD = 0.05              # Уровень значимости ДИ Гринвуда
    NUM_POINTS_FOR_KDE = 100                # Количество точек для расчёта KDE

    # ----- Экземпляры классов -----
    generator = GenerateFunctions()
    plotter = PlottingFunctions()
    censor = CensoringFunctions()
    km = KaplanMeyerFunctions()
    kde = KernelDensityEstimation()
    data_utils = DataUtils()

    # 1. Получаем выборку из разных распределений
    supported_distributions = ['lognormal', 'weibull', 'gamma']
    if distribution_name not in supported_distributions:
        raise ValueError(
            f"Unsupported distribution: {distribution_name}. "
            f"Valid options: {supported_distributions}"
        )

    data = generator.generate_sample(n=n, distribution=distribution_name, **kwargs)
    plotter.paint_distribution_hist(data, distribution_name)
    plt.show()

    print(f"Сгенерированная выборка:\n{data}")

    plotter.paint_distribution_time_life(data)
    plt.show()

    # 2. Цензурирование выборки разными способами
    # Цензурируем по верхнему и нижнему порогам
    data = censor.threshold_censoring(data, threshold=threshold)
    # Цензурируем долю случайных значений (случайно распределённых левых и правых) от всех
    data = censor.random_censoring(data, l_share=l_share, i_share=i_share, r_share=r_share, threshold=threshold, coeff=deep_coeff)

    print(f"Цензурированная выборка:\n{data}")

    plotter.paint_distribution_time_life(data)
    plt.show()

    # 3. Строим оценку Каплана-Майера
    s_mark = km.kaplan_mayer_mark(data)
    print(f"Оценка Каплана-Мейера:\n{s_mark}")

    if not s_mark.empty and len(s_mark) > 1:
        # Получаем для оценки Каплана-Майера доверительный интервал по формуле Гринвуда
        confidence_interval = km.greenwood_confidence_interval(data, s_mark, ALPHA_FOR_GREENWOOD)

        # Сравниваем с реальным значением
        plotter.paint_confidence_interval_with_kaplan_mayer(confidence_interval, s_mark)
        x_mass, true_value = plotter.add_distribution_on_plot(distribution_name, max_size=max(s_mark.index), **kwargs)

        # Считаем значение метрик:
        plotter.get_metrics_for_kaplan(x_mass, s_mark, true_value)
        plt.show()

    # Получаем максимальный размер выборки
    if s_mark.empty or len(s_mark) == 1:
        max_sample_size = data_utils.get_max_size(data)
    else:
        max_sample_size = max(s_mark.index)

    # 4. Строим ядерную оценку плотности
    # Считаем значения x, y и 2ой производной для конкретных ядерных оценок
    x_gauss, y_gauss, sd_gauss = kde.calculate_kde(data, 'gauss', max_sample_size, NUM_POINTS_FOR_KDE)
    x_unif, y_unif, sd_unif = kde.calculate_kde(data, 'uniform', max_sample_size, NUM_POINTS_FOR_KDE)
    x_epach, y_epach, sd_epach = kde.calculate_kde(data, 'epachnikov', max_sample_size, NUM_POINTS_FOR_KDE)
    x_bi, y_bi, sd_bi = kde.calculate_kde(data, 'bisquare', max_sample_size, NUM_POINTS_FOR_KDE)

    # Рисуем ядерные оценки
    plotter.paint_kde(x_gauss, y_gauss, 'Gaussian Kernel Distribution Function', color="red")
    plotter.paint_kde(x_unif, y_unif, 'Uniform Kernel Distribution Function', color="orange")
    plotter.paint_kde(x_epach, y_epach, 'Epachnikov Kernel Distribution Function', color="cyan")
    plotter.paint_kde(x_bi, y_bi, 'Bisquare Kernel Distribution Function', color="green")

    # Добавляем теоретическую плотность
    x_mass, true_value = plotter.add_distribution_pdf_on_plot(distribution_name, max_size=max_sample_size, **kwargs, points_number=NUM_POINTS_FOR_KDE)

    plotter.get_metrics_for_kde(x_mass, y_gauss, true_value, "Gaussian", color="red")
    plotter.get_metrics_for_kde(x_mass, y_unif, true_value, "Uniform", color="orange")
    plotter.get_metrics_for_kde(x_mass, y_epach, true_value, "Epachnikov", color="cyan")
    plotter.get_metrics_for_kde(x_mass, y_bi, true_value, "Bisquare", color="green")
    plt.show()

    # 5. Интегрируем ядерную оценку и сравниваем с теоретической функцией.
    y_gauss_integral = plotter.paint_integrated_kernel_mark(x_gauss, y_gauss, sd_gauss, "Интеграл от ядерной оценки с Гауссовским ядром", color="red")
    y_unif_integral = plotter.paint_integrated_kernel_mark(x_unif, y_unif, sd_unif, "Интеграл от ядерной оценки с Прямоугольным ядром", color="orange")
    y_epach_integral = plotter.paint_integrated_kernel_mark(x_epach, y_epach, sd_epach, "Интеграл от ядерной оценки с ядром Епанечникова", color="cyan")
    y_bi_integral = plotter.paint_integrated_kernel_mark(x_bi, y_bi, sd_bi, "Интеграл от ядерной оценки с Биквадратным ядром", color="green")

    x_mass, true_value = plotter.add_distribution_on_plot(distribution_name, max_size=max_sample_size, **kwargs, points_number=NUM_POINTS_FOR_KDE)

    plotter.get_metrics_for_kde(x_mass, y_gauss_integral, true_value, "Gaussian", color="red")
    plotter.get_metrics_for_kde(x_mass, y_unif_integral, true_value, "Uniform", color="orange")
    plotter.get_metrics_for_kde(x_mass, y_epach_integral, true_value, "Epachnikov", color="cyan")
    plotter.get_metrics_for_kde(x_mass, y_bi_integral, true_value, "Bisquare", color="green")
    plt.show()


def research(
        n: int,
        distribution_name: str,
        threshold: float = 100.0,
        l_share: float = 0.1,
        i_share: float = 0.1,
        r_share: float = 0.1,
        **kwargs: Dict[str, float]
) -> None:
    """
        Функция производит исследования зависимости значения метрик от различных параметров

        :param n: Количество элементов в выборке
        :param distribution_name:
               'lognormal' - Логнормальное распределение
               'weibull' - Распределение Вейбулла
               'gamma' - Гамма-распределение
        :param kwargs: Параметры распределения (например, shape, scale, loc).
        :param r_share: Доля правостороннего цензурирования
        :param i_share: Доля интервального цензурирования
        :param l_share: Доля левостороннего цензурирования
        :param threshold: Порог цензурирования
        """
    # Функции печати
    np.set_printoptions(suppress=True)
    pd.set_option('display.max_columns', None)  # Показывать все столбцы

    # --------- КОНСТАНТЫ ----------
    NUM_POINTS_FOR_KDE = 100

    # ----- Экземпляры классов -----
    generator = GenerateFunctions()
    plotter = PlottingFunctions()
    censor = CensoringFunctions()
    km = KaplanMeyerFunctions()
    kde = KernelDensityEstimation()
    data_utils = DataUtils()

    # Получаем выборку из разных распределений
    supported_distributions = ['lognormal', 'weibull', 'gamma']
    if distribution_name not in supported_distributions:
        raise ValueError(
            f"Unsupported distribution: {distribution_name}. "
            f"Valid options: {supported_distributions}"
        )

    # # 1. Зависимость от доли цензурирования
    # Все виды
    for cens in range(0, 70, 5):
        cens /= 200
        l_share = cens
        i_share = cens
        r_share = cens
        deep_cens = 2 / 3
    # Правостороннее
    # for cens in range(0, 100, 5):
    #     cens /= 100
    #     l_share = 0.0
    #     i_share = 0.0
    #     r_share = cens
    #     deep_cens = 2 / 3

    # # 2. Зависимость от порога цензурирования
    # Вейбулл
    # for thrsh in range(12, 0, -1):
    #     thrsh /= 2
    #     threshold = thrsh
    #     deep_cens = 2 / 3
    # Логнормальное
    # for thrsh in range(20, 0, -1):
    #     thrsh /= 2
    #     threshold = thrsh
    #     deep_cens = 2 / 3
    # Гамма
    # for thrsh in range(35, 0, -1):
    #     threshold = thrsh
    #     deep_cens = 2 / 3

    # # 3. Зависимость от вида цензурирования
    # for cens in range(0, 10, 1):
    #     i_share = cens / 10
    #     r_share = 0.45 - cens / 20
    #     l_share = 0.45 - cens / 20
    #     deep_cens = 2 / 3

    # # 4. Зависимость от глубины цензурирования
    # for deep_cens in range(0, 100, 5):
    #     deep_cens /= 100

        mean_mmd_km = 0
        mean_smse_km = 0

        mean_mmd_g = 0
        mean_mmd_u = 0
        mean_mmd_e = 0
        mean_mmd_b = 0

        mean_smse_g = 0
        mean_smse_u = 0
        mean_smse_e = 0
        mean_smse_b = 0

        for i in range(50):
            # Генерируем
            data = generator.generate_sample(n, distribution=distribution_name, **kwargs)

            # Цензурируем
            data = censor.threshold_censoring(data, threshold=threshold)
            data = censor.random_censoring(data, l_share=l_share, i_share=i_share, r_share=r_share, threshold=threshold, coeff=deep_cens)

            # Строим оценку Каплана-Майера
            s_mark = km.kaplan_mayer_mark(data)

            if not s_mark.empty and len(s_mark) > 1:
                # Сравниваем с реальным значением
                x_mass, true_value = plotter.add_distribution_on_plot(distribution_name, max_size=max(s_mark.index),
                                                                      is_research=True, **kwargs)
                # Считаем значение метрик:
                mmd_km, smse_km = plotter.get_metrics_for_kaplan(x_mass, s_mark, true_value, is_research=True)

            if s_mark.empty or len(s_mark) == 1:
                max_sample_size = data_utils.get_max_size(data)
            else:
                max_sample_size = max(s_mark.index)

            # Строим ядерную оценку плотности
            x_gauss, y_gauss, sd_gauss = kde.calculate_kde(data, 'gauss', max_sample_size, NUM_POINTS_FOR_KDE)
            x_unif, y_unif, sd_unif = kde.calculate_kde(data, 'uniform', max_sample_size, NUM_POINTS_FOR_KDE)
            x_epach, y_epach, sd_epach = kde.calculate_kde(data, 'epachnikov', max_sample_size, NUM_POINTS_FOR_KDE)
            x_bi, y_bi, sd_bi = kde.calculate_kde(data, 'bisquare', max_sample_size, NUM_POINTS_FOR_KDE)

            # Интегрируем ядерную оценку и сравниваем с теоретической функцией.
            y_gauss_integral = plotter.paint_integrated_kernel_mark(x_gauss, y_gauss, sd_gauss, "Интеграл от ядерной оценки с Гауссовским ядром", color="red", is_research=True)
            y_unif_integral = plotter.paint_integrated_kernel_mark(x_unif, y_unif, sd_unif, "Интеграл от ядерной оценки с Прямоугольным ядром", color="orange", is_research=True)
            y_epach_integral = plotter.paint_integrated_kernel_mark(x_epach, y_epach, sd_epach, "Интеграл от ядерной оценки с ядром Епанечникова", color="cyan", is_research=True)
            y_bi_integral = plotter.paint_integrated_kernel_mark(x_bi, y_bi, sd_bi, "Интеграл от ядерной оценки с Биквадратным ядром", color="green", is_research=True)

            x_mass, true_value = plotter.add_distribution_on_plot(distribution_name, max_size=max_sample_size, points_number=NUM_POINTS_FOR_KDE, is_research=True, **kwargs)

            mmd_g, smse_g = plotter.get_metrics_for_kde(x_mass, y_gauss_integral, true_value, "Gaussian", color="red", is_research=True)
            mmd_u, smse_u = plotter.get_metrics_for_kde(x_mass, y_unif_integral, true_value, "Uniform", color="orange", is_research=True)
            mmd_e, smse_e = plotter.get_metrics_for_kde(x_mass, y_epach_integral, true_value, "Epachnikov", color="cyan", is_research=True)
            mmd_b, smse_b = plotter.get_metrics_for_kde(x_mass, y_bi_integral, true_value, "Bisquare", color="green", is_research=True)

            # Вычисляем метрики
            mean_mmd_km += mmd_km
            mean_smse_km += smse_km

            mean_mmd_g += mmd_g
            mean_mmd_u += mmd_u
            mean_mmd_e += mmd_e
            mean_mmd_b += mmd_b

            mean_smse_g += smse_g
            mean_smse_u += smse_u
            mean_smse_e += smse_e
            mean_smse_b += smse_b

        print(f"{round(mean_mmd_km / 50, 3)} {round(mean_mmd_g / 50, 3)} {round(mean_mmd_u / 50, 3)} {round(mean_mmd_e / 50, 3)} {round(mean_mmd_b / 50, 3)}", end=" ")
        print(f"{round(mean_smse_km / 50, 3)} {round(mean_smse_g / 50, 3)} {round(mean_smse_u / 50, 3)} {round(mean_smse_e / 50, 3)} {round(mean_smse_b / 50, 3)}", end=" ")

        # 1. Зависимость от Доли цензурирования
        print(f" - cens = {cens}")
        # 2. Зависимость от Порога цензурирования
        # print(f" - cens = {threshold}")
        # 3. Зависимость от Типа цензурирования
        # print(f" - Curr_cens = {cens / 10}, others = {0.45 - cens / 20}")
        # 4. Зависимость от Глубины цензурирования
        # print(f" - deep_cens = {deep_cens}")


if __name__ == '__main__':
    research_flag = False
    if not research_flag:
        main(20, "weibull", shape=3, scale=2, loc=0, l_share=0.2, i_share=0.2, r_share=0.2, threshold=50, deep_coeff=1/3)
    else:
        research(100, "weibull", shape=3, scale=2, loc=0, l_share=0.0, i_share=0.0, r_share=0.0)

# Примеры использования
# main(100, "lognormal", mean=0, std_dev=1, threshold=100, l_share=0.1, i_share=0.1, r_share=0.1)
# main(100, "weibull", shape=3, scale=2, loc=0, threshold=100, l_share=0.1, i_share=0.1, r_share=0.1)
# main(100, "gamma", shape=1, scale=5, threshold=100, l_share=0.1, i_share=0.1, r_share=0.1)
