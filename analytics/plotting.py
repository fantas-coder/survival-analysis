import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import weibull_min, lognorm, gamma
from scipy.interpolate import make_interp_spline

from typing import Dict, Tuple

from analytics.utils import DataUtils


class PlottingFunctions:
    """
    Класс для функций построения графиков и визуализации данных
    """
    def paint_distribution_hist(
            self,
            sample: pd.DataFrame,
            distribution_name: str
    ) -> None:
        """
        Функция рисует гистограмму и сглаживающую кривую через центры бинов для выборки

        :param sample: pd.DataFrame, полученный из функции generate_sample()
        :param distribution_name: Название гистограммы
        """
        durations = sample['interval_end'] - sample['interval_start']  # Длительности событий
        n = len(sample)
        bins = 1 + int(np.log2(n))  # Формула Стёрджесса

        # Построение гистограммы
        counts, bins, _ = plt.hist(
            durations,
            bins=bins,
            density=True,
            color='white',
            edgecolor='black',
            alpha=0.7
        )

        # Сглаживание данных (только если есть хотя бы 4 бина)
        if len(counts) >= 4:
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            # Создание сплайна для сглаживания линии
            x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
            spl = make_interp_spline(bin_centers, counts, k=3)  # k=3 для кубического сплайна
            y_smooth = spl(x_smooth)
            # Построение сглаженной линии
            plt.plot(x_smooth, y_smooth, color='black', linestyle='-', linewidth=2, label='Сглаженная плотность')
            # Добавление точек центров бинов
            plt.scatter(bin_centers, counts, color='black', marker='o', label='Центры бинов')

        plt.title(f'Плотность распределения {distribution_name}')
        plt.xlabel('Значение')
        plt.ylabel('Плотность вероятности')
        plt.grid(alpha=0.3)
        plt.legend()

    def paint_distribution_time_life(
            self,
            sample: pd.DataFrame
    ) -> None:
        """
        Функция рисует график эксперимента, как график времени жизни

        :param sample: pd.DataFrame, полученный из функции generate_sample()
        """
        data_utils = DataUtils()

        data_utils.check_right_df_format(sample)
        real_sample = data_utils.get_real_sample(sample)

        n = len(sample)
        indices = np.arange(n)

        # Определяем цвета на основе 'is_not_censored'
        line_colors = pd.Series('grey', index=sample.index)

        real_sample_mask = sample.index.isin(real_sample.index)
        line_colors[real_sample_mask] = 'black'

        # Рисуем горизонтальные линии от interval_start до interval_end
        plt.hlines(
            y=indices,
            xmin=sample['interval_start'],
            xmax=sample['interval_end'],
            color=line_colors, linestyles='-')

        # Рисуем точки в начале и в конце для значений НЕцензурированных элементов
        plt.scatter(
            x=sample['interval_start'][sample['left_censored_value'].isna()],
            y=indices[sample['left_censored_value'].isna()],
            c='black', marker='o', label='real')

        plt.scatter(
            x=real_sample['interval_end'],
            y=indices[real_sample.index],
            c='black', marker='o')

        # Рисуем крестики в начале и в конце для цензурированных элементов
        # Правое цензурирование
        plt.scatter(
            x=sample['interval_end'][sample['is_not_right_censored'] == 0],
            y=indices[sample['is_not_right_censored'] == 0],
            c='lime', marker='x', label='right censored'
        )
        # Левое цензурирование
        plt.scatter(
            x=np.zeros(len(indices[~sample['left_censored_value'].isna()])),
            y=indices[~sample['left_censored_value'].isna()],
            c='skyblue', marker='x', label='left censored'
        )
        plt.scatter(
            x=sample['left_censored_value'][~sample['left_censored_value'].isna()],
            y=indices[~sample['left_censored_value'].isna()],
            c='skyblue', marker='o')
        plt.hlines(
            y=indices[~sample['left_censored_value'].isna()],
            xmin=np.zeros(len(indices[~sample['left_censored_value'].isna()])),
            xmax=sample['left_censored_value'][~sample['left_censored_value'].isna()],
            color='skyblue', linestyles='--', alpha=0.35)
        # Интервальное цензурирование
        plt.scatter(
            x=sample['interval_end'][~sample['interval_censored_value'].isna()],
            y=indices[~sample['interval_censored_value'].isna()],
            c='orange', marker='x', label='interval censored'
        )
        plt.scatter(
            x=sample['interval_censored_value'][~sample['interval_censored_value'].isna()],
            y=indices[~sample['interval_censored_value'].isna()],
            c='orange', marker='o')
        plt.hlines(
            y=indices[~sample['interval_censored_value'].isna()],
            xmin=sample['interval_end'][~sample['interval_censored_value'].isna()],
            xmax=sample['interval_censored_value'][~sample['interval_censored_value'].isna()],
            color='orange', linestyles='--', alpha=0.35)

        plt.title('Распределение времени рождения и смерти')
        plt.xlabel('Время рождения и смерти, Xi')
        plt.ylabel('Индекс, i')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

    def add_distribution_on_plot(
            self,
            distribution_type: str,
            max_size: float = 10.0,
            points_number: int = 1000,
            is_research: bool = False,
            **kwargs: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Функция добавляет на график 1 - F(x), где F(x) - теоретическая функция распределения.

        :param distribution_type: Название распределения из списка ['lognormal', 'weibull', 'gamma']
        :param max_size: Ограничение по оси X
        :param points_number: Количество точек для рисования
        :param kwargs: Параметры распределения
        :param is_research: Флаг, отключающий рисование
        :return: Два np.ndarray: Массив X-ов и Массив Y-ов
        """
        distribution_functions = {
            'weibull': weibull_min,
            'lognormal': lognorm,
            'gamma': gamma
        }

        if distribution_type not in distribution_functions:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

        distribution = distribution_functions[distribution_type]
        x, cdf, label = None, None, None

        if distribution_type == 'weibull':
            loc = kwargs.get('loc', 0.0)
            shape = kwargs.get('shape', 1.0)
            scale = kwargs.get('scale', 1.0)
            x_min = loc if loc < max_size else loc - 1e-5
            x = np.linspace(x_min, max_size, points_number)
            cdf = 1 - distribution.cdf(x, shape, loc=loc, scale=scale)
            if not is_research:
                label = f'Weibull CDF (shape={shape}, scale={scale}, loc={loc})'

        elif distribution_type == 'lognormal':
            mean = kwargs.get('mean', 0.0)
            std_dev = kwargs.get('std_dev', 1.0)
            scale = np.exp(mean)
            x = np.linspace(0.001, max_size, points_number)
            cdf = 1 - distribution.cdf(x, std_dev, scale=scale)
            if not is_research:
                label = f'Lognormal CDF (std_dev={std_dev}, mean={mean})'

        elif distribution_type == 'gamma':
            shape = kwargs.get('shape', 1.0)
            scale = kwargs.get('scale', 1.0)
            x = np.linspace(0, max_size, points_number)
            cdf = 1 - distribution.cdf(x, shape, scale=scale)
            if not is_research:
                label = f'Gamma CDF (shape={shape}, scale={scale})'

        if not is_research:
            plt.plot(x, cdf, color="black", linestyle="--", linewidth=2, label=label)
            plt.legend()
            plt.grid(color="grey", alpha=0.3)

        return x, cdf

    def add_distribution_pdf_on_plot(
            self,
            distribution_type: str,
            max_size: float = 10.0,
            points_number: int = 1000,
            is_research: bool = False,
            **kwargs: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Функция добавляет на график теоретическое значение плотности вероятности f(x).

        :param distribution_type: Название распределения из списка ['lognormal', 'weibull', 'gamma']
        :param max_size: Ограничение по оси X
        :param points_number: Количество точек для рисования
        :param is_research: Флаг, отключающий рисование
        :param kwargs: Параметры распределения
        :return: Массив X-ов и Массив Y-ов
        """
        distribution_functions = {
            'weibull': weibull_min,
            'lognormal': lognorm,
            'gamma': gamma
        }

        if distribution_type not in distribution_functions:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

        distribution = distribution_functions[distribution_type]

        if distribution_type == 'weibull':
            loc = kwargs.get('loc', 0.0)
            shape = kwargs.get('shape', 1.0)
            scale = kwargs.get('scale', 1.0)
            x_min = loc if loc < max_size else loc - 1e-5
            x = np.linspace(x_min + 1e-5, max_size, points_number)
            pdf = distribution.pdf(x, shape, loc=loc, scale=scale)
            if not is_research:
                label = f'Weibull Distribution (shape={shape}, scale={scale}, loc={loc})'

        elif distribution_type == 'lognormal':
            mean = kwargs.get('mean', 0.0)
            std_dev = kwargs.get('std_dev', 1.0)
            scale = np.exp(mean)
            x = np.linspace(1e-5, max_size, points_number)
            pdf = distribution.pdf(x, std_dev, scale=scale)
            if not is_research:
                label = f'Lognormal Distribution (s={std_dev}, scale={scale})'

        elif distribution_type == 'gamma':
            shape = kwargs.get('shape', 1.0)
            scale = kwargs.get('scale', 1.0)
            x = np.linspace(1e-5, max_size, points_number)
            pdf = distribution.pdf(x, shape, scale=scale)
            if not is_research:
                label = f'Gamma Distribution (a={shape}, scale={scale})'

        if not is_research:
            plt.plot(x, pdf, label=label, color="black", linestyle="--", linewidth=2)
            plt.legend()
            plt.grid(color="grey", alpha=0.3)

        return x, pdf

    def paint_kaplan_mayer(
            self,
            mark: pd.Series,
            label: str = None,
            color: str = "blue"
    ) -> None:
        """
        Функция рисует ступенчатый график, представляющий собой оценку Каплана-Майера

        :param mark: pd.Series, полученный из функции kaplan_mayer_mark()
        :param label: Подпись для линии
        :param color: Цвет линии
        """
        if label:
            plt.plot([mark.index[0], mark.index[1]], [mark.values[0], mark.values[0]], linestyle='solid',
                     color=color, label=label)
        else:
            plt.plot([mark.index[0], mark.index[1]], [mark.values[0], mark.values[0]], linestyle='solid',
                     color=color)

        for i in range(1, len(mark) - 1):
            plt.plot([mark.index[i], mark.index[i + 1]], [mark.values[i], mark.values[i]], linestyle='solid',
                     color=color)
            plt.plot([mark.index[i + 1], mark.index[i + 1]], [mark.values[i], mark.values[i + 1]], linestyle='dotted',
                     color=color)
        plt.title('Оценка Каплана-Майера')
        plt.xlabel('time (время)')
        plt.ylabel('Survival function (функция выживаемости)')
        plt.grid(True)

    def paint_confidence_interval_with_kaplan_mayer(
            self,
            mark_interval: Tuple[pd.Series, pd.Series],
            km_mark: pd.Series
    ) -> None:
        """
        Функция строит, как оценку Каплана-Майера, так и её доверительный интервал

        :param mark_interval: Список из двух pd.Series, полученный из функции greenwood_confidence_interval()
        :param km_mark: pd.Series, полученный из функции kaplan_mayer_mark()
        """
        self.paint_kaplan_mayer(km_mark, label="Оценка Каплана – Мейера")
        self.paint_kaplan_mayer(mark_interval[0], label="Доверительный интервал Гринвуда", color="lightblue")
        self.paint_kaplan_mayer(mark_interval[1], color="lightblue")

    def get_metrics_for_kaplan(
            self,
            x: np.ndarray,
            mark: pd.Series,
            true_v: np.ndarray,
            is_research: bool = False
    ) -> Tuple[float, float]:
        """
        Функция считает две метрики для оценки Каплана-Мейера:
        максимум модуля разности и корень среднеквадратичного отклонения и рисует их на графике

        :param x: Массив X-ов, по которым считаются метрики
        :param mark: pd.Series, полученный из функции kaplan_mayer_mark()
        :param true_v: Массив Y-ов, теоретической функции выживаемости (1 - F(x))
        :param is_research: Флаг, отключающий рисование
        :return: Значение метрик: максимум модуля разности и корень среднеквадратичного отклонения,
                 округлённые до 3ёх знаков после запятой
        """
        # Делаем одинаковую размерность для mark и true_v
        bins = mark.index.tolist() + [np.inf]  # Добавляем бесконечность для последнего интервала
        labels = mark.values.tolist()
        new_mark = pd.cut(x, bins=bins, labels=labels, right=False).astype(float)

        # Считаем метрику Максимум Модуля Разности MMD
        difference = np.abs(new_mark - true_v)
        max_difference = np.max(difference)
        if not is_research:
            max_index = np.argmax(difference)
            plt.vlines(x[max_index], new_mark[max_index], true_v[max_index], colors='r', linestyles='dashed',
                       label=f'MMD = {round(max_difference, 3)}', lw=3)
            plt.legend()

        # Считаем метрику Среднее Квадратичное Отклонение MSE
        difference = (new_mark - true_v) ** 2
        mse = np.sum(difference) / len(x)
        if not is_research:
            plt.gca().add_artist(Line2D([0], [0], color='w', lw=0, label=f'SMSE = {round(np.sqrt(mse), 3)}'))
            plt.legend()

        return round(max_difference, 3), round(np.sqrt(mse), 3)

    def paint_kde(
            self,
            x_values: np.ndarray,
            kde_values: np.ndarray,
            title: str = 'Kernel Density Estimation',
            color: str = "red"
    ) -> None:
        """
        Функция рисует ядерную оценку плотности

        :param x_values: Массив X-ов, по которым рисуется оценка
        :param kde_values: Массив Y-ов - значение оценки
        :param title: Название графика и линии
        :param color: Цвет нарисованной линии
        """
        plt.plot(x_values, kde_values, color=color, label=title.split()[0])
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

    def paint_integrated_kernel_mark(
            self,
            x_values: np.ndarray,
            kde_values: np.ndarray,
            second_derivative: np.ndarray,
            label: str = "kde_integral",
            color: str = "red",
            is_research: bool = False
    ) -> np.ndarray:
        """
        Функция строит и рисует ядерную оценку функции выживаемости (1 - интеграл от ядерной оценки плотности)

        :param x_values: Массив X-ов, по которым рисуется оценка
        :param kde_values: Массив Y-ов - значение оценки
        :param second_derivative: Значение второй производной
        :param label: Название линии
        :param color: Цвет линии
        :param is_research: Флаг, отключающий рисование
        :return: Массив значений функции выживаемости
        """
        integrals = np.zeros_like(x_values)
        if not is_research:
            error_estimates = np.zeros_like(x_values)

        # Используем метод трапеций
        for i in range(1, len(x_values)):
            integrals[i] = np.trapezoid(kde_values[:i + 1], x_values[:i + 1])

            if not is_research:
                a, b = x_values[i], x_values[0]
                local_n = i
                error_estimates[i] = -((b - a) ** 3 / (12 * local_n ** 2)) * second_derivative[i]

        if not is_research:
            plt.plot(x_values, 1 - integrals, color=color, label=label)
            plt.fill_between(x_values, 1 - integrals + error_estimates, 1 - integrals - error_estimates,
                             color='gray', alpha=0.3)
            plt.title('Kernel Distribution Function')
            plt.xlabel('x')
            plt.ylabel('Distribution')
            plt.legend()
            plt.grid(True)

        return 1 - integrals

    def get_metrics_for_kde(
            self,
            x: np.ndarray,
            mark: np.ndarray,
            true_v: np.ndarray,
            kernel: str,
            color: str,
            is_research: bool = False
    ) -> Tuple[float, float]:
        """
        Функция считает две метрики для Ядерной оценки:
        максимум модуля разности и корень среднеквадратичного отклонения и рисует их на графике

        :param x: Массив X-ов, по которым считаются метрики
        :param mark: Массив Y-ов - значение оценки
        :param true_v: Массив Y-ов, теоретической функции выживаемости (1 - F(x))
        :param kernel: Название ядра
        :param color: Цвет метрики
        :param is_research: Флаг, отключающий рисование
        :return: Значение метрик: максимум модуля разности и корень среднеквадратичного отклонения,
                округлённые до 3ёх знаков после запятой
        """
        if len(mark) != len(true_v):
            raise ValueError(f"Размерности оценки {len(mark)} и истинных значений {len(true_v)} не совпадают")

        # Считаем метрику Максимум Модуль Разности
        difference = np.abs(mark - true_v)
        max_difference = np.max(difference)

        if not is_research:
            max_index = np.argmax(difference)
            plt.vlines(x[max_index], mark[max_index], true_v[max_index], colors=color, linestyles='dashed',
                       label=f'MMD for {kernel} = {round(max_difference, 3)}', lw=2)
            plt.legend()

        # Считаем метрику Среднее Квадратичное Отклонение MSE
        difference = (mark - true_v) ** 2
        mse = np.sum(difference) / len(x)

        if not is_research:
            plt.gca().add_artist(Line2D([0], [0], color='w', lw=0, label=f'SMSE for {kernel} = {round(np.sqrt(mse), 3)}'))
            plt.legend()

        return round(max_difference, 3), round(np.sqrt(mse), 3)
