import numpy as np
import pandas as pd

from analytics.kernels import KernelsFunctions
from analytics.utils import DataUtils

from typing import Callable, Tuple


class KernelDensityEstimation:
    """
    Класс для функций ядерной оценки плотности (KDE).
    """
    def get_win_width_by_silverman(
            self,
            sample: pd.DataFrame
    ) -> float:
        """
        Функция получает оценку ширины окна по правилу Сильвермана

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: Значение параметра ширины окна
        """
        data_utils = DataUtils()

        real_sample = data_utils.get_real_sample(sample)
        sample_list = (real_sample['interval_end'] - real_sample['interval_start']).values

        if len(sample_list) < 2:
            return 0.5

        n = len(sample_list)
        std_dev = np.std(sample_list, ddof=1)

        q1 = np.percentile(sample_list, 25)
        q3 = np.percentile(sample_list, 75)
        iqr = q3 - q1

        return 0.9 * min(std_dev, iqr / 1.34) * n ** (-1 / 5)

    def make_v_vector(
            self,
            censoring_sample: np.ndarray,
            another_censoring_sample: np.ndarray = None
    ) -> np.ndarray:
        """
        Функция создаёт массив v, который показывает количество элементов равных текущему элементу в массиве

        :param censoring_sample: Массив, в котором производятся вычисления
        :param another_censoring_sample: Второй массив, если задан, то v будет показывать количество индексов,
               значения которых в первом и втором массивах равны (для интервального цензурирования)
        :return: Массив, который показывает количество вхождений каждого уникального элемента исходного массива
        """
        if another_censoring_sample is None:
            return np.array([sum(1 for x in censoring_sample if x == value) for value in sorted(set(censoring_sample))])

        # Создаем словарь для подсчета количества вхождений каждой уникальной пары
        pair_counts = dict()
        # Проходим по массивам и подсчитываем количество вхождений каждой пары
        for left, right in zip(censoring_sample, another_censoring_sample):
            pair = (left, right)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return np.array(list(pair_counts.values()))

    def get_kde_value(
            self,
            x: float,
            sample: pd.DataFrame,
            win_width: float,
            kernel: Callable[[float], float],
            integral_kernel: Callable[[float], float]
    ) -> float:
        """
        Функция считает ядерную оценку в конкретной точке x

        :param x: Точка, в которой считается ядерная оценка
        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :param win_width: Ширина окна, полученная из функции get_win_width_by_silverman()
        :param kernel: Функция ядра из KernelsFunctions
        :param integral_kernel: Функция интеграла от ядра из KernelsFunctions
        :return: Значение ядерной оценки в точке x
        """
        n = len(sample)

        # Для реальных значений
        data_utils = DataUtils()

        real_sample = data_utils.get_real_sample(sample)
        real_sample = (real_sample['interval_end'] - real_sample['interval_start']).values

        n_real = len(real_sample)

        kde_real_sum = sum(kernel((x - real_sample[i]) / win_width) for i in range(n_real))

        # Для цензурированных значений справа
        r_censoring_sample = (sample[sample['is_not_right_censored'] == 0]['interval_end'] -
                              sample[sample['is_not_right_censored'] == 0]['interval_start']).values

        n_cens_s = len(set(r_censoring_sample))

        v = self.make_v_vector(sorted(r_censoring_sample))
        r_censoring_sample_set = sorted(set(r_censoring_sample))

        kde_r_cens_sum = 0
        for i in range(n_cens_s):
            l_i = r_censoring_sample_set[i]
            term1 = (v[i] * l_i * win_width) / (x ** 2)
            term2 = integral_kernel(1 / (x * win_width))
            term3 = integral_kernel((1 / (x * win_width)) - (1 / (l_i * win_width)))
            kde_r_cens_sum += term1 * (term2 - term3)

        # Для цензурированных значений слева
        l_censoring_sample = sample[~sample['left_censored_value'].isna()]['left_censored_value'].values

        n_cens_s = len(set(l_censoring_sample))

        v = self.make_v_vector(sorted(l_censoring_sample))
        l_censoring_sample_set = sorted(set(l_censoring_sample))

        kde_l_cens_sum = 0
        for i in range(n_cens_s):
            l_i = 0
            delta_i = l_censoring_sample_set[i]
            term1 = (win_width * v[i]) / delta_i
            term2 = integral_kernel((x - l_i) / win_width)
            term3 = integral_kernel((x - l_i - delta_i) / win_width)
            kde_l_cens_sum += term1 * (term2 - term3)

        # Для цензурированных значений интервально
        i_censoring_l = (sample[~sample['interval_censored_value'].isna()]['interval_end'] -
                         sample[~sample['interval_censored_value'].isna()]['interval_start']).values
        i_censoring_r = (sample[~sample['interval_censored_value'].isna()]['interval_censored_value'] -
                         sample[~sample['interval_censored_value'].isna()]['interval_start']).values

        # Создаем множество для отслеживания уникальных пар
        unique_pairs = set()

        # Создаем списки для хранения отфильтрованных значений
        filtered_i_censoring_l = []
        filtered_i_censoring_r = []

        # Проходим по массивам и добавляем уникальные пары в списки
        for left, right in zip(i_censoring_l, i_censoring_r):
            pair = (left, right)
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                filtered_i_censoring_l.append(left)
                filtered_i_censoring_r.append(right)

        # Преобразуем списки обратно в массивы
        filtered_i_censoring_l = np.array(filtered_i_censoring_l)
        filtered_i_censoring_r = np.array(filtered_i_censoring_r)

        n_cens_s = len(filtered_i_censoring_l)
        v = self.make_v_vector(i_censoring_l, i_censoring_r)

        kde_i_cens_sum = 0
        for i in range(n_cens_s):
            l_i = filtered_i_censoring_l[i]
            delta_i = filtered_i_censoring_r[i] - filtered_i_censoring_l[i]
            term1 = (win_width * v[i]) / delta_i
            term2 = integral_kernel((x - l_i) / win_width)
            term3 = integral_kernel((x - l_i - delta_i) / win_width)
            kde_i_cens_sum += term1 * (term2 - term3)
        return (kde_real_sum + kde_r_cens_sum + kde_l_cens_sum + kde_i_cens_sum) / (n * win_width)

    def get_second_derivative_kde_value(
            self,
            sample: np.ndarray,
            win_width: float,
            kernel: Callable[[float], float]
    ) -> np.ndarray:
        """
        Функция считает вторую производную ядерной оцени в каждой точке

        :param sample: Значения ядерной оценки
        :param win_width: Ширина окна, полученная из функции get_win_width_by_silverman()
        :param kernel: Функция ядра из KernelsFunctions
        :return: Массив со значениями второй производной для ядерной оценки размерностью, как sample
        """
        n = len(sample)
        second_derivative = np.array(n)
        for x in sample:
            kde_value = 0
            for i in range(n):
                u = (x - sample[i]) / win_width
                kde_value += (u ** 2 - 1) * kernel(u)
            second_derivative = np.append(second_derivative, kde_value / (n * win_width ** 3))
        return second_derivative

    def calculate_kde(
            self,
            sample: pd.DataFrame,
            kernel_type: str,
            max_size: float,
            n: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Функция считает все параметры для ядерной оценки плотности и её саму

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :param kernel_type: Название ядра из списка ['gauss', 'uniform', 'epachnikov', 'bisquare']
        :param max_size: Значение до которого рассчитывается ядерная оценка
        :param n: Количество точек в получаемой ядерной оценки
        :return: Кортеж из трёх элементов: массива X-ов ядерной оценки,
                 массива Y-ов (значения ядерной оценки), массива Y-ов (значения второй производной для ядерной оценки),
                 размерности всех массивов n
        """
        kernels_func = KernelsFunctions()

        kernels = {
            'gauss': kernels_func.gauss_kernel,
            'uniform': kernels_func.uniform_kernel,
            'epachnikov': kernels_func.epachnikov_kernel,
            'bisquare': kernels_func.bisquare_kernel
        }
        integral_kernels = {
            'gauss': kernels_func.gauss_integral_kernel,
            'uniform': kernels_func.uniform_integral_kernel,
            'epachnikov': kernels_func.epachnikov_integral_kernel,
            'bisquare': kernels_func.bisquare_integral_kernel
        }

        if kernel_type not in kernels:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        kernel = kernels[kernel_type]
        integral_kernel = integral_kernels[kernel_type]

        win_width = self.get_win_width_by_silverman(sample)
        x_values = np.linspace(0.001, max_size, n)

        kde_values = self.get_kde_value(x_values, sample, win_width, kernel, integral_kernel)

        second_derivative = self.get_second_derivative_kde_value(kde_values, win_width, kernel)
        return x_values, kde_values, second_derivative
