import numpy as np
import pandas as pd
from scipy.stats import norm

from lifelines import KaplanMeierFitter

from analytics.utils import DataUtils

from typing import Tuple


class KaplanMeyerFunctions:
    """
    Класс для функций построения оценки Каплана-Мейера
    """
    def get_real_mark(
            self,
            sample: pd.DataFrame
    ) -> tuple[callable, np.ndarray]:
        """
        Функция получает эмпирическую функцию распределения по полным данным

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: tuple, (функция ЭФР, отсортированный массив данных)
        """
        data_utils = DataUtils()

        real_sample = data_utils.get_real_sample(sample)
        real_list = (real_sample['interval_end'] - real_sample['interval_start']).values

        real_list = real_list[~np.isnan(real_list)]
        sorted_data = np.sort(real_list)
        n = len(sorted_data)

        # Если данных нет, возвращаем функцию, которая всегда возвращает 0
        if n == 0:
            def ecdf(_):
                return 0.0
            return ecdf

        # Вычисляем доли для ЭФР
        # Для каждого значения x в sorted_data, ECDF(x) = (число элементов <= x) / n
        def ecdf(
                x: float | np.ndarray
        ) -> float | np.ndarray:
            """
            Эмпирическая функция распределения для реальных данных.

            :param x: Точка или массив точек, для которых нужно вычислить ЭФР
            :return: Значение(я) ЭФР в точке(ах) x
            """
            if isinstance(x, (int, float)):
                return np.mean(sorted_data <= x)
            else:
                return np.array([np.mean(sorted_data <= xi) for xi in x])

        return ecdf, sorted_data

    def prepare_to_kaplan_adaptive_v2(
            self,
            sample: pd.DataFrame
    ) -> pd.Series:
        """
        Функция приводит все типы цензурирования к правостороннему с помощью адаптивного подхода
        (Все значения пытаются замениться на средние значение ЭФР)

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: pd.Series, где data = is_not_censored, index - значение случ величины
        """
        series_data = []    # Индикатор цензурирования: 1 - нецензурировано, 0 - цензурировано
        series_index = []   # Оценка времени

        # Получаем функцию ЭФР
        ecdf_func, sorted_data = self.get_real_mark(sample)
        n = len(sorted_data)

        def inverse_ecdf(x: float) -> float:
            """
            Обратная ЭФР: находит минимальное x, такое что F(x) >= p.
            """
            if n == 0 or x <= 0:
                return 0.0
            if x >= 1:
                return sorted_data[-1]
            # Находим индекс i, такой что i/n >= p
            i = int(np.ceil(x * n))  # np.ceil, чтобы получить i, где F(x_i) >= p
            if i == 0:
                return sorted_data[0]  # Для малых p берем минимальное значение
            return sorted_data[i - 1]  # Берем x_{i-1}, где F(x_{i-1}) >= p

        for index, row in sample.iterrows():
            # Левостороннее цензурирование
            if pd.notna(row['left_censored_value']):
                series_data.append(0)
                f_b = ecdf_func(row['left_censored_value'])
                p = f_b / 2
                estimated_time = inverse_ecdf(p)
                series_index.append(estimated_time)
            # Интервальное цензурирование
            elif pd.notna(row['interval_censored_value']):
                series_data.append(0)
                f_a = ecdf_func(row['interval_end'] - row['interval_start'])
                f_b = ecdf_func(row['interval_censored_value'] - row['interval_start'])
                p = (f_a + f_b) / 2
                estimated_time = inverse_ecdf(p)
                series_index.append(estimated_time)
            # Правостороннее цензурирование
            elif row['is_not_right_censored'] == 0:
                series_data.append(0)
                series_index.append(row['interval_end'] - row['interval_start'])
            # Полные данные
            else:
                series_data.append(1)
                series_index.append(row['interval_end'] - row['interval_start'])

        return pd.Series(series_data, index=series_index)

    def prepare_to_kaplan_pessimistic(
            self,
            sample: pd.DataFrame
    ) -> pd.Series:
        """
        Функция приводит все типы цензурирования к правостороннему пессимистичным подходом
        (Предполагаем, что все события, которые произошли до времени t, произошли в самом начале интервала цензурирования)

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: pd.Series, где data = is_not_censored, index - значение случ величины
        """
        series_data = []
        series_index = []

        for index, row in sample.iterrows():
            if not pd.isna(row['left_censored_value']):
                series_data.append(0)
                series_index.append(row['left_censored_value'] // 2)
                continue

            interval_length = row['interval_end'] - row['interval_start']

            if row['is_not_right_censored'] == 0 or not pd.isna(row['interval_censored_value']):
                series_data.append(0)
                series_index.append(interval_length)
            else:
                series_data.append(1)
                series_index.append(interval_length)

        return pd.Series(series_data, index=series_index)

    def prepare_to_kaplan_adaptive(
            self,
            sample: pd.DataFrame,
            mean_duration: float
    ) -> pd.Series:
        """
        Функция приводит все типы цензурирования к правостороннему с помощью адаптивного подхода
        (Все значения пытаются замениться на среднее значение наступления события)

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :param mean_duration: Среднее значение полных наработок
        :return: pd.Series, где data = is_not_censored, index - значение случ величины
        """
        series_data = []
        series_index = []

        for index, row in sample.iterrows():
            if not pd.isna(row['left_censored_value']):
                series_data.append(0)
                left_censored_value = row['left_censored_value']
                if left_censored_value > mean_duration:
                    series_index.append(mean_duration)
                else:
                    series_index.append(left_censored_value)
            elif not pd.isna(row['interval_censored_value']):
                series_data.append(0)
                interval_censored_value = row['interval_censored_value'] - row['interval_start']
                interval_end = row['interval_end'] - row['interval_start']
                if interval_censored_value < mean_duration:
                    series_index.append(interval_censored_value)
                elif interval_end > mean_duration:
                    series_index.append(interval_end)
                else:
                    series_index.append(mean_duration)
            elif row['is_not_right_censored'] == 0:
                series_data.append(0)
                series_index.append(row['interval_end'] - row['interval_start'])
            else:
                series_data.append(1)
                series_index.append(row['interval_end'] - row['interval_start'])

        return pd.Series(series_data, index=series_index)

    def kaplan_mayer_mark(
            self,
            sample: pd.Series
    ) -> pd.Series:
        """
        Функция получает оценку Каплана-Майера

        :param sample: pd.Series, из функции prepare_to_kaplan()
        :return: pd.Series, где data - значение оценки, index - значение времени
        """
        sample_series = self.prepare_to_kaplan_adaptive_v2(sample)

        # Сортируем выборку и создаём pd.Series для оценки Каплана-Мейера
        sample_series = sample_series.sort_index()
        sample_set = sorted(set(sample_series.index))
        number_of_life = len(sample_series)
        km_mark = pd.Series(data=[1], index=[0], name="*S(t)")
        km_mark.index.name = "time"

        # Считаем оценку Каплана-Мейера
        res_mark = 1
        for t in sample_set:
            x_t = sample_series[sample_series.index == t]
            deaths_number = int(x_t.sum())
            if deaths_number > 0:
                f = 1 - (deaths_number / number_of_life)
                res_mark *= f
                km_mark[t] = res_mark
            number_of_life -= len(x_t)

        return km_mark

    def find_dispersion(
            self,
            sample: pd.Series,
            mark: pd.Series
    ) -> pd.Series:
        """
        Функция ищет дисперсию по формуле Гринвуда

        :param sample: pd.Series, из функции prepare_to_kaplan()
        :param mark: Оценка Каплана-Мейера из функции kaplan_mayer_mark()
        :return: pd.Series, где data - значение дисперсии, index - значение времени
        """
        sample_series = self.prepare_to_kaplan_adaptive_v2(sample)

        sample_series = sample_series.sort_index()
        number_of_life = len(sample_series)
        dispersion = pd.Series(data=np.zeros(len(mark)), index=mark.index)

        res_sum = 0
        for t in mark.index:
            x_t = sample_series[sample_series.index == t]
            deaths_number = int(x_t.sum())
            if number_of_life > deaths_number > 0:
                res_sum += deaths_number / ((number_of_life - deaths_number) * number_of_life)
                dispersion[t] = mark[t] ** 2 * res_sum
            number_of_life -= len(x_t)

        return dispersion

    def greenwood_confidence_interval(
            self,
            sample_series: pd.Series,
            mark: pd.Series,
            alpha: float
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Функция получает доверительный интервал по формуле Гринвуда, значения большие 1 и меньшие 0 обрезаются

        :param sample_series: pd.Series, из функции prepare_to_kaplan()
        :param mark: Оценка Каплана-Мейера из функции kaplan_mayer_mark()
        :param alpha: Доверительная вероятность для доверительного интервала
        :return: Список из двух pd.Series, представляющих левое и правое значения доверительного интервала
        """
        se = np.sqrt(self.find_dispersion(sample_series, mark))

        # Получаем квантиль нормального распределения порядка 1 - alpha / 2
        quantile = norm.ppf(1 - (alpha / 2))

        lower_bound = mark - quantile * se
        lower_bound[lower_bound < 0] = 0

        upper_bound = mark + quantile * se
        upper_bound[upper_bound > 1] = 1

        return lower_bound, upper_bound

    def km_fitter(
            self,
            sample: pd.Series
    ) -> pd.Series:
        """
        Функция получает оценку Каплана-Майера из встроенного метода

        :param sample: pd.Series, из функции prepare_to_kaplan()
        :return: pd.Series, где data - значение оценки, index - значение времени
        """
        sample_series = self.prepare_to_kaplan_adaptive_v2(sample)

        duration = sample_series.index
        event_observed = sample_series.values

        kmf = KaplanMeierFitter()
        kmf.fit(
            duration,
            event_observed,
            alpha=0.05,
        )

        kmf.plot_survival_function(ci_show=False, color="lime")
        return kmf.survival_function_
