import numpy as np
import pandas as pd

from analytics.utils import DataUtils


class CensoringFunctions:
    """
    Класс для функций, связанных с цензурированием данных
    """
    def threshold_censoring(
            self,
            sample: pd.DataFrame,
            threshold: float
    ) -> pd.DataFrame:
        """
        Функция цензурирует данные, которые больше правого порогового значения

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :param threshold: Пороговое значение
        :return: Изменённый pd.DataFrame с цензурированными справа данными, подходящими под условие
        """
        data_utils = DataUtils()

        data_utils.check_right_df_format(sample)
        sample = sample.copy()

        # Удаляем строки, где 'interval_start' больше порога 'r_threshold'
        sample = sample[sample['interval_start'] <= threshold].reset_index(drop=True)

        # Правое цензурирование
        right_censored_mask = (sample['interval_end'] > threshold)
        # Обновляем столбец 'is_not_right_censored' для элементов, где 'interval_end' больше порога 'r_threshold'
        sample.loc[right_censored_mask, 'is_not_right_censored'] = 0
        # Обновляем 'interval_end' для элементов, у которых 'is_not_right_censored' == 0
        sample.loc[right_censored_mask, 'interval_end'] = threshold

        return sample

    def random_censoring(
            self,
            sample: pd.DataFrame,
            l_share: float = 0.0,
            i_share: float = 0.0,
            r_share: float = 0.0,
            threshold: float = None,
            coeff: float = 1 / 3
    ) -> pd.DataFrame:
        """
        Функция цензурирует случайные элементы выборки, количество задаётся процентами относительно всей выборки

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :param l_share: Доля левостороннего цензурирования (от 0 до 1)
        :param i_share: Доля интервального цензурирования (от 0 до 1)
        :param r_share: Доля правостороннего цензурирования (от 0 до 1)
        :param threshold: Пороговое значение, если была применена функция threshold_censoring
        :param coeff: Глубина цензурирования (от 0 до 1)
        :return: Изменённый pd.DataFrame с цензурированными данными
        """
        data_utils = DataUtils()

        data_utils.check_right_df_format(sample)

        # Корректируем коэффициенты
        total_share = l_share + i_share + r_share
        if total_share > 1.0:
            raise ValueError(f"Суммарная доля цензурирования {total_share} > 1")

        # Считаем количество данных, которых нужно зацензурировать
        sample_length = len(sample)

        # Считаем количество значений, к которым надо применить правое цензурирование
        right_censored_length = sample_length - (sample['is_not_right_censored'] == 1).sum()
        right_censoring_count = round((sample_length * r_share) - right_censored_length)
        # Считаем количество значений, к которым надо применить левое цензурирование
        left_censoring_count = round(sample_length * l_share)
        # Считаем количество значений, к которым надо применить интервальное цензурирование
        interval_censoring_count = round(sample_length * i_share)

        # Правое цензурирование
        if right_censoring_count > 0:
            # Выбор случайных индексов для цензурирования
            random_indices = sample[
                (sample['interval_censored_value'].isna()) & (sample['is_not_right_censored'] == 1) & (
                    sample['left_censored_value'].isna())].sample(n=right_censoring_count,
                                                                  replace=False).index
            # Обновление значений в столбце 'is_not_right_censored'
            sample.loc[random_indices, 'is_not_right_censored'] = 0

            # Обновление значений 'interval_end' на случайное значения между delta [a + (b - a)*coeff] и b
            delta = sample.loc[random_indices, 'interval_start'] + (
                    sample.loc[random_indices, 'interval_end'] - sample.loc[random_indices, 'interval_start']) * coeff

            sample.loc[random_indices, 'interval_end'] = np.random.uniform(
                delta,
                sample.loc[random_indices, 'interval_end']
            )

        # Левое цензурирование
        if left_censoring_count > 0:
            # Выбор случайных индексов для цензурирования
            random_indices = sample[
                (sample['interval_censored_value'].isna()) & (sample['is_not_right_censored'] == 1) & (
                    sample['left_censored_value'].isna())].sample(
                n=left_censoring_count,
                replace=False).index

            # Обновление значений 'left_censored_value' на случайное значения между b-a и delta [b-a + (b - a)*(1 - coeff)]
            delta = sample.loc[random_indices, 'interval_end'] - sample.loc[random_indices, 'interval_start'] + (
                    sample.loc[random_indices, 'interval_end'] - sample.loc[random_indices, 'interval_start']) * (
                                1 - coeff)

            new_value = np.random.uniform(
                sample.loc[random_indices, 'interval_end'],
                delta
            )
            # Проверка на выход за границы
            if threshold:
                new_value = np.where(new_value > threshold, threshold, new_value)

            sample.loc[random_indices, 'left_censored_value'] = new_value

            # Обновление значений 'interval_end и 'interval_start' на np.na
            sample.loc[random_indices, 'interval_end'] = np.nan
            sample.loc[random_indices, 'interval_start'] = np.nan

        # Интервальное цензурирование
        if interval_censoring_count > 0:
            # Выбор случайных индексов для цензурирования
            random_indices = sample[
                (sample['interval_censored_value'].isna()) & (sample['is_not_right_censored'] == 1) & (
                    sample['left_censored_value'].isna())].sample(
                n=interval_censoring_count,
                replace=False).index

            # Обновление значений 'interval_censored_value' на случайное значения между b и delta [b + (b - a)*(1 - coeff)]
            delta = sample.loc[random_indices, 'interval_end'] + (
                    sample.loc[random_indices, 'interval_end'] - sample.loc[random_indices, 'interval_start']) * (
                                1 - coeff)

            new_value = np.random.uniform(
                sample.loc[random_indices, 'interval_end'],
                delta
            )

            if threshold:
                new_value = np.where(new_value > threshold, threshold, new_value)

            sample.loc[random_indices, 'interval_censored_value'] = new_value

            # Обновление значений 'interval_end' на случайное значения между delta [a + (b - a)*coeff] и b
            delta = sample.loc[random_indices, 'interval_start'] + (
                    sample.loc[random_indices, 'interval_end'] - sample.loc[random_indices, 'interval_start']) * coeff

            sample.loc[random_indices, 'interval_end'] = np.random.uniform(
                delta,
                sample.loc[random_indices, 'interval_end']
            )

        return sample
