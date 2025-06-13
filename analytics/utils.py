import pandas as pd


class DataUtils:
    """
    Класс, содержащий вспомогательные функции для работы с данными.
    """
    def check_right_df_format(
            self,
            df: pd.DataFrame
    ) -> None:
        """
        Функция проверяет pd.DataFrame на наличие нужных столбцов

        :param df: pd.DataFrame, которая подвергается проверке
        """
        required_columns = ['interval_start', 'interval_end', 'is_not_right_censored', 'left_censored_value',
                            'interval_censored_value']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise TypeError(f"Не правильный формат данных, отсутствуют столбцы: {', '.join(missing_columns)}")

    def get_real_sample(
            self,
            sample: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Функция получает полные данные из всех

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: pd.DataFrame, содержащий только полные данные
        """
        return sample[
            (sample['is_not_right_censored'] == 1) &
            (sample['left_censored_value'].isna()) &
            (sample['interval_censored_value'].isna())
            ]

    def get_max_size(
            self,
            sample: pd.DataFrame
    ) -> float:
        """
        Функция получает максимальную длину выборки, учитывая все виды цензурирования

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: Максимальный размер выборки
        """
        # Максимальное значение по левому цензурированию
        max_distance1 = sample['left_censored_value'].max()

        # Максимальное значение по интервальному цензурированию
        max_distance2 = (sample['interval_censored_value'] - sample['interval_start']).max()

        # Максимальное значение без цензуриоования
        max_distance3 = (sample['interval_end'] - sample['interval_start']).max()

        # Максимальное значение
        max_distance = max([x for x in [max_distance1, max_distance2, max_distance3] if pd.notna(x)])

        return max_distance + 1e-6

    def get_mean_real_duration(
            self,
            sample: pd.DataFrame
    ) -> float:
        """
        Функция получает среднее значение продолжительности жизни у полных наблюдений

        :param sample: pd.DataFrame, со столбцами, как из функции generate_sample()
        :return: Среднее значение полных наработок
        """
        real_sample = self.get_real_sample(sample)

        if real_sample.empty:
            return 0.0

        durations = real_sample['interval_end'] - real_sample['interval_start']
        return durations.mean()
