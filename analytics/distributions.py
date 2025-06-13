import numpy as np
import pandas as pd

from typing import Dict


class GenerateFunctions:
    """
    Класс для генерации выборок из различных распределений
    """
    def lognormal_distribution(
            self,
            n: int = 1,
            mean: float = 0.0,
            std_dev: float = 1.0,
            loc: float = 0.0
    ) -> np.ndarray:
        """
        Функция создаёт выборку длинной n из логнормального распределения

        :param n: Количество элементов в выборке
        :param mean: Математическое ожидание (a)
        :param std_dev: Стандартное отклонение (сигма)
        :param loc: Параметр сдвига
        :return: Список (np.ndarray) из Логнормального распределения длинной n с указанными параметрами
        """
        return np.random.lognormal(mean, std_dev, n) + loc

    def weibull_distribution(
            self,
            n: int = 1,
            shape: float = 1.0,
            scale: float = 1.0,
            loc: float = 0.0
    ) -> np.ndarray:
        """
        Функция создаёт выборку длинной n из распределения Вейбулла

        :param n: Количество элементов в выборке
        :param shape: Параметр формы k
        :param scale: Параметр масштаба λ
        :param loc: Параметр сдвига a
        :return: Список (np.ndarray) из распределения Вейбулла длинной n с указанными параметрами
        """
        return scale * np.random.weibull(shape, n) + loc

    def gamma_distribution(
            self,
            n: int = 1,
            shape: float = 1.0,
            scale: float = 1.0
    ) -> np.ndarray:
        """
        Функция создаёт выборку длинной n из Гамма-распределения
        :param n: Количество элементов в выборке
        :param shape: Параметр формы λ
        :param scale: Параметр масштаба a
        :return: Список (np.ndarray) из Гамма-распределения длинной n с указанными параметрами
        """
        return np.random.gamma(shape, scale, n)

    def generate_sample(
            self,
            n: int,
            distribution: str,
            **kwargs: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Функция создаёт выборку случайных интервалов из выбранного закона распределения
        :param n: Количество элементов в выборке
        :param distribution: Название распределения для генерации из списка ['lognormal', 'weibull', 'gamma']
        :param kwargs: Параметры для выбранного распределения
        :return: pd.DataFrame с колонками:
                 'interval_start' - время вхождения объекта в эксперимент,
                 'interval_end' - время наступления искомого события у объекта,
                 'is_not_right_censored' - флаг указывающий на наличие правостороннего цензурирования (значение: 1),
                 'left_censored_value' - величина, указывающая на наличие левостороннего цензурирования
                 (значение: np.nun (без цензурирования)),
                 'interval_censored_value' - величина, указывающая на наличие интервального цензурирования
                 (значение: np.nun (без цензурирования))
        """
        distribution_functions = {
            'lognormal': self.lognormal_distribution,
            'weibull': self.weibull_distribution,
            'gamma': self.gamma_distribution
        }

        if distribution not in distribution_functions:
            raise ValueError(f"Unsupported distribution type: {distribution}")

        distribution_function = distribution_functions[distribution]

        sample1 = distribution_function(n, **kwargs)
        sample2 = distribution_function(n, **kwargs)

        interval_starts = sample1
        interval_ends = sample1 + sample2

        return pd.DataFrame({
            'interval_start': interval_starts,
            'interval_end': interval_ends,
            'is_not_right_censored': np.ones(len(interval_starts)),
            'left_censored_value': np.full(len(interval_starts), np.nan),
            'interval_censored_value': np.full(len(interval_starts), np.nan)
        })
