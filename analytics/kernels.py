import numpy as np
from scipy.stats import norm


class KernelsFunctions:
    """
    Класс для функций ядер и их интегралов
    """
    def gauss_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения Гауссовского ядра в точке x

        :param x: Точка (или массив точек), где рассчитывается значение ядра
        :return: Значение Гауссовского ядра в точках x
        """
        return np.exp(-(x ** 2) / 2) / np.sqrt(2 * np.pi)

    def gauss_integral_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения интеграла от -ꝏ до x для Гауссовского ядра

        :param x: Точка (или массив точек), где рассчитывается значение интеграла
        :return: Значение Интеграла от Гауссовского ядра от -ꝏ до x
        """
        return norm.cdf(x)

    def uniform_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения Прямоугольного ядра в точке x

        :param x: Точка (или массив точек), где рассчитывается значение ядра
        :return: Значение Прямоугольного ядра в точках x
        """
        result = np.zeros_like(x)
        mask = (x >= -0.5) & (x <= 0.5)
        result[mask] = 1
        return result

    def uniform_integral_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения интеграла от -ꝏ до x для Прямоугольного ядра

        :param x: Точка (или массив точек), где рассчитывается значение интеграла
        :return: Значение Интеграла от Прямоугольного ядра от -ꝏ до x
        """
        result = np.zeros_like(x)
        mask1 = (x > -0.5) & (x <= 0.5)
        mask2 = x > 0.5

        result[mask1] = x[mask1] + 0.5
        result[mask2] = 1
        return result

    def epachnikov_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения ядра Эпачникова в точке x

        :param x: Точка (или массив точек), где рассчитывается значение ядра
        :return: Значение ядра Эпачникова в точках x
        """
        result = np.zeros_like(x)
        mask = (x >= -1) & (x <= 1)
        result[mask] = (3 * (1 - x[mask] ** 2)) / 4
        return result

    def epachnikov_integral_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения интеграла от -ꝏ до x для ядра Эпачникова

        :param x: Точка (или массив точек), где рассчитывается значение интеграла
        :return: Значение Интеграла от ядра Эпачникова от -ꝏ до x
        """
        result = np.zeros_like(x)
        mask1 = (x >= -1) & (x <= 1)
        mask2 = x > 1

        result[mask1] = (x[mask1] * (3 - x[mask1] ** 2) + 2) / 4
        result[mask2] = 1
        return result

    def bisquare_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения Биквадратного ядра в точке x

        :param x: Точка (или массив точек), где рассчитывается значение ядра
        :return: Значение Биквадратного ядра в точках x
        """
        result = np.zeros_like(x)
        mask = (x >= -1) & (x <= 1)
        result[mask] = (15 * (1 - x[mask] ** 2) ** 2) / 16
        return result

    def bisquare_integral_kernel(
            self,
            x: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Функция считает значения интеграла от -ꝏ до x для Биквадратного ядра

        :param x: Точка (или массив точек), где рассчитывается значение интеграла
        :return: Значение Интеграла от Биквадратного ядра от -ꝏ до x
        """
        result = np.zeros_like(x)
        mask1 = (x >= -1) & (x <= 1)
        mask2 = x > 1

        result[mask1] = (15 * x[mask1] - 10 * x[mask1] ** 3 + 3 * x[mask1] ** 5 + 8) / 16
        result[mask2] = 1
        return result
