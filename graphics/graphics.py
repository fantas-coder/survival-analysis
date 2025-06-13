import pandas as pd
import matplotlib.pyplot as plt


# 1. Загрузка данных
df = pd.read_excel("graphics_for_deeplom.xlsx", sheet_name=0)

# 2. Подготовка списков столбцов
mmd_cols = [col for col in df.columns if col.startswith('MMD')]
smse_cols = [col for col in df.columns if col.startswith('SMSE')]

n_mark = len(mmd_cols)

# 3. Список цветов
if n_mark == 5:
    colors = ['blue', 'red', 'orange', 'cyan', 'green']
else:
    colors = ['red', 'orange', 'cyan', 'green']

# 4. Построение графика MMD
plt.figure(figsize=(12, 6))  # Размер графика

# Weibull, Lognormal, Gamma
# 1. Доля цензурирования
plt.title('MMD vs. Cens Rate Weibull')
plt.xlabel('Cens Rate')
# 2. Порог цензурирования
# plt.title('MMD vs. Threshold with Gamma')
# plt.xlabel('Threshold')
# 3. Тип цензурирования
# Right, Interval, Left
# plt.title('MMD vs. Interval Cens with Weibull')
# plt.xlabel('Interval Cens Rate')
# 4. Глубина цензурирования
# plt.title('MMD vs. Deep Cens with Weibull')
# plt.xlabel('Deep Cens Coeff')
plt.ylabel('MMD value')
for i, col in enumerate(mmd_cols):
    color = colors[i % len(colors)]
    # 1. Доля цензурирования
    plt.plot(df['Censor_Rate'], df[col], label=col, color=color)
    # 2. Порог цензурирования
    # plt.plot(df['Threshold'], df[col], label=col, color=color)
    # 3. Тип цензурирования
    # plt.plot(df['curr_cens'], df[col], label=col, color=color)
    # 4. Глубина цензурирования
    # plt.plot(df['coeff'], df[col], label=col, color=color)
plt.legend()
plt.grid(True)  # Добавляем сетку
# plt.gca().invert_xaxis()
plt.tight_layout()  # Предотвращает перекрытия элементов графика
plt.show()

# 4. Построение графика SMSE
plt.figure(figsize=(12, 6))  # Размер графика

# Weibull, Lognormal, Gamma
# 1. Доля цензурирования
plt.title('SMSE vs. Cens Rate with Weibull')
plt.xlabel('Cens Rate Coeff')
# 2. Порог цензурирования
# plt.title('SMSE vs. Threshold with Gamma')
# plt.xlabel('Threshold')
# 3. Тип цензурирования
# Right, Interval, Left
# plt.title('SMSE vs. Interval Cens with Weibull')
# plt.xlabel('Interval Cens Rate')
# 4. Глубина цензурирования
# plt.title('SMSE vs. Deep Cens with Weibull')
# plt.xlabel('Deep Cens Coeff')
plt.ylabel('SMSE value')
for i, col in enumerate(smse_cols):
    color = colors[i % len(colors)]
    # 1. Доля цензурирования
    plt.plot(df['Censor_Rate'], df[col], label=col, color=color)
    # 2. Порог цензурирования
    # plt.plot(df['Threshold'], df[col], label=col, color=color)
    # 3. Тип цензурирования
    # plt.plot(df['curr_cens'], df[col], label=col, color=color)
    # 4. Глубина цензурирования
    # plt.plot(df['coeff'], df[col], label=col, color=color)
plt.legend()
plt.grid(True)  # Добавляем сетку
# plt.gca().invert_xaxis()
plt.tight_layout()  # Предотвращает перекрытия элементов графика
plt.show()
