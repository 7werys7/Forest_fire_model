import matplotlib.pyplot as plt
import numpy as np

# Генерация 100 равномерно распределённых точек от 0 до 10
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Построение графика
plt.plot(x, y)

# Добавление подписей и заголовка
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')

# Отображение графика
plt.show()
