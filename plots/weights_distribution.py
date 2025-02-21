import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

def get_weights(model):
    """Собрать все веса модели (исключая bias) в один массив."""
    weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:  # Исключаем bias
            weights.extend(param.detach().cpu().flatten().numpy())
            # param.detach() - Отключает тензор от вычислительного графа PyTorch (градиенты не вычисляются)
            # cpu() - Перемещает тензор из GPU (если он там был) в CPU (для корректной работы numpy)
            # flatten() - Преобразует многомерный тензор в одномерный вектоhр
            # numpy() - Конвертирует тензор PyTorch в массив NumPy.
    return np.array(weights)

def plot_weights_distribution(initial_weights, reg_weights):
    ''' Визуализация распределения весов без регуляризации и после L2-регуляризации '''
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    plt.figure(figsize=(12, 6))

    # Гистограмма весов без регуляризации
    plt.hist(initial_weights, bins=100, alpha=0.5, label='Без регуляризации', color='blue')

    # Гистограмма весов после обучения с L2-регуляризацией
    plt.hist(reg_weights, bins=100, alpha=0.5, label='С L2-регуляризацией', color='red')

    # Настройки графика
    plt.xlabel('Значение весов')
    plt.ylabel('Частота')
    plt.title('Распределение весов до и после L2-регуляризации')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Логарифмическая шкала для оси Y (опционально)
    
    # Определяем путь для сохранения
    save_path = os.path.join(
        config['visualization']['base_dir'],
        'weights_distribution.png'
    )
    
    plt.savefig(save_path, bbox_inches="tight")
    #plt.show()