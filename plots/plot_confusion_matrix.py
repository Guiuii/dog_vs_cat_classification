from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

def plot_confusion_matrix(test_labels, test_predicted, regularization_type):
    ''' Визуализация confusion matrix '''
    
    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Вычисление матрицы
    cm = confusion_matrix(test_labels, test_predicted)

    # Визуализация
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        cmap='flare',
        fmt='d',  # Целочисленный формат
        xticklabels=('cats', 'dogs'),
        yticklabels=('cats', 'dogs')
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix")
    
    # Определяем путь для сохранения
    save_subdir = config['visualization']['subdirs'][regularization_type]
    save_path = os.path.join(
        config['visualization']['base_dir'],
        save_subdir,
        config['visualization']['filenames']['confusion_matrix']
    )
    
    plt.savefig(save_path, bbox_inches="tight")
    #plt.show()
    
