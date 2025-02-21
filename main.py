from torchvision import transforms
import numpy as np
import os
from PIL import Image
import kagglehub
import yaml
import torch
from torch.utils.data import Dataset
from model_construction import model_construction
from train_and_evaluation import train_and_evaluation
from plots.plot_metrics import plot_metrics
from plots.plot_confusion_matrix import plot_confusion_matrix
from data_split import data_split
from plots.weights_distribution import get_weights, plot_weights_distribution
from transforms_from_config import create_transforms

# Загрузка конфига
with open("config.yaml") as f:
    config = yaml.safe_load(f)
    
# Создание директории
os.makedirs(config['visualization']['base_dir'], exist_ok=True)
    
# Загрузка последней версии
path = kagglehub.dataset_download(
    config["data"]["dataset_path"])

print("Path to dataset files:", path)

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Путь к директории с данными. Внутри должно быть N папок, каждая соответствует своему классу.
        :param transform: Трансформации, которые будут применяться к изображениям.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Список всех папок (классов) в корневой директории
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        # ["cat", "dog"]

        # Количество классов
        self.num_classes = len(self.classes)

        # Создание маппинга класс -> индекс
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        # {"cat":0, "dog":1}

        # Список всех изображений и их классов
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name) # путь к директории конкретного класса
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, self.class_to_idx[class_name])) 
                    # (путь к изображению, номер класса)

        # One-hot encoding для классов
        self.one_hot_classes = np.eye(self.num_classes)
        # numpy.eye(num of rows in output) 
        # Return a 2-D array with ones on the diagonal and zeros elsewhere.
        # [[1, 0],
        # [0, 1]]

    def __len__(self):
        """Возвращает длину датасета"""
        return len(self.samples) # кол-во изображений всего

    def __getitem__(self, idx):
        """
        Возвращает элемент датасета по индексу.
        :param idx: Индекс элемента
        :return: Кортеж (изображение, one-hot класс)
        """
        img_path, class_idx = self.samples[idx] # путь к изображению, номер класса

        # Загрузка изображения
        image = Image.open(img_path).convert('RGB') # PIL

        # Применение трансформаций (если есть)
        if self.transform:
            image = self.transform(image)

        # One-hot encoding для класса
        label = self.one_hot_classes[class_idx]

        return image, torch.tensor(label, dtype=torch.float32) 
        # тензор, представляющий метку класса в формате one-hot encoding.
        # tensor([0., 1.])
        
# Создание датасета
dataset = ClassificationDataset(
    root_dir=os.path.join(path, "animals"),
    transform=create_transforms(config["augmentation"]) # функция create_transforms возвращает transforms.Compose()
)
# Преимущества аугментации: увеличение объема данных, повышение разнообразия, борьба с переобучением

print(f"Количество классов: {dataset.num_classes}")
print(f"Классы: {dataset.classes}")
print(f"Количество образцов: {len(dataset)}\n")

# Получение первого элемента
image, label = dataset[0]
print(f"Размер изображения: {image.size()}")
#print(f"One-hot метка: {label}")

# _________ Model without regularization __________
print("Model without regularization")
train_loader, test_loader = data_split(dataset)
model, criterion, optimizer = model_construction(dataset, config_section='base')
metrics, test_labels, test_predicted = train_and_evaluation(model, criterion, optimizer, train_loader, test_loader)
    
plot_metrics(metrics, 'without_reg')
plot_confusion_matrix(test_labels, test_predicted, 'without_reg')

# ________ Model with L2-regularization __________
print("Model with L2-regularization")
model_reg, criterion, optimizer_reg = model_construction(dataset, config_section='l2_reg')

metrics_reg, test_labels_reg, test_predicted_reg = train_and_evaluation(model_reg, criterion, 
                                                                        optimizer_reg, train_loader, 
                                                                        test_loader)
    
plot_metrics(metrics_reg, 'l2_reg')
plot_confusion_matrix(test_labels_reg, test_predicted_reg, 'l2_reg')

# ________ Weights distribution before and after regularization ________
initial_weights = get_weights(model)
reg_weights = get_weights(model_reg)
plot_weights_distribution(initial_weights, reg_weights)