from torchvision import transforms, models
import os
import torch.optim as optim
import torch.nn as nn
import torch
import yaml

def model_construction(dataset, config_section):
    """Создает модель на основе указанной секции конфига"""
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model'][config_section]
    
    # Создание предобученной модели resnet18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.__dict__[model_cfg['name']](
        pretrained=model_cfg['pretrained']
    )
    # Веса модели инициализируются на основе обучения на наборе данных ImageNet
    model.fc = nn.Linear(model.fc.in_features, model_cfg['num_classes'])  # Замена последнего слоя под количество классов (1000 -> 2)
    # model.fc.in_features — это количество входных признаков для последнего слоя (512 для ResNet-18)
    # входных: 512, выходных: 2
    model = model.to(device)

    # Функция потерь:
    criterion = nn.CrossEntropyLoss() # Разница между площадью распределения модели и площадью разметки

    # Оптимизатор
    optimizer = optim.Adam(
        model.parameters(), # model.parameters() — все обучаемые параметры модели
        lr = model_cfg['lr'],
        weight_decay = model_cfg['weight_decay']) 
    
    return model, criterion, optimizer
