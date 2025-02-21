from torch.utils.data import random_split, DataLoader
import torch
import yaml

def data_split(dataset):
    
    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_size = int(config['data']['train_ratio'] * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                               generator=torch.Generator().manual_seed(18)) #torch.utils.data

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    # Вместо обработки одного элемента за раз, DataLoader группирует элементы в батчи, это ускоряет обучение
    # Загрузка данных и передача их в модель по одному элементу создает дополнительные накладные расходы
    # При batch_size=32 градиенты становятся более стабильными, так как они усредняются по нескольким примерам.
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False)

    print(f"Количество классов: {dataset.num_classes}")
    print(f"Классы: {dataset.classes}")
    print(f"Количество образцов: {len(dataset)}")
    print(f"Train size: {train_size}, Test size: {test_size}")
    
    return train_loader, test_loader