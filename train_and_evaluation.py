from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
import yaml

def train_and_evaluation(model, criterion, optimizer, train_loader, test_loader):

    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Процесс обучения
    num_epochs = config['training']['num_epochs'] # Кол-во полных проходов по датасету

    full_train_loss = []
    full_train_accuracy = []
    full_train_precision = []
    full_train_recall = []

    full_test_loss = []
    full_test_accuracy = []
    full_test_precision = []
    full_test_recall = []

    for epoch in range(num_epochs):
        model.train() # Переводит модель в режим обучения.
        running_loss = 0.0 # Накопление значения функции потерь
        train_correct = 0 # Подсчет количества правильных предсказаний модели на тренировочных данных
        train_total = 0 # Подсчет общего количества обработанных данных за эпоху
        train_predicted = []
        train_labels = []

        print(f"Epoch {epoch+1}/{num_epochs}")
        train_progress = tqdm(train_loader, desc="Training", leave=False) # Отображение прогресс-бара во время обучения
        # train_progress — это итератор, который возвращает батчи из train_loader, но с визуализацией прогресса

        for inputs, labels in train_progress: # для каждой картинки из батча
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.argmax(labels, dim=1)  # Преобразование one-hot в индексы

            optimizer.zero_grad() # Обнуление градиентов параметров модели перед каждым шагом обучения

            outputs = model(inputs) # Проход батча изображений через модель и получение предсказания
            loss = criterion(outputs, labels) # Вычисление значений функции потерь
            loss.backward() # Вычисление градиентов функции потерь по параметрам модели (градиенты накапливаются)
            optimizer.step() # Обновление весов на основе вычисленных градиентов

            running_loss += loss.item() * inputs.size(0) # Значение loss для batch x batch_size
            _, predicted = outputs.max(1) # Индекс класса с максимальной вероятностью (1=dim)
            train_total += labels.size(0) # Кол-во элементов в каждом батче
            train_correct += (predicted == labels).sum().item()

            # Сохранение предсказаний и истинных меток
            train_predicted.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / train_total
        train_acc = train_correct / train_total

        # Precision и Recall
        train_precision = precision_score(train_labels, train_predicted, average='macro')  
        # macro для многоклассовой классификации
        train_recall = recall_score(train_labels, train_predicted, average='macro')

        full_train_loss.append(train_loss)
        full_train_accuracy.append(train_acc)
        full_train_precision.append(train_precision)
        full_train_recall.append(train_recall)

        # ___________________ Тестирование ____________________
        model.eval() 
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_predicted = []
        test_labels = []

        with torch.no_grad(): # Отключает вычисление градиентов, что ускоряет процесс и уменьшает использование памяти
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = torch.argmax(labels, dim=1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                # Сохранение предсказаний и истинных меток
                test_predicted.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_loss /= test_total
        test_acc = test_correct / test_total

        # Precision и Recall
        test_precision = precision_score(test_labels, test_predicted, average='macro')  # macro для многоклассовой классификации
        test_recall = recall_score(test_labels, test_predicted, average='macro')

        full_test_loss.append(test_loss)
        full_test_accuracy.append(test_acc)
        full_test_precision.append(test_precision)
        full_test_recall.append(test_recall)

        print(f"Epoch [{epoch+1}/{num_epochs}]\n")
        print('Train metrics')
        print(f"Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}\n")

        print('Test metrics')
        print(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}\n")

    metrics =  {1: ('Loss', full_train_loss, full_test_loss),
               2: ('Accuracy', full_train_accuracy, full_test_accuracy),
               3: ('Precision', full_train_precision, full_test_precision),
               4: ('Recall', full_train_recall, full_test_recall)}
        
    return metrics, test_labels, test_predicted
        