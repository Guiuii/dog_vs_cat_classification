import os
import matplotlib.pyplot as plt
import yaml

def plot_metrics(metrics, regularization_type):
    ''' Визуализация графиков зависимости Loss, Accuracy, Precision, Recall от кол-ва эпох '''
    
    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    plt.figure(figsize=(10,10))

    # metrics =  {1: ('Loss', full_train_loss, full_test_loss),
    #            2: ('Accuracy', full_train_accuracy, full_test_accuracy),
    #            3: ('Precision', full_train_precision, full_test_precision),
    #            4: ('Recall', full_train_recall, full_test_recall)}
    epochs_num = config["training"]["num_epochs"]
    
    for ind, val in metrics.items():
        plt.subplot(2, 2, ind)
        name, train, test = val
        plt.plot(range(1, epochs_num+1), train, label = 'Train',
                 color = config['visualization']['colors']['train'])
        plt.plot(range(1, epochs_num+1), test, label = 'Test',
                 color = config['visualization']['colors']['test'])
        plt.xlabel('Epochs') 
        plt.ylabel(name)  
        plt.title(name) 
        plt.legend()

    # Определяем путь для сохранения
    save_subdir = config['visualization']['subdirs'][regularization_type]
    save_path = os.path.join(
        config['visualization']['base_dir'],
        save_subdir,
        config['visualization']['filenames']['metrics']
    )
    
    plt.savefig(save_path, bbox_inches="tight")
    #plt.show()