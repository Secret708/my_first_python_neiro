# программа для соединения двух датасетов в один

import numpy as np
import os
from collections import defaultdict

def merge_datasets_sorted(dataset1_path, dataset2_path, output_path=None):
    """
    Объединяет два датасета, сортируя данные по классам (цифрам)
    """
    
    print("🔄 Начинаем объединение с сортировкой по классам...")
    
    # Загружаем датасеты
    print(f"📁 Загружаем {dataset1_path}")
    dataset1 = np.load(dataset1_path)
    images1 = dataset1['images']
    labels1 = dataset1['labels']
    
    print(f"📁 Загружаем {dataset2_path}")
    dataset2 = np.load(dataset2_path)
    images2 = dataset2['images']
    labels2 = dataset2['labels']
    
    print(f"📊 Dataset1: {len(images1)} изображений")
    print(f"📊 Dataset2: {len(images2)} изображений")
    
    # Собираем все данные в словарь по классам
    class_data = defaultdict(lambda: {'images': [], 'labels': []})
    
    # Добавляем данные из первого датасета
    for img, lbl in zip(images1, labels1):
        class_data[lbl]['images'].append(img)
        class_data[lbl]['labels'].append(lbl)
    
    # Добавляем данные из второго датасета
    for img, lbl in zip(images2, labels2):
        class_data[lbl]['images'].append(img)
        class_data[lbl]['labels'].append(lbl)
    
    # Сортируем классы по возрастанию (0, 1, 2, ... 9)
    sorted_classes = sorted(class_data.keys())
    
    # Собираем отсортированные данные
    sorted_images = []
    sorted_labels = []
    
    print("\n📦 Собираем данные по классам:")
    for class_id in sorted_classes:
        class_images = class_data[class_id]['images']
        class_labels = class_data[class_id]['labels']
        
        sorted_images.extend(class_images)
        sorted_labels.extend(class_labels)
        
        print(f"   Цифра {class_id}: {len(class_images)} примеров")
    
    # Конвертируем в numpy arrays
    final_images = np.array(sorted_images)
    final_labels = np.array(sorted_labels)
    
    # Генерируем имя файла если не указано
    if output_path is None:
        base1 = os.path.splitext(dataset1_path)[0]
        base2 = os.path.splitext(dataset2_path)[0]
        output_path = f"merged_sorted_{os.path.basename(base1)}_{os.path.basename(base2)}.npz"
    
    # Сохраняем объединенный датасет
    print(f"\n💾 Сохраняем объединенный датасет как {output_path}")
    np.savez(output_path, images=final_images, labels=final_labels)
    
    # Проверяем результат
    print("✅ Проверяем результат...")
    merged_data = np.load(output_path)
    check_images = merged_data['images']
    check_labels = merged_data['labels']
    
    print("\n🎉 Объединение завершено!")
    print(f"📈 Итоговая статистика:")
    print(f"   📊 Всего изображений: {len(check_images)}")
    print(f"   🏷️  Всего меток: {len(check_labels)}")
    
    # Показываем распределение по классам
    unique, counts = np.unique(check_labels, return_counts=True)
    print("   📊 Распределение по цифрам:")
    for digit, count in zip(unique, counts):
        print(f"      Цифра {digit}: {count} примеров")
    
    # Показываем первые несколько меток для проверки порядка
    print(f"\n🔍 Первые 30 меток (для проверки порядка):")
    print(f"   {check_labels[:30]}")
    
    return True

def main(): # главный запуск
    # ⚠️ УКАЖИТЕ ПУТИ К ВАШИМ ДАТАСЕТАМ ЗДЕСЬ ⚠️
    dataset1_path = "dataset_1.npz"      # Первый датасет
    dataset2_path = "dataset_2.npz"    # Второй датасет  
    output_path = "dataset_3.npz"   # Результат
    
    # Проверяем существование файлов
    for path in [dataset1_path, dataset2_path]:
        if not os.path.exists(path):
            print(f"❌ Файл {path} не найден!")
            print("Проверьте пути к файлам и попробуйте снова.")
            return
    
    # Выполняем объединение
    success = merge_datasets_sorted(dataset1_path, dataset2_path, output_path)
    
    if success:
        print("\n✨ Объединение с сортировкой по классам прошло успешно!")
        print("   Теперь все цифры идут по порядку: 0, 0, 0..., 1, 1, 1..., 2, 2, 2... и т.д.")
    else:
        print("\n💥 Произошла ошибка при объединении!")
        
if __name__ == '__main__':
    main()