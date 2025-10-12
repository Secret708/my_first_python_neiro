# программа для перемешивания датасета (для лучшего обучения нейросети)

import numpy as np
import os

def shuffle_dataset(input_path, output_path=None, random_seed=42):
    """
    Перемешивает датасет .npz сохраняя соответствие images-labels
    
    Args:
        input_path: путь к исходному датасету
        output_path: путь для сохранения перемешанного датасета
        random_seed: seed для воспроизводимости
    """
    
    print("🔀 Начинаем перемешивание датасета...")
    
    # Загружаем исходный датасет
    print(f"📁 Загружаем {input_path}")
    data = np.load(input_path)
    images = data['images']
    labels = data['labels']
    
    print(f"📊 Исходные данные: {len(images)} изображений")
    
    # Проверяем соответствие размеров
    if len(images) != len(labels):
        print("❌ Ошибка: количество images и labels не совпадает!")
        return False
    
    # Показываем исходное распределение
    print("\n📈 Исходное распределение по классам:")
    unique, counts = np.unique(labels, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"   Цифра {digit}: {count} примеров")
    
    # Показываем первые 20 меток до перемешивания
    print(f"\n🔢 Первые 20 меток ДО перемешивания:")
    print(f"   {labels[:20]}")
    
    # Генерируем случайные индексы
    np.random.seed(random_seed)
    indices = np.random.permutation(len(images))
    
    # Перемешиваем данные используя одинаковые индексы
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    
    # Генерируем имя файла если не указано
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_shuffled.npz"
    
    # Сохраняем перемешанный датасет
    print(f"💾 Сохраняем перемешанный датасет как {output_path}")
    np.savez(output_path, images=shuffled_images, labels=shuffled_labels)
    
    # Проверяем результат
    print("✅ Проверяем результат перемешивания...")
    shuffled_data = np.load(output_path)
    check_images = shuffled_data['images']
    check_labels = shuffled_data['labels']
    
    print(f"\n🎉 Перемешивание завершено!")
    print(f"📊 Перемешанных данных: {len(check_images)} изображений")
    
    # Показываем первые 20 меток после перемешивания
    print(f"🔢 Первые 20 меток ПОСЛЕ перемешивания:")
    print(f"   {check_labels[:20]}")
    
    # Проверяем, что распределение сохранилось
    print(f"\n📊 Проверяем сохранение распределения:")
    unique_after, counts_after = np.unique(check_labels, return_counts=True)
    for digit, count in zip(unique_after, counts_after):
        print(f"   Цифра {digit}: {count} примеров")
    
    # Сравниваем с исходным распределением
    if np.array_equal(counts, counts_after):
        print("   ✅ Распределение сохранено правильно!")
    else:
        print("   ❌ Ошибка: распределение изменилось!")
    
    # Информация о файле
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"💾 Размер файла: {file_size:.2f} МБ")
    
    return True

def check_shuffle_quality(dataset_path, sample_size=50):
    """
    Проверяет качество перемешивания
    """
    print(f"\n🔍 Проверка качества перемешивания...")
    
    data = np.load(dataset_path)
    labels = data['labels']
    
    # Анализируем разнообразие в первых sample_size примерах
    first_samples = labels[:sample_size]
    unique_in_first = len(np.unique(first_samples))
    
    print(f"   В первых {sample_size} примерах:")
    print(f"   Уникальных цифр: {unique_in_first}/10")
    print(f"   Разнообразие: {unique_in_first/10*100:.1f}%")
    
    # Показываем частоту появления каждой цифры в начале
    print(f"   Распределение в начале:")
    unique, counts = np.unique(first_samples, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"      {digit}: {count} раз")
    
    return unique_in_first

def main():
    # ⚠️ УКАЖИТЕ ПУТЬ К ВАШЕМУ ДАТАСЕТУ ЗДЕСЬ ⚠️
    input_dataset = "custom_dataset_10000.npz"          # Исходный датасет
    output_dataset = "custom_dataset_shuffled_10000.npz" # Перемешанный датасет
    
    # Проверяем существование файла
    if not os.path.exists(input_dataset):
        print(f"❌ Файл {input_dataset} не найден!")
        print("Проверьте путь к файлу и попробуйте снова.")
        return
    
    # Выполняем перемешивание
    success = shuffle_dataset(input_dataset, output_dataset, random_seed=42)
    
    if success:
        # Проверяем качество перемешивания
        diversity = check_shuffle_quality(output_dataset)
        
        if diversity >= 8:  # Хорошее перемешивание - минимум 8 разных цифр в начале
            print(f"\n✨ Отличное перемешивание! Данные готовы для обучения.")
        else:
            print(f"\n⚠️  Перемешивание прошло, но можно улучшить разнообразие.")
        
        print(f"\n🎯 Теперь используйте {output_dataset} для обучения модели!")
    else:
        print("\n💥 Произошла ошибка при перемешивании!")

if __name__ == "__main__":
    main()