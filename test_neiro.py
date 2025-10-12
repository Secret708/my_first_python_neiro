# программа для теста нейросети для данных 
# из папки my_digits, которые мы получили 
# из файла main.py 

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob

# Загружаем модель, обученную на ваших данных
model = tf.keras.models.load_model('овощ.h5')
print("✅ Модель загружена!")

def preprocess_image(image_path):
    """Упрощенная предобработка - как при создании датасета"""
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)
    
    # ИНВЕРТИРУЕМ: белое на черном -> черное на белом
    img_array = 255 - img_array
    
    # Нормализуем
    img_array = img_array.astype("float32") / 255.0
    
    # Добавляем размерности
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def show_image_preview(image_path):
    """Показывает как модель видит вашу цифру"""
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)
    img_array = 255 - img_array
    
    print("\n" + "="*40)
    print("Как модель видит вашу цифру:")
    print("="*40)
    
    for y in range(28):
        line = ""
        for x in range(28):
            brightness = img_array[y][x]
            if brightness > 128:
                line += "██"
            elif brightness > 64:
                line += "░░"
            else:
                line += "  "
        print(line)
    print("="*40)

def predict_digit(image_path):
    show_image_preview(image_path)
    
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    print(f"🎯 Результат распознавания:")
    print(f"   Цифра: {predicted_digit}")
    print(f"   Уверенность: {confidence:.2%}")
    
    # Показываем все вероятности
    print("\n📊 Все вероятности:")
    for digit, prob in enumerate(predictions[0]):
        marker = " ←" if digit == predicted_digit else ""
        print(f"   {digit}: {prob:.2%}{marker}")
    
    return predicted_digit

# Основная программа
if __name__ == "__main__":
    folder = "my_digits"
    if os.path.exists(folder):
        image_files = glob.glob(f"{folder}/*.png")
        
        if image_files:
            print(f"🔍 Найдено {len(image_files)} изображений")
            for image_file in image_files:
                print(f"\nАнализируем: {os.path.basename(image_file)}")
                predict_digit(image_file)
        else:
            print("❌ Нет PNG файлов. Сначала нарисуйте цифры!")
    else:
        print("❌ Папка 'my_digits' не найдена!")