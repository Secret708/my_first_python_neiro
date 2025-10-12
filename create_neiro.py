# программа для создания нейросети и её обучения

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# Пробуем загрузить кастомный датасет, если есть
if os.path.exists('dataset_1.npz'):
    print("🎯 Загружаем КАСТОМНЫЙ датасет...")
    custom_data = np.load('dataset_1.npz')
    x_train = custom_data['images']
    y_train = custom_data['labels']
    
    # Разделяем на тренировочные и тестовые
    split_idx = int(0.8 * len(x_train))
    x_train, x_test = x_train[:split_idx], x_train[split_idx:]
    y_train, y_test = y_train[:split_idx], y_train[split_idx:]
    
    print(f"Кастомный датасет: {len(x_train)} тренировочных, {len(x_test)} тестовых")
    
else:
    # загружаем MNIST датасет для обучения
    print("📚 Загружаем MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

# Добавляем измерение для канала
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Форма данных: {x_train.shape}")

# ПРОСТАЯ но эффективная модель
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # softmax вместо logits
])
# Компилируем модель 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Обучаем
print("Обучаем модель...")
history = model.fit(x_train, y_train,
                    epochs=15,
                    batch_size=32,
                    validation_data=(x_test, y_test))

model.save('овощ.h5')
print("✅ Модель сохранена!")

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"🎯 Точность модели: {test_acc:.4f}")