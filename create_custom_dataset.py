# с помощью этого файла можно создать свой датасет
# с его помощью потом нейросеть будет обучаться

import pygame
import numpy as np
from PIL import Image

pygame.init()

WIDTH, HEIGHT = 280, 280
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Создаем кастомный датасет - Рисуйте цифры")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RADIUS = 5

canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BLACK)

# Словарь для хранения данных
current_digit = 0
dataset = {"images": [], "labels": []}

def redraw_window():
    WIN.blit(canvas, (0, 0))
    # Показываем текущую цифру для рисования
    font = pygame.font.SysFont('Arial', 30)
    text = font.render(f'Рисуйте цифру: {current_digit}', True, WHITE)
    WIN.blit(text, (10, 10))
    pygame.display.update()

def save_current_digit():
    if len(dataset["images"]) > 0:
        # Сохраняем массив в файл
        np.savez('custom_dataset.npz', 
                 images=np.array(dataset["images"]), 
                 labels=np.array(dataset["labels"]))
        print(f"✅ Датасет сохранен! Примеров: {len(dataset['images'])}")

def preprocess_surface(surface):
    """Конвертируем pygame surface в numpy array как в MNIST"""
    # Получаем пиксели
    pixel_array = pygame.surfarray.array3d(surface)
    
    # Конвертируем в grayscale и инвертируем
    gray_array = 255 - np.mean(pixel_array, axis=2)
    
    # Масштабируем до 28x28
    img = Image.fromarray(gray_array.astype('uint8'))
    img = img.resize((28, 28), Image.LANCZOS)
    
    final_array = np.array(img).astype("float32") / 255.0
    
    return final_array

def main():
    global current_digit
    
    run = True
    drawing = False
    
    print("🎨 Создаем кастомный датасет!")
    print("Цифры 0-9: нажмите соответствующую клавишу")
    print("Пробел: очистить холст")
    print("S: сохранить текущую цифру в датасет") 
    print("Enter: закончить создание датасета")
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            
            if event.type == pygame.MOUSEMOTION and drawing:
                pos = pygame.mouse.get_pos()
                pygame.draw.circle(canvas, WHITE, pos, RADIUS)
            
            if event.type == pygame.KEYDOWN:
                # Цифры 0-9 - выбираем какую цифру рисуем
                if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                                pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]:
                    current_digit = event.key - pygame.K_0
                    print(f"Теперь рисуем цифру: {current_digit}")
                
                # S - сохранить текущий рисунок в датасет
                elif event.key == pygame.K_s:
                    if current_digit is not None:
                        processed_img = preprocess_surface(canvas)
                        dataset["images"].append(processed_img)
                        dataset["labels"].append(current_digit)
                        print(f"✅ Сохранена цифра {current_digit}. Всего: {len(dataset['images'])}")
                    else:
                        print("❌ Сначала выберите цифру (0-9)")
                
                # Пробел - очистить холст
                elif event.key == pygame.K_SPACE:
                    canvas.fill(BLACK)
                    print("Холст очищен")
                
                # Enter - сохранить датасет и выйти
                elif event.key == pygame.K_RETURN:
                    save_current_digit()
                    run = False
        
        redraw_window()
    
    pygame.quit()

if __name__ == "__main__":
    main()