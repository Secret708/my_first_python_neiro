# файл для создания данных, которые будет
# распознавать ваша нейросеть

import pygame
import os
import time

pygame.init()

WIDTH, HEIGHT = 280, 280 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Нарисуй цифру')

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RADIUS = 7 # лучший радиус

canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BLACK)

def redraw_window():
    screen.blit(canvas, (0, 0))
    pygame.display.update()
    
def preprocess_and_save_digit(surface, folder="my_digits"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Создаем копию поверхности
    processed_surface = surface.copy()
    
    # Создаем временную поверхность для обработки
    temp_surface = pygame.Surface((WIDTH, HEIGHT))
    temp_surface.fill(BLACK)
    temp_surface.blit(processed_surface, (0, 0))
    
    # Масштабируем до 28x28 для предпросмотра в консоли
    scaled = pygame.transform.scale(temp_surface, (28, 28))
    pygame.image.save(scaled, f"{folder}/preview_{int(time.time())}.png")
    
    filename = f"{folder}/digit_{int(time.time())}.png"
    pygame.image.save(temp_surface, filename)
    print(f"Цифра сохранена как {filename}")
    
    # Показываем ASCII-превью в консоли
    show_ascii_preview(temp_surface)

def show_ascii_preview(surface):
    """Показывает ASCII превью цифры в консоли"""
    # Масштабируем до 28x28
    scaled = pygame.transform.scale(surface, (28, 28))
    
    # Получаем пиксели
    pixel_array = pygame.surfarray.array3d(scaled)
    
    print("\n" + "="*30)
    print("ASCII превью вашей цифры:")
    print("="*30)
    
    # Выводим упрощенное ASCII представление
    for y in range(0, 28, 2):  # Берем каждый второй пиксель для читаемости
        line = ""
        for x in range(28):
            # Проверяем яркость пикселя
            brightness = pixel_array[x][y][0]  # R канал (в grayscale все одинаковы)
            if brightness > 128:
                line += "██"
            else:
                line += "  "
        print(line)
    print("="*30)
    
def main(): # главный цикл программы
    run = True
    drawing = False
    
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
            if event.type == pygame.MOUSEMOTION and drawing:
                pos = pygame.mouse.get_pos()
                pygame.draw.circle(canvas, WHITE, pos, RADIUS)
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    preprocess_and_save_digit(canvas)
                    
                if event.key == pygame.K_c:
                    canvas.fill(BLACK)
                    
        redraw_window()
        
    pygame.quit()
    
if __name__ == '__main__':
    main()