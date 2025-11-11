"""
Реализация метода ломаных (метод Пана) для поиска глобального минимума
липшицевой функции на отрезке.
"""

import time
from typing import List, Tuple, Callable, Optional
import numpy as np


class BrokenLineSolver:
    """
    Решатель задачи поиска глобального минимума методом ломаных.
    """
    
    def __init__(self, func: Callable[[float], float], 
                 a: float, b: float, 
                 eps: float,
                 lipschitz_constant: Optional[float] = None,
                 adaptive_lipschitz: bool = True,
                 r: float = 2.0):
        """
        Инициализация решателя.
        
        Args:
            func: Функция для минимизации
            a: Левая граница отрезка
            b: Правая граница отрезка
            eps: Требуемая точность
            lipschitz_constant: Константа Липшица (если None, будет оценена)
            adaptive_lipschitz: Использовать адаптивную оценку константы Липшица
            r: Параметр надежности для адаптивной оценки (используется если adaptive_lipschitz=True)
        """
        self.func = func
        self.a = a
        self.b = b
        self.eps = eps
        self.adaptive_lipschitz = adaptive_lipschitz
        self.r = r
        
        # Начальная оценка константы Липшица
        if lipschitz_constant is None:
            from function_parser import estimate_lipschitz_constant
            self.L = estimate_lipschitz_constant(func, a, b)
        else:
            self.L = lipschitz_constant
        
        # Точки, в которых вычислена функция
        self.points: List[Tuple[float, float]] = []  # (x, f(x))
        
        # История итераций
        self.iterations = 0
        self.start_time = None
        self.end_time = None
    
    def solve(self) -> Tuple[float, float, dict]:
        """
        Решает задачу поиска минимума.
        
        Returns:
            (x_min, f_min, info) где:
            - x_min: приближенное значение аргумента минимума
            - f_min: приближенное значение минимума
            - info: словарь с информацией о решении
        """
        self.start_time = time.time()
        
        # Начальные точки - концы отрезка
        f_a = self.func(self.a)
        f_b = self.func(self.b)
        
        self.points = [(self.a, f_a), (self.b, f_b)]
        self.points.sort(key=lambda p: p[0])  # Сортируем по x
        
        self.iterations = 0
        EPS_MIN = 1e-12
        
        while True:
            self.iterations += 1
            
            # Адаптивная оценка константы Липшица
            if self.adaptive_lipschitz and len(self.points) >= 2:
                self._update_lipschitz_constant()
            
            # Находим точку минимума ломаной
            x_new = self._find_minimum_of_broken_line()
            
            # Вычисляем значение функции в новой точке
            f_new = self.func(x_new)
            
            # Добавляем точку
            self.points.append((x_new, f_new))
            self.points.sort(key=lambda p: p[0])
            
            # Проверяем условие остановки
            # Находим минимальное значение функции среди всех точек
            f_min = min(p[1] for p in self.points)
            x_min = min(p[0] for p in self.points if abs(p[1] - f_min) < 1e-10)
            
            # Проверка по длине интервала (более надежный критерий)
            max_interval_length = 0.0
            for i in range(len(self.points) - 1):
                dx = self.points[i + 1][0] - self.points[i][0]
                max_interval_length = max(max_interval_length, dx)
            
            # Оценка точности: находим минимальное значение ломаной на всем отрезке
            # и сравниваем с минимальным значением функции
            broken_line_min_value = self._find_minimum_broken_line_value()
            error_estimate = abs(f_min - broken_line_min_value)
            
            # Останавливаемся, если достигнута точность по интервалу И по оценке
            if max_interval_length < self.eps and error_estimate < self.eps:
                self.end_time = time.time()
                elapsed_time = self.end_time - self.start_time
                
                info = {
                    'iterations': self.iterations,
                    'time': elapsed_time,
                    'lipschitz_constant': self.L,
                    'points_count': len(self.points),
                    'error_estimate': error_estimate
                }
                
                return x_min, f_min, info
    
    def _update_lipschitz_constant(self):
        """
        Обновляет оценку константы Липшица на основе уже вычисленных точек.
        Используется адаптивный подход: M = max |f(x_i) - f(x_j)| / |x_i - x_j|
        """
        EPS_MIN = 1e-12
        M = 0.0
        
        for i in range(1, len(self.points)):
            dx = self.points[i][0] - self.points[i - 1][0]
            if dx > EPS_MIN:
                lipschitz_estimate = abs(self.points[i][1] - self.points[i - 1][1]) / dx
                M = max(M, lipschitz_estimate)
        
        if M < EPS_MIN:
            M = 1.0
        
        # Используем параметр надежности r
        self.L = self.r * M
    
    def _find_minimum_broken_line_value(self) -> float:
        """
        Находит минимальное значение ломаной функции на всем отрезке.
        """
        min_value = float('inf')
        
        # Проверяем все точки пересечения ломаных
        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]
            
            x_intersect = self._find_intersection(x1, f1, x2, f2)
            
            if self.a <= x_intersect <= self.b:
                value = self._evaluate_broken_line(x_intersect)
                min_value = min(min_value, value)
        
        # Также проверяем концы отрезка
        min_value = min(min_value, self._evaluate_broken_line(self.a))
        min_value = min(min_value, self._evaluate_broken_line(self.b))
        
        return min_value
    
    def _find_minimum_of_broken_line(self) -> float:
        """
        Находит точку минимума вспомогательной ломаной функции.
        Ломаная строится как нижняя оценка исходной функции.
        """
        min_value = float('inf')
        min_x = self.a
        
        # Проверяем все точки пересечения ломаных
        for i in range(len(self.points) - 1):
            x1, f1 = self.points[i]
            x2, f2 = self.points[i + 1]
            
            # Точка пересечения двух лучей ломаной
            # Ломаная: max(f(x_i) - L*|x - x_i|) для всех i
            # Находим пересечение между точками x1 и x2
            x_intersect = self._find_intersection(x1, f1, x2, f2)
            
            if self.a <= x_intersect <= self.b:
                value = self._evaluate_broken_line(x_intersect)
                if value < min_value:
                    min_value = value
                    min_x = x_intersect
        
        return min_x
    
    def _find_intersection(self, x1: float, f1: float, 
                          x2: float, f2: float) -> float:
        """
        Находит точку пересечения двух лучей ломаной между x1 и x2.
        Луч из точки (x_i, f_i): f(x) = f_i - L*|x - x_i|
        """
        # Между x1 и x2 ломаная определяется как:
        # max(f1 - L*(x - x1), f2 - L*(x2 - x))
        # Точка пересечения: f1 - L*(x - x1) = f2 - L*(x2 - x)
        # f1 - L*x + L*x1 = f2 - L*x2 + L*x
        # f1 + L*x1 - f2 + L*x2 = 2*L*x
        # x = (f1 + L*x1 - f2 + L*x2) / (2*L)
        
        if abs(self.L) < 1e-10:
            return (x1 + x2) / 2
        
        x_intersect = (f1 + self.L * x1 - f2 + self.L * x2) / (2 * self.L)
        
        # Ограничиваем интервалом [x1, x2]
        x_intersect = max(x1, min(x2, x_intersect))
        
        return x_intersect
    
    def _evaluate_broken_line(self, x: float) -> float:
        """
        Вычисляет значение ломаной функции в точке x.
        Ломаная: max_i (f(x_i) - L*|x - x_i|)
        """
        max_value = float('-inf')
        
        for x_i, f_i in self.points:
            value = f_i - self.L * abs(x - x_i)
            max_value = max(max_value, value)
        
        return max_value
    
    def get_broken_line_points(self, n_points: int = 1000) -> Tuple[List[float], List[float]]:
        """
        Возвращает точки для построения графика ломаной функции.
        
        Returns:
            (x_values, y_values) - массивы координат для графика
        """
        x_values = np.linspace(self.a, self.b, n_points)
        y_values = [self._evaluate_broken_line(x) for x in x_values]
        
        return x_values.tolist(), y_values

