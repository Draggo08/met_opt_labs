"""
Парсер для функций одной переменной.
Поддерживает математические функции: sin, cos, exp, log, sqrt и т.д.
"""

import math
import re
from typing import Callable


def parse_function(func_str: str) -> Callable[[float], float]:
    """
    Парсит строку функции и возвращает callable функцию.
    
    Примеры:
    - "x + sin(3.14159*x)"
    - "x**2 - 5*x + 6"
    - "rastrigin(x)"
    """
    # Очищаем строку от пробелов и f(x) = 
    func_str = func_str.strip()
    func_str = re.sub(r'^f\s*\(\s*x\s*\)\s*=\s*', '', func_str, flags=re.IGNORECASE)
    
    # Заменяем ^ на ** для совместимости с Python
    func_str = func_str.replace('^', '**')
    
    # Создаем безопасное окружение для eval
    safe_dict = {
        'x': None,  # Будет заменено при вызове
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'sqrt': math.sqrt,
        'abs': abs,
        'pi': math.pi,
        'e': math.e,
        'pow': pow,
        '__builtins__': {},
    }
    
    # Компилируем выражение
    try:
        code = compile(func_str, '<string>', 'eval')
    except SyntaxError as e:
        raise ValueError(f"Ошибка синтаксиса в функции: {e}")
    
    def func(x: float) -> float:
        safe_dict['x'] = x
        try:
            return float(eval(code, safe_dict))
        except Exception as e:
            raise ValueError(f"Ошибка вычисления функции в точке x={x}: {e}")
    
    return func


def estimate_lipschitz_constant(func: Callable[[float], float], 
                                 a: float, b: float, 
                                 n_samples: int = 1000) -> float:
    """
    Оценивает константу Липшица функции на отрезке [a, b].
    Использует численную оценку максимального модуля производной.
    """
    h = (b - a) / n_samples
    max_derivative = 0.0
    
    for i in range(n_samples):
        x = a + i * h
        # Численная производная
        if i == 0:
            x1 = x + h
            derivative = (func(x1) - func(x)) / h
        elif i == n_samples - 1:
            x0 = x - h
            derivative = (func(x) - func(x0)) / h
        else:
            x0 = x - h
            x1 = x + h
            derivative = (func(x1) - func(x0)) / (2 * h)
        
        max_derivative = max(max_derivative, abs(derivative))
    
    # Добавляем запас для надежности
    return max_derivative * 1.1 if max_derivative > 0 else 1.0

