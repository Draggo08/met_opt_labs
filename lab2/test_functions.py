"""
Тестовые функции для демонстрации метода ломаных.
"""

import math
import numpy as np


def rastrigin_1d(x: float) -> float:
    """
    Функция Растригина (одномерная версия).
    f(x) = x^2 - 10*cos(2*pi*x) + 10
    Имеет множество локальных минимумов.
    """
    return x**2 - 10 * math.cos(2 * math.pi * x) + 10


def ackley_1d(x: float) -> float:
    """
    Функция Экли (одномерная версия).
    f(x) = -20*exp(-0.2*sqrt(0.5*x^2)) - exp(0.5*cos(2*pi*x)) + e + 20
    """
    return (-20 * math.exp(-0.2 * math.sqrt(0.5 * x**2)) - 
            math.exp(0.5 * math.cos(2 * math.pi * x)) + 
            math.e + 20)


def test_function_1(x: float) -> float:
    """
    Тестовая функция с несколькими локальными минимумами.
    f(x) = x + sin(3.14159*x)
    """
    return x + math.sin(3.14159 * x)


def test_function_2(x: float) -> float:
    """
    Тестовая функция с несколькими локальными минимумами.
    f(x) = (x-2)^2 * sin(5*x) + 0.1*x^2
    """
    return (x - 2)**2 * math.sin(5 * x) + 0.1 * x**2


def test_function_3(x: float) -> float:
    """
    Тестовая функция с несколькими локальными минимумами.
    f(x) = x^4 - 10*x^2 + 9*x + 5*sin(3*x)
    """
    return x**4 - 10 * x**2 + 9 * x + 5 * math.sin(3 * x)

