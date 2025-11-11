#!/usr/bin/env python3
"""Скрипт для решения задачи 2 - демонстрация метода ломаных"""

import sys
import time
from function_parser import parse_function
from broken_line import BrokenLineSolver
from visualization import plot_solution
from test_functions import rastrigin_1d, ackley_1d, test_function_1


def solve_example(func_str: str, func, a: float, b: float, eps: float, 
                  name: str, output_file: str = None):
    """Решает задачу для одной функции и выводит результаты"""
    print("\n" + "="*80)
    print(f"ПРИМЕР: {name}")
    print("="*80)
    print(f"Функция: f(x) = {func_str}")
    print(f"Отрезок: [{a}, {b}]")
    print(f"Точность: {eps}")
    
    # Создаем решатель
    solver = BrokenLineSolver(func=func, a=a, b=b, eps=eps)
    
    print(f"Константа Липшица (оценка): {solver.L:.6f}")
    
    # Решаем
    start_time = time.time()
    x_min, f_min, info = solver.solve()
    total_time = time.time() - start_time
    
    # Выводим результаты
    print("\n" + "-"*80)
    print("РЕЗУЛЬТАТЫ:")
    print("-"*80)
    print(f"Приближенное значение аргумента минимума: x* = {x_min:.10f}")
    print(f"Приближенное значение минимума: f(x*) = {f_min:.10f}")
    print(f"Число итераций: {info['iterations']}")
    print(f"Затраченное время: {info['time']:.6f} секунд")
    print(f"Количество вычисленных точек: {info['points_count']}")
    print(f"Оценка погрешности: {info['error_estimate']:.10f}")
    
    # Визуализация
    output_path = output_file or f"visualization_{name.lower().replace(' ', '_')}.png"
    plot_solution(solver, func, x_min, f_min, save_path=output_path)
    
    return x_min, f_min, info


def main():
    print("="*80)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: МЕТОД ЛОМАНЫХ")
    print("="*80)
    
    # Пример 1: Функция Растригина
    solve_example(
        func_str="x^2 - 10*cos(2*pi*x) + 10",
        func=rastrigin_1d,
        a=-5.0,
        b=5.0,
        eps=0.01,
        name="Функция Растригина",
        output_file="rastrigin_result.png"
    )
    
    # Пример 2: Функция Экли
    solve_example(
        func_str="-20*exp(-0.2*sqrt(0.5*x^2)) - exp(0.5*cos(2*pi*x)) + e + 20",
        func=ackley_1d,
        a=-5.0,
        b=5.0,
        eps=0.01,
        name="Функция Экли",
        output_file="ackley_result.png"
    )
    
    # Пример 3: Простая функция из задания
    solve_example(
        func_str="x + sin(3.14159*x)",
        func=test_function_1,
        a=0.0,
        b=10.0,
        eps=0.01,
        name="Функция x + sin(3.14159*x)",
        output_file="test_function_result.png"
    )
    
    print("\n" + "="*80)
    print("ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ")
    print("="*80)


if __name__ == "__main__":
    main()

