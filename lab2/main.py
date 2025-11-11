"""
Главный файл программы для поиска глобального минимума методом ломаных.
Использование: python main.py <функция> <a> <b> <eps> [--lipschitz L]
"""

import sys
import argparse
from function_parser import parse_function
from broken_line import BrokenLineSolver
from visualization import plot_solution


def main():
    parser = argparse.ArgumentParser(
        description='Поиск глобального минимума методом ломаных'
    )
    parser.add_argument('function', type=str, 
                       help='Функция f(x), например: "x + sin(3.14159*x)"')
    parser.add_argument('a', type=float, help='Левая граница отрезка')
    parser.add_argument('b', type=float, help='Правая граница отрезка')
    parser.add_argument('eps', type=float, help='Требуемая точность')
    parser.add_argument('--lipschitz', type=float, default=None,
                       help='Константа Липшица (если не указана, будет оценена)')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения графика')
    
    args = parser.parse_args()
    
    if args.a >= args.b:
        print("Ошибка: a должно быть меньше b")
        sys.exit(1)
    
    if args.eps <= 0:
        print("Ошибка: eps должно быть положительным")
        sys.exit(1)
    
    try:
        # Парсим функцию
        print(f"Функция: f(x) = {args.function}")
        func = parse_function(args.function)
        
        # Создаем решатель
        solver = BrokenLineSolver(
            func=func,
            a=args.a,
            b=args.b,
            eps=args.eps,
            lipschitz_constant=args.lipschitz
        )
        
        print(f"Отрезок: [{args.a}, {args.b}]")
        print(f"Точность: {args.eps}")
        print(f"Константа Липшица: {solver.L:.6f}")
        print("\n" + "="*60)
        print("РЕШЕНИЕ")
        print("="*60)
        
        # Решаем задачу
        x_min, f_min, info = solver.solve()
        
        # Выводим результаты
        print(f"\n✓ Решение найдено!")
        print(f"\nПриближенное значение аргумента минимума: x* = {x_min:.10f}")
        print(f"Приближенное значение минимума: f(x*) = {f_min:.10f}")
        print(f"\nЧисло итераций: {info['iterations']}")
        print(f"Затраченное время: {info['time']:.6f} секунд")
        print(f"Количество вычисленных точек: {info['points_count']}")
        print(f"Оценка погрешности: {info['error_estimate']:.10f}")
        
        # Визуализация
        print("\n" + "="*60)
        print("ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        plot_solution(solver, func, x_min, f_min, save_path=args.output)
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

