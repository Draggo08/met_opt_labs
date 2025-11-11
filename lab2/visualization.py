"""
Модуль для визуализации результатов метода ломаных.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Callable, Optional
from broken_line import BrokenLineSolver


def plot_solution(solver: BrokenLineSolver, 
                  func: Callable[[float], float],
                  x_min: float, f_min: float,
                  save_path: Optional[str] = None):
    """
    Строит график функции, ломаной и найденного минимума.
    
    Args:
        solver: Решатель после выполнения solve()
        func: Исходная функция
        x_min: Найденная точка минимума
        f_min: Найденное значение минимума
        save_path: Путь для сохранения графика (если None, показывается)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Сетка для построения графиков
    x_plot = np.linspace(solver.a, solver.b, 1000)
    y_func = [func(x) for x in x_plot]
    
    # График исходной функции
    ax.plot(x_plot, y_func, 'b-', linewidth=2, label='Исходная функция f(x)')
    
    # График ломаной
    x_broken, y_broken = solver.get_broken_line_points(n_points=1000)
    ax.plot(x_broken, y_broken, 'r--', linewidth=1.5, alpha=0.7, label='Ломаная (нижняя оценка)')
    
    # Точки, в которых вычислена функция
    points_x = [p[0] for p in solver.points]
    points_y = [p[1] for p in solver.points]
    ax.scatter(points_x, points_y, color='green', s=50, zorder=5, label='Вычисленные точки')
    
    # Найденный минимум
    ax.plot(x_min, f_min, 'ro', markersize=12, zorder=6, label=f'Найденный минимум: x*={x_min:.6f}, f*={f_min:.6f}')
    
    # Вертикальная линия в точке минимума
    ax.axvline(x=x_min, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Метод ломаных: поиск глобального минимума', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранен в {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(func: Callable[[float], float],
                    a: float, b: float,
                    x_min: float, f_min: float,
                    save_path: Optional[str] = None):
    """
    Строит упрощенный график только функции и найденного минимума.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_plot = np.linspace(a, b, 1000)
    y_func = [func(x) for x in x_plot]
    
    ax.plot(x_plot, y_func, 'b-', linewidth=2, label='f(x)')
    ax.plot(x_min, f_min, 'ro', markersize=10, zorder=5, 
            label=f'Минимум: x*={x_min:.6f}, f*={f_min:.6f}')
    ax.axvline(x=x_min, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('График функции и найденный минимум', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

