"""
Главный файл программы для решения ЗЛП.
Использование: python main.py input.txt
"""

import sys
from simplex import LinearProgrammingSolver, SolutionStatus
from parser import parse_lp_file


def main():
    if len(sys.argv) < 2:
        print("Использование: python main.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Парсим входной файл
        print(f"Чтение файла: {input_file}")
        data = parse_lp_file(input_file)
        
        # Создаем решатель
        solver = LinearProgrammingSolver()
        solver.load_from_dict(data)
        
        print("\n=== Исходная задача ===")
        print(f"Целевая функция: {'max' if solver.opt_type.value == 'max' else 'min'}", end=" ")
        print(" + ".join([f"{solver.c[i]:.2f}*x{i+1}" for i in range(solver.n_vars)]))
        print("\nОграничения:")
        for i in range(solver.n_constraints):
            constraint_str = " + ".join([f"{solver.A[i,j]:.2f}*x{j+1}" for j in range(solver.n_vars) if abs(solver.A[i,j]) > 1e-6])
            print(f"  {constraint_str} {solver.constraint_types[i].value} {solver.b[i]:.2f}")
        
        # Решаем задачу
        print("\n=== Решение ===")
        status, x, obj_value = solver.solve()
        
        if status == SolutionStatus.OPTIMAL:
            print("✓ Решение найдено!")
            print(f"\nОптимальное значение целевой функции: {obj_value:.6f}")
            print("\nОптимальный вектор x:")
            for i in range(len(x)):
                print(f"  x{i+1} = {x[i]:.6f}")
        elif status == SolutionStatus.UNBOUNDED:
            print("✗ Задача неограничена!")
            print("Целевая функция уходит в бесконечность (для max -> +∞, для min -> -∞)")
        elif status == SolutionStatus.INFEASIBLE:
            print("✗ Задача неразрешима!")
            print("Область допустимых решений пуста")
        else:
            print("✗ Ошибка при решении задачи")
    
    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

