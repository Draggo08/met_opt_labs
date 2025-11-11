#!/usr/bin/env python3
"""Скрипт для решения задачи 1"""

import sys
sys.path.insert(0, '/Users/daniilgelm/Docs/met_opt/lab1')

from simplex import LinearProgrammingSolver
from parser import parse_lp_file

# Решаем задачу
data = parse_lp_file('task1.txt')
solver = LinearProgrammingSolver()
solver.load_from_dict(data)

print("=" * 60)
print("ЗАДАЧА ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ")
print("=" * 60)
print(f"\nЦелевая функция: {'max' if solver.opt_type.value == 'max' else 'min'}", end=" ")
print(" + ".join([f"{solver.c[i]:.1f}*x{i+1}" for i in range(solver.n_vars)]))
print("\nОграничения:")
for i in range(solver.n_constraints):
    constraint_str = " + ".join([f"{solver.A[i,j]:.1f}*x{j+1}" for j in range(solver.n_vars) if abs(solver.A[i,j]) > 1e-6])
    print(f"  {constraint_str} {solver.constraint_types[i].value} {solver.b[i]:.1f}")

print("\n" + "=" * 60)
print("РЕШЕНИЕ")
print("=" * 60)

status, x, obj_value = solver.solve()

if status.value == "optimal":
    print("\n✓ РЕШЕНИЕ НАЙДЕНО!")
    print(f"\nОптимальное значение целевой функции: Z* = {obj_value:.6f}")
    print("\nОптимальная точка x*:")
    for i in range(len(x)):
        print(f"  x{i+1}* = {x[i]:.6f}")
    
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ОГРАНИЧЕНИЙ:")
    print("=" * 60)
    for i in range(solver.n_constraints):
        lhs = sum(solver.A[i, j] * x[j] for j in range(solver.n_vars))
        ct = solver.constraint_types[i]
        if ct.value == "<=":
            check = lhs <= solver.b[i] + 1e-6
            print(f"Ограничение {i+1}: {lhs:.6f} <= {solver.b[i]:.6f} {'✓' if check else '✗'}")
        elif ct.value == ">=":
            check = lhs >= solver.b[i] - 1e-6
            print(f"Ограничение {i+1}: {lhs:.6f} >= {solver.b[i]:.6f} {'✓' if check else '✗'}")
        elif ct.value == "=":
            check = abs(lhs - solver.b[i]) < 1e-6
            print(f"Ограничение {i+1}: {lhs:.6f} = {solver.b[i]:.6f} {'✓' if check else '✗'}")
    
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ:")
    print("=" * 60)
    print(f"Оптимальная точка: x* = ({', '.join([f'{x[i]:.6f}' for i in range(len(x))])})")
    print(f"Значение целевой функции в оптимальной точке: Z* = {obj_value:.6f}")
    
elif status.value == "unbounded":
    print("\n✗ ЗАДАЧА НЕОГРАНИЧЕНА!")
    print("Целевая функция уходит в бесконечность.")
    print("Решений нет (критерий уходит в +∞).")
    
elif status.value == "infeasible":
    print("\n✗ ЗАДАЧА НЕРАЗРЕШИМА!")
    print("Область допустимых решений пуста.")
    print("Решений нет (множество X пусто).")

