"""
Модуль для решения задач линейного программирования симплекс-методом.
Автор: Гельм Даниил
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class OptimizationType(Enum):
    """Тип оптимизации: максимизация или минимизация."""
    MAX = "max"
    MIN = "min"


class ConstraintType(Enum):
    """Тип ограничения: <=, >=, =."""
    LE = "<="
    GE = ">="
    EQ = "="


class SolutionStatus(Enum):
    """Статус решения ЗЛП."""
    OPTIMAL = "optimal"
    UNBOUNDED = "unbounded"
    INFEASIBLE = "infeasible"


class LinearProgrammingSolver:
    """
    Решатель задач линейного программирования.
    Использует симплекс-метод с двухфазным подходом.
    """
    
    def __init__(self):
        self.c = None  # Коэффициенты целевой функции
        self.A = None  # Матрица ограничений
        self.b = None  # Правая часть ограничений
        self.opt_type = None  # Тип оптимизации
        self.constraint_types = None  # Типы ограничений
        self.n_vars = 0  # Количество переменных
        self.n_constraints = 0  # Количество ограничений
        
    def load_from_dict(self, data: Dict):
        """
        Загружает ЗЛП из словаря.
        
        Формат:
        {
            'objective': {'coeffs': [c1, c2, ...], 'type': 'max'/'min'},
            'constraints': [
                {'coeffs': [a1, a2, ...], 'type': '<='/'>='/'=', 'rhs': b}
            ]
        }
        """
        obj = data['objective']
        self.c = np.array(obj['coeffs'], dtype=float)
        self.opt_type = OptimizationType.MAX if obj['type'] == 'max' else OptimizationType.MIN
        self.n_vars = len(self.c)
        
        constraints = data['constraints']
        self.n_constraints = len(constraints)
        self.A = np.zeros((self.n_constraints, self.n_vars))
        self.b = np.zeros(self.n_constraints)
        self.constraint_types = []
        
        for i, constraint in enumerate(constraints):
            self.A[i] = np.array(constraint['coeffs'], dtype=float)
            self.b[i] = float(constraint['rhs'])
            self.constraint_types.append(ConstraintType(constraint['type']))
    
    def to_canonical_form(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Преобразует ЗЛП в канонический вид.
        
        Возвращает:
        - A_canon: матрица ограничений в каноническом виде
        - b_canon: правая часть в каноническом виде
        - c_canon: коэффициенты целевой функции в каноническом виде
        - basis: список индексов базисных переменных
        - artificial_vars: список индексов искусственных переменных
        """
        # Если минимизация, меняем знак целевой функции
        c_canon = self.c.copy()
        if self.opt_type == OptimizationType.MIN:
            c_canon = -c_canon
        
        # Подсчитываем количество дополнительных переменных
        n_slack = 0  # Слабая переменная для <=
        n_surplus = 0  # Избыточная переменная для >=
        n_artificial = 0  # Искусственная переменная для >= и =
        
        for ct in self.constraint_types:
            if ct == ConstraintType.LE:
                n_slack += 1
            elif ct == ConstraintType.GE:
                n_surplus += 1
                n_artificial += 1
            elif ct == ConstraintType.EQ:
                n_artificial += 1
        
        total_vars = self.n_vars + n_slack + n_surplus + n_artificial
        
        # Создаем расширенную матрицу
        A_canon = np.zeros((self.n_constraints, total_vars))
        b_canon = self.b.copy()
        
        # Копируем исходную матрицу A
        A_canon[:, :self.n_vars] = self.A
        
        # Индексы для дополнительных переменных
        idx_slack = self.n_vars
        idx_surplus = self.n_vars + n_slack
        idx_artificial = self.n_vars + n_slack + n_surplus
        
        slack_idx = 0
        surplus_idx = 0
        artificial_idx = 0
        basis = []
        artificial_vars = []
        
        # Обрабатываем каждое ограничение
        for i, ct in enumerate(self.constraint_types):
            if ct == ConstraintType.LE:
                # Добавляем слабую переменную
                A_canon[i, idx_slack + slack_idx] = 1.0
                basis.append(idx_slack + slack_idx)
                slack_idx += 1
            elif ct == ConstraintType.GE:
                # Добавляем избыточную переменную (со знаком -)
                A_canon[i, idx_surplus + surplus_idx] = -1.0
                # Добавляем искусственную переменную
                art_idx = idx_artificial + artificial_idx
                A_canon[i, art_idx] = 1.0
                basis.append(art_idx)
                artificial_vars.append(art_idx)
                surplus_idx += 1
                artificial_idx += 1
            elif ct == ConstraintType.EQ:
                # Добавляем искусственную переменную
                art_idx = idx_artificial + artificial_idx
                A_canon[i, art_idx] = 1.0
                basis.append(art_idx)
                artificial_vars.append(art_idx)
                artificial_idx += 1
        
        # Расширяем целевую функцию
        c_canon_extended = np.zeros(total_vars)
        c_canon_extended[:self.n_vars] = c_canon
        
        return A_canon, b_canon, c_canon_extended, basis, artificial_vars
    
    def two_phase_simplex(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, 
                         basis: List[int], artificial_vars: List[int]) -> Tuple[SolutionStatus, Optional[np.ndarray], 
                                                    Optional[float], Optional[np.ndarray]]:
        """
        Двухфазный симплекс-метод.
        
        Фаза 1: Нахождение начального опорного решения (минимизация суммы искусственных переменных).
        Фаза 2: Решение исходной задачи.
        """
        n_constraints, n_vars = A.shape
        
        if not artificial_vars:
            # Нет искусственных переменных - можно сразу решать основную задачу
            return self.simplex_method(A, b, c, basis)
        
        # ФАЗА 1: Минимизация суммы искусственных переменных
        # Целевая функция для фазы 1: минимизировать сумму искусственных переменных
        # Для максимизации (симплекс-метод работает с максимизацией) используем отрицательные коэффициенты
        c_phase1 = np.zeros(n_vars)
        for idx in artificial_vars:
            c_phase1[idx] = -1.0  # Минимизируем сумму = максимизируем отрицательную сумму
        
        # Решаем вспомогательную задачу
        status_phase1, x_phase1, obj_phase1, basis_phase1 = self.simplex_method(
            A, b, c_phase1, basis.copy()
        )
        
        # Если оптимальное значение < 0 (т.е. сумма искусственных переменных > 0), задача неразрешима
        # obj_phase1 - это значение -sum(artificial_vars), так что если obj_phase1 < -1e-6, то sum > 1e-6
        if status_phase1 == SolutionStatus.INFEASIBLE or \
           status_phase1 == SolutionStatus.UNBOUNDED or \
           basis_phase1 is None or \
           (status_phase1 == SolutionStatus.OPTIMAL and obj_phase1 < -1e-6):
            return SolutionStatus.INFEASIBLE, None, None, None
        
        # ФАЗА 2: Решение исходной задачи
        # Удаляем столбцы искусственных переменных из A и c
        keep_cols = [i for i in range(n_vars) if i not in artificial_vars]
        A_phase2 = A[:, keep_cols]
        c_phase2 = c[keep_cols]
        
        # Формируем базис для фазы 2 из неискусственных переменных
        basis_phase2_mapped = []
        for idx in basis_phase1:
            if idx in keep_cols:
                new_idx = keep_cols.index(idx)
                if new_idx not in basis_phase2_mapped:
                    basis_phase2_mapped.append(new_idx)
        
        # Если базиса недостаточно, добавляем небазисные переменные
        # Проверяем, что базисная матрица невырождена
        while len(basis_phase2_mapped) < n_constraints:
            added = False
            for i in range(len(keep_cols)):
                if i not in basis_phase2_mapped:
                    # Пробуем добавить эту переменную в базис
                    test_basis = basis_phase2_mapped + [i]
                    if len(test_basis) <= n_constraints:
                        B_test = A_phase2[:, test_basis]
                        try:
                            np.linalg.inv(B_test)
                            basis_phase2_mapped.append(i)
                            added = True
                            break
                        except np.linalg.LinAlgError:
                            continue
            if not added:
                # Если не можем добавить переменные, используем текущий базис
                # и добавляем любые оставшиеся переменные
                for i in range(len(keep_cols)):
                    if i not in basis_phase2_mapped:
                        basis_phase2_mapped.append(i)
                        if len(basis_phase2_mapped) >= n_constraints:
                            break
                break
        
        return self.simplex_method(A_phase2, b, c_phase2, basis_phase2_mapped)
    
    def _can_add_to_basis(self, A: np.ndarray, basis: List[int], var_idx: int) -> bool:
        """Проверяет, можно ли добавить переменную в базис."""
        if var_idx in basis:
            return False
        
        # Проверяем, что столбец линейно независим от текущих базисных
        basis_cols = A[:, basis]
        new_col = A[:, var_idx].reshape(-1, 1)
        
        # Простая проверка: если столбец не нулевой и не пропорционален существующим
        if np.allclose(new_col, 0):
            return False
        
        return True
    
    def simplex_method(self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
                      basis: List[int]) -> Tuple[SolutionStatus, Optional[np.ndarray],
                                                 Optional[float], Optional[np.ndarray]]:
        """
        Симплекс-метод для решения ЗЛП в каноническом виде.
        
        Возвращает:
        - status: статус решения
        - x: оптимальный вектор (только исходные переменные)
        - obj_value: оптимальное значение целевой функции
        - basis: финальный базис
        """
        n_constraints, n_vars = A.shape
        max_iterations = 1000
        iteration = 0
        
        # Проверяем, что базис корректен
        if len(basis) != n_constraints:
            return SolutionStatus.INFEASIBLE, None, None, None
        
        while iteration < max_iterations:
            iteration += 1
            
            # Формируем базисную матрицу
            B = A[:, basis]
            
            # Проверяем, что базисная матрица невырождена
            try:
                B_inv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                return SolutionStatus.INFEASIBLE, None, None, None
            
            # Базисное решение
            x_b = B_inv @ b
            
            # Проверяем допустимость базисного решения
            if np.any(x_b < -1e-6):
                # Базисное решение недопустимо - нужно найти новый базис
                # Это может произойти при переходе между фазами
                return SolutionStatus.INFEASIBLE, None, None, None
            
            # Коэффициенты целевой функции для базисных переменных
            c_b = c[basis]
            
            # Вычисляем оценки (reduced costs)
            # delta_j = c_j - c_B * B^(-1) * A_j
            pi = c_b @ B_inv  # Симплекс-множители
            reduced_costs = c - pi @ A
            
            # Проверяем оптимальность
            # Для максимизации: все reduced costs должны быть <= 0
            # Для базисных переменных reduced costs = 0 по определению
            is_optimal = True
            for j in range(n_vars):
                if j not in basis and reduced_costs[j] > 1e-6:
                    is_optimal = False
                    break
            
            if is_optimal:
                # Решение оптимально
                x = np.zeros(n_vars)
                for i, idx in enumerate(basis):
                    x[idx] = x_b[i]
                
                obj_value = c @ x
                
                # Возвращаем только исходные переменные
                x_original = x[:self.n_vars]
                
                # Если была минимизация, меняем знак обратно
                if self.opt_type == OptimizationType.MIN:
                    obj_value = -obj_value
                
                return SolutionStatus.OPTIMAL, x_original, obj_value, basis
            
            # Находим входящую переменную (с положительной оценкой для максимизации)
            entering_idx = None
            max_reduced_cost = -1e10
            for j in range(n_vars):
                if j not in basis and reduced_costs[j] > 1e-6:
                    if reduced_costs[j] > max_reduced_cost:
                        max_reduced_cost = reduced_costs[j]
                        entering_idx = j
            
            if entering_idx is None:
                # Не нашли входящую переменную, но оценки не все неотрицательны
                # Это может быть из-за численных ошибок
                x = np.zeros(n_vars)
                for i, idx in enumerate(basis):
                    x[idx] = x_b[i]
                obj_value = c @ x
                if self.opt_type == OptimizationType.MIN:
                    obj_value = -obj_value
                return SolutionStatus.OPTIMAL, x[:self.n_vars], obj_value, basis
            
            # Находим выходящую переменную (минимальное отношение)
            A_entering = A[:, entering_idx]
            ratios = []
            for i in range(n_constraints):
                if A_entering[i] > 1e-6:
                    ratio = x_b[i] / A_entering[i]
                    ratios.append((ratio, i))
            
            if not ratios:
                # Нет ограничений на рост входящей переменной - задача неограничена
                return SolutionStatus.UNBOUNDED, None, None, None
            
            # Выбираем минимальное положительное отношение
            ratios.sort()
            leaving_pos = ratios[0][1]
            leaving_idx = basis[leaving_pos]
            
            # Обновляем базис
            basis[leaving_pos] = entering_idx
        
        # Превышено максимальное количество итераций
        return SolutionStatus.INFEASIBLE, None, None, None
    
    def solve(self) -> Tuple[SolutionStatus, Optional[np.ndarray], Optional[float]]:
        """
        Решает ЗЛП.
        
        Возвращает:
        - status: статус решения
        - x: оптимальный вектор
        - obj_value: оптимальное значение целевой функции
        """
        # Преобразуем в канонический вид
        A_canon, b_canon, c_canon, basis, artificial_vars = self.to_canonical_form()
        
        # Решаем двухфазным симплекс-методом
        status, x, obj_value, _ = self.two_phase_simplex(A_canon, b_canon, c_canon, basis, artificial_vars)
        
        return status, x, obj_value

