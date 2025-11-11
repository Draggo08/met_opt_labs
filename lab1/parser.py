"""
Парсер для чтения ЗЛП из текстового файла.
Формат файла:
max/min: c1 x1 + c2 x2 + ... + cn xn
subject to:
a11 x1 + a12 x2 + ... + a1n xn <=/>=/= b1
a21 x1 + a22 x2 + ... + a2n xn <=/>=/= b2
...
"""

import re
from typing import Dict
from simplex import LinearProgrammingSolver, OptimizationType, ConstraintType


def parse_lp_file(filename: str) -> Dict:
    """
    Парсит файл с ЗЛП.
    
    Формат:
    max: 3 x1 + 2 x2
    subject to:
    2 x1 + x2 <= 10
    x1 + 2 x2 <= 8
    x1 >= 0
    x2 >= 0
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Удаляем пустые строки и комментарии
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    if not lines:
        raise ValueError("Файл пуст")
    
    # Парсим целевую функцию
    obj_line = lines[0].lower()
    if obj_line.startswith('max:'):
        opt_type = 'max'
        obj_str = obj_line[4:].strip()
    elif obj_line.startswith('min:'):
        opt_type = 'min'
        obj_str = obj_line[4:].strip()
    else:
        raise ValueError("Первая строка должна начинаться с 'max:' или 'min:'")
    
    # Парсим коэффициенты целевой функции
    coeffs = _parse_expression(obj_str)
    
    # Парсим ограничения
    constraints = []
    i = 1
    if i < len(lines) and lines[i].lower().startswith('subject to:'):
        i += 1
    
    while i < len(lines):
        constraint_line = lines[i]
        
        # Пропускаем ограничения неотрицательности (x >= 0)
        if re.match(r'^x\d+\s*>=\s*0$', constraint_line, re.IGNORECASE):
            i += 1
            continue
        
        # Парсим ограничение
        constraint = _parse_constraint(constraint_line, len(coeffs))
        if constraint:
            constraints.append(constraint)
        
        i += 1
    
    return {
        'objective': {
            'coeffs': coeffs,
            'type': opt_type
        },
        'constraints': constraints
    }


def _parse_expression(expr: str) -> list:
    """Парсит выражение вида 'c1 x1 + c2 x2 + ...' и возвращает список коэффициентов."""
    var_pattern = r'x(\d+)'
    matches = re.findall(var_pattern, expr)
    
    if not matches:
        return []
    
    max_idx = max(int(m) for m in matches)
    coeffs = [0.0] * max_idx
    
    # Разбиваем выражение на члены с учетом знаков
    # Используем более простой подход: ищем все пары (коэффициент, переменная)
    # Разделяем по знакам + и -, но сохраняем их
    parts = re.split(r'([+-])', expr)
    
    # Обрабатываем первый член отдельно
    i = 0
    current_sign = 1
    
    # Если выражение начинается с минуса
    if parts and parts[0].strip() == '' and len(parts) > 1 and parts[1] == '-':
        current_sign = -1
        i = 2
    elif parts and parts[0].strip() == '' and len(parts) > 1 and parts[1] == '+':
        current_sign = 1
        i = 2
    
    # Обрабатываем первый член (если он есть)
    if i < len(parts) and parts[i].strip():
        first_part = parts[i].strip()
        var_match = re.search(var_pattern, first_part)
        if var_match:
            var_idx = int(var_match.group(1)) - 1
            coeff_str = first_part[:var_match.start()].strip()
            if not coeff_str:
                coeff = 1.0
            else:
                coeff_str = coeff_str.replace(' ', '')
                try:
                    coeff = float(coeff_str)
                except ValueError:
                    coeff = 1.0
            coeffs[var_idx] = current_sign * coeff
        i += 1
    
    # Обрабатываем остальные члены
    while i < len(parts):
        if parts[i] == '+':
            current_sign = 1
        elif parts[i] == '-':
            current_sign = -1
        else:
            part = parts[i].strip()
            if part:
                var_match = re.search(var_pattern, part)
                if var_match:
                    var_idx = int(var_match.group(1)) - 1
                    coeff_str = part[:var_match.start()].strip()
                    if not coeff_str:
                        coeff = 1.0
                    else:
                        coeff_str = coeff_str.replace(' ', '')
                        try:
                            coeff = float(coeff_str)
                        except ValueError:
                            coeff = 1.0
                    coeffs[var_idx] = current_sign * coeff
        i += 1
    
    return coeffs


def _parse_constraint(line: str, n_vars: int) -> Dict:
    """Парсит ограничение вида 'a1 x1 + a2 x2 + ... <=/>=/= b'."""
    # Ищем знак отношения
    if '<=' in line:
        op = '<='
    elif '>=' in line:
        op = '>='
    elif '=' in line and '<=' not in line and '>=' not in line:
        op = '='
    else:
        return None
    
    # Разделяем на левую и правую части
    parts = re.split(r'<=|>=|=', line, 1)
    if len(parts) != 2:
        return None
    
    left_expr = parts[0].strip()
    right_expr = parts[1].strip()
    
    # Парсим левую часть
    coeffs = _parse_expression(left_expr)
    
    # Дополняем до нужной длины
    while len(coeffs) < n_vars:
        coeffs.append(0.0)
    
    # Парсим правую часть (свободный член)
    rhs = float(right_expr)
    
    return {
        'coeffs': coeffs[:n_vars],
        'type': op,
        'rhs': rhs
    }

