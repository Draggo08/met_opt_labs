"""
Решение задачи динамического программирования для управления инвестиционным портфелем
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Situation(Enum):
    """Типы ситуаций на рынке"""
    FAVORABLE = "благоприятная"
    NEUTRAL = "нейтральная"
    NEGATIVE = "негативная"


@dataclass
class StageData:
    """Данные для одного этапа"""
    probabilities: Dict[Situation, float]
    coefficients: Dict[Situation, Dict[str, float]]  # {situation: {asset: coefficient}}


@dataclass
class State:
    """Состояние портфеля"""
    cb1: float  # ЦБ1
    cb2: float  # ЦБ2
    deposit: float  # Депозиты
    free: float  # Свободные средства
    
    def __hash__(self):
        return hash((round(self.cb1, 2), round(self.cb2, 2), 
                     round(self.deposit, 2), round(self.free, 2)))
    
    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return (abs(self.cb1 - other.cb1) < 1e-6 and
                abs(self.cb2 - other.cb2) < 1e-6 and
                abs(self.deposit - other.deposit) < 1e-6 and
                abs(self.free - other.free) < 1e-6)


@dataclass
class Action:
    """Действие управления"""
    buy_cb1: int = 0  # Количество пакетов ЦБ1 (1 пакет = 25 д.е.)
    buy_cb2: int = 0  # Количество пакетов ЦБ2 (1 пакет = 200 д.е.)
    buy_deposit: int = 0  # Количество пакетов депозитов (1 пакет = 100 д.е.)
    sell_cb1: int = 0
    sell_cb2: int = 0
    sell_deposit: int = 0
    
    def cost(self) -> float:
        """Стоимость действия (сколько нужно свободных средств)"""
        cost = (self.buy_cb1 * 25 + self.buy_cb2 * 200 + 
                self.buy_deposit * 100)
        return cost
    
    def income(self) -> float:
        """Доход от продажи"""
        income = (self.sell_cb1 * 25 + self.sell_cb2 * 200 + 
                  self.sell_deposit * 100)
        return income


class PortfolioOptimizer:
    """Оптимизатор портфеля методом динамического программирования"""
    
    # Размеры пакетов (1/4 от первоначальной стоимости)
    PACKET_CB1 = 25.0  # 100 / 4
    PACKET_CB2 = 200.0  # 800 / 4
    PACKET_DEPOSIT = 100.0  # 400 / 4
    
    def __init__(self, initial_state: State, stages_data: List[StageData]):
        """
        Инициализация оптимизатора
        
        Args:
            initial_state: Начальное состояние портфеля
            stages_data: Данные для каждого этапа (список из 3 элементов)
        """
        self.initial_state = initial_state
        self.stages_data = stages_data
        self.num_stages = len(stages_data)
        
        # Кэш для хранения оптимальных значений функции Беллмана
        # F[stage][state] = (optimal_value, optimal_action)
        self.F: Dict[int, Dict[State, Tuple[float, Optional[Action]]]] = {}
        
    def _get_all_possible_actions(self, state: State) -> List[Action]:
        """
        Генерирует все возможные действия для данного состояния
        
        Ограничения:
        - Нельзя брать кредит (только свободные средства)
        - Можно покупать/продавать только целые пакеты
        - Нельзя продать больше, чем есть
        """
        actions = []
        
        # Максимальное количество пакетов, которые можно купить
        max_buy_cb1 = min(int(state.free / self.PACKET_CB1), 10)  # Ограничение для ускорения
        max_buy_cb2 = min(int(state.free / self.PACKET_CB2), 5)
        max_buy_deposit = min(int(state.free / self.PACKET_DEPOSIT), 8)
        
        # Максимальное количество пакетов, которые можно продать
        max_sell_cb1 = min(int(state.cb1 / self.PACKET_CB1), 10)
        max_sell_cb2 = min(int(state.cb2 / self.PACKET_CB2), 5)
        max_sell_deposit = min(int(state.deposit / self.PACKET_DEPOSIT), 8)
        
        # Добавляем действие "ничего не делать"
        actions.append(Action())
        
        # Генерируем действия только на покупку или только на продажу
        # (реже комбинированные действия для уменьшения сложности)
        
        # Только покупки
        for buy_cb1 in range(max_buy_cb1 + 1):
            for buy_cb2 in range(max_buy_cb2 + 1):
                for buy_deposit in range(max_buy_deposit + 1):
                    buy_cost = (buy_cb1 * self.PACKET_CB1 + 
                               buy_cb2 * self.PACKET_CB2 + 
                               buy_deposit * self.PACKET_DEPOSIT)
                    if buy_cost > state.free or buy_cost == 0:
                        continue
                    actions.append(Action(
                        buy_cb1=buy_cb1,
                        buy_cb2=buy_cb2,
                        buy_deposit=buy_deposit
                    ))
        
        # Только продажи
        for sell_cb1 in range(1, max_sell_cb1 + 1):
            actions.append(Action(sell_cb1=sell_cb1))
        for sell_cb2 in range(1, max_sell_cb2 + 1):
            actions.append(Action(sell_cb2=sell_cb2))
        for sell_deposit in range(1, max_sell_deposit + 1):
            actions.append(Action(sell_deposit=sell_deposit))
        
        # Комбинированные действия (ограниченный набор)
        # Покупка одного типа и продажа другого
        for buy_cb1 in range(1, min(max_buy_cb1, 5) + 1):
            for sell_cb2 in range(1, min(max_sell_cb2, 3) + 1):
                cost = buy_cb1 * self.PACKET_CB1
                income = sell_cb2 * self.PACKET_CB2
                if cost <= state.free + income and sell_cb2 * self.PACKET_CB2 <= state.cb2:
                    actions.append(Action(buy_cb1=buy_cb1, sell_cb2=sell_cb2))
        
        for buy_cb2 in range(1, min(max_buy_cb2, 3) + 1):
            for sell_cb1 in range(1, min(max_sell_cb1, 5) + 1):
                cost = buy_cb2 * self.PACKET_CB2
                income = sell_cb1 * self.PACKET_CB1
                if cost <= state.free + income and sell_cb1 * self.PACKET_CB1 <= state.cb1:
                    actions.append(Action(buy_cb2=buy_cb2, sell_cb1=sell_cb1))
        
        # Фильтруем недопустимые действия
        valid_actions = []
        for action in actions:
            # Проверяем ограничения
            if action.cost() > state.free + action.income():
                continue
            if (action.sell_cb1 * self.PACKET_CB1 > state.cb1 or
                action.sell_cb2 * self.PACKET_CB2 > state.cb2 or
                action.sell_deposit * self.PACKET_DEPOSIT > state.deposit):
                continue
            
            # Проверяем, что после действия все значения неотрицательны
            new_state = self._apply_action(state, action)
            if (new_state.cb1 >= 0 and new_state.cb2 >= 0 and 
                new_state.deposit >= 0 and new_state.free >= 0):
                valid_actions.append(action)
        
        return valid_actions
    
    def _apply_action(self, state: State, action: Action) -> State:
        """Применяет действие к состоянию и возвращает новое состояние"""
        new_cb1 = state.cb1 + action.buy_cb1 * self.PACKET_CB1 - action.sell_cb1 * self.PACKET_CB1
        new_cb2 = state.cb2 + action.buy_cb2 * self.PACKET_CB2 - action.sell_cb2 * self.PACKET_CB2
        new_deposit = (state.deposit + action.buy_deposit * self.PACKET_DEPOSIT - 
                      action.sell_deposit * self.PACKET_DEPOSIT)
        new_free = (state.free - action.cost() + action.income())
        
        return State(cb1=new_cb1, cb2=new_cb2, deposit=new_deposit, free=new_free)
    
    def _apply_situation(self, state: State, stage_data: StageData, 
                        situation: Situation) -> State:
        """Применяет ситуацию к состоянию (изменение стоимости активов)"""
        coeffs = stage_data.coefficients[situation]
        
        new_cb1 = state.cb1 * coeffs['cb1']
        new_cb2 = state.cb2 * coeffs['cb2']
        new_deposit = state.deposit * coeffs['deposit']
        # Свободные средства не изменяются
        
        return State(cb1=new_cb1, cb2=new_cb2, deposit=new_deposit, free=state.free)
    
    def _bellman_value(self, stage: int, state: State) -> Tuple[float, Optional[Action]]:
        """
        Вычисляет значение функции Беллмана для данного этапа и состояния
        
        Использует обратное прохождение (от последнего этапа к первому)
        """
        # Проверяем кэш
        if stage in self.F and state in self.F[stage]:
            return self.F[stage][state]
        
        # Инициализируем кэш для этапа, если его еще нет
        if stage not in self.F:
            self.F[stage] = {}
        
        stage_data = self.stages_data[stage]
        
        if stage == self.num_stages - 1:
            # Последний этап: максимизируем финальное состояние
            # (сумма всех активов)
            optimal_value = state.cb1 + state.cb2 + state.deposit + state.free
            optimal_action = None  # На последнем этапе действий нет
            self.F[stage][state] = (optimal_value, optimal_action)
            return optimal_value, optimal_action
        
        # Для промежуточных этапов используем рекуррентное соотношение Беллмана
        optimal_value = float('-inf')
        optimal_action = None
        
        # Перебираем все возможные действия
        actions = self._get_all_possible_actions(state)
        
        for action in actions:
            # Применяем действие
            state_after_action = self._apply_action(state, action)
            
            # Вычисляем ожидаемое значение (критерий Байеса)
            expected_value = 0.0
            
            for situation in Situation:
                prob = stage_data.probabilities[situation]
                # Применяем ситуацию
                state_after_situation = self._apply_situation(
                    state_after_action, stage_data, situation
                )
                # Рекурсивно вычисляем значение для следующего этапа
                next_value, _ = self._bellman_value(stage + 1, state_after_situation)
                expected_value += prob * next_value
            
            # Обновляем оптимальное значение
            if expected_value > optimal_value:
                optimal_value = expected_value
                optimal_action = action
        
        self.F[stage][state] = (optimal_value, optimal_action)
        return optimal_value, optimal_action
    
    def solve(self) -> Tuple[float, List[Action]]:
        """
        Решает задачу динамического программирования
        
        Returns:
            (optimal_value, optimal_actions): оптимальное значение и последовательность действий
        """
        # Очищаем кэш
        self.F = {}
        
        # Вычисляем оптимальное значение для начального состояния
        optimal_value, _ = self._bellman_value(0, self.initial_state)
        
        # Восстанавливаем оптимальную последовательность действий
        optimal_actions = []
        current_state = self.initial_state
        
        for stage in range(self.num_stages):
            _, action = self._bellman_value(stage, current_state)
            if action is not None:
                optimal_actions.append(action)
                # Применяем действие
                current_state = self._apply_action(current_state, action)
                # Применяем ожидаемую ситуацию (для демонстрации)
                # В реальности ситуация случайна, но для траектории используем среднее
                stage_data = self.stages_data[stage]
                # Используем наиболее вероятную ситуацию для траектории
                most_prob_situation = max(
                    stage_data.probabilities.items(),
                    key=lambda x: x[1]
                )[0]
                current_state = self._apply_situation(
                    current_state, stage_data, most_prob_situation
                )
            else:
                optimal_actions.append(None)
        
        return optimal_value, optimal_actions
    
    def get_final_state(self, actions: List[Action]) -> State:
        """Вычисляет финальное состояние при заданной последовательности действий"""
        state = self.initial_state
        
        for stage, action in enumerate(actions):
            if action is not None:
                state = self._apply_action(state, action)
            
            # Применяем ожидаемую ситуацию
            stage_data = self.stages_data[stage]
            most_prob_situation = max(
                stage_data.probabilities.items(),
                key=lambda x: x[1]
            )[0]
            state = self._apply_situation(state, stage_data, most_prob_situation)
        
        return state


def create_stages_data() -> List[StageData]:
    """Создает данные для всех этапов согласно условию задачи"""
    stages = []
    
    # Этап 1
    stage1 = StageData(
        probabilities={
            Situation.FAVORABLE: 0.60,
            Situation.NEUTRAL: 0.30,
            Situation.NEGATIVE: 0.10
        },
        coefficients={
            Situation.FAVORABLE: {'cb1': 1.20, 'cb2': 1.10, 'deposit': 1.07},
            Situation.NEUTRAL: {'cb1': 1.05, 'cb2': 1.02, 'deposit': 1.03},
            Situation.NEGATIVE: {'cb1': 0.80, 'cb2': 0.95, 'deposit': 1.00}
        }
    )
    stages.append(stage1)
    
    # Этап 2
    stage2 = StageData(
        probabilities={
            Situation.FAVORABLE: 0.30,
            Situation.NEUTRAL: 0.20,
            Situation.NEGATIVE: 0.50
        },
        coefficients={
            Situation.FAVORABLE: {'cb1': 1.4, 'cb2': 1.15, 'deposit': 1.01},
            Situation.NEUTRAL: {'cb1': 1.05, 'cb2': 1.00, 'deposit': 1.00},
            Situation.NEGATIVE: {'cb1': 0.60, 'cb2': 0.90, 'deposit': 1.00}
        }
    )
    stages.append(stage2)
    
    # Этап 3
    stage3 = StageData(
        probabilities={
            Situation.FAVORABLE: 0.40,
            Situation.NEUTRAL: 0.40,
            Situation.NEGATIVE: 0.20
        },
        coefficients={
            Situation.FAVORABLE: {'cb1': 1.15, 'cb2': 1.12, 'deposit': 1.05},
            Situation.NEUTRAL: {'cb1': 1.05, 'cb2': 1.01, 'deposit': 1.01},
            Situation.NEGATIVE: {'cb1': 0.70, 'cb2': 0.94, 'deposit': 1.00}
        }
    )
    stages.append(stage3)
    
    return stages


def main():
    """Основная функция для запуска программы"""
    # Начальное состояние
    initial_state = State(
        cb1=100.0,      # ЦБ1
        cb2=800.0,      # ЦБ2
        deposit=400.0,  # Депозиты
        free=600.0      # Свободные средства
    )
    
    # Данные для этапов
    stages_data = create_stages_data()
    
    print("=" * 80)
    print("РЕШЕНИЕ ЗАДАЧИ ДИНАМИЧЕСКОГО ПРОГРАММИРОВАНИЯ")
    print("Управление инвестиционным портфелем")
    print("=" * 80)
    print()
    
    print("Начальное состояние портфеля:")
    print(f"  ЦБ1: {initial_state.cb1} д.е.")
    print(f"  ЦБ2: {initial_state.cb2} д.е.")
    print(f"  Депозиты: {initial_state.deposit} д.е.")
    print(f"  Свободные средства: {initial_state.free} д.е.")
    print(f"  Всего: {initial_state.cb1 + initial_state.cb2 + initial_state.deposit + initial_state.free} д.е.")
    print()
    
    # Создаем оптимизатор
    optimizer = PortfolioOptimizer(initial_state, stages_data)
    
    print("Решение задачи...")
    print("(Это может занять некоторое время из-за большого пространства состояний)")
    print()
    
    # Решаем задачу
    optimal_value, optimal_actions = optimizer.solve()
    
    print("=" * 80)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 80)
    print()
    print(f"Максимальный ожидаемый доход (критерий Байеса): {optimal_value:.2f} д.е.")
    print(f"Начальная сумма: {initial_state.cb1 + initial_state.cb2 + initial_state.deposit + initial_state.free:.2f} д.е.")
    print(f"Ожидаемый прирост: {optimal_value - (initial_state.cb1 + initial_state.cb2 + initial_state.deposit + initial_state.free):.2f} д.е.")
    print(f"Ожидаемая доходность: {(optimal_value / (initial_state.cb1 + initial_state.cb2 + initial_state.deposit + initial_state.free) - 1) * 100:.2f}%")
    print()
    
    print("Оптимальная стратегия управления:")
    print()
    
    current_state = initial_state
    for stage in range(len(stages_data)):
        print(f"--- ЭТАП {stage + 1} ---")
        print(f"Состояние перед этапом:")
        print(f"  ЦБ1: {current_state.cb1:.2f} д.е.")
        print(f"  ЦБ2: {current_state.cb2:.2f} д.е.")
        print(f"  Депозиты: {current_state.deposit:.2f} д.е.")
        print(f"  Свободные средства: {current_state.free:.2f} д.е.")
        print()
        
        action = optimal_actions[stage]
        if action is not None:
            print("Действие:")
            if action.buy_cb1 > 0:
                print(f"  Купить ЦБ1: {action.buy_cb1} пакет(ов) × {optimizer.PACKET_CB1} = {action.buy_cb1 * optimizer.PACKET_CB1} д.е.")
            if action.buy_cb2 > 0:
                print(f"  Купить ЦБ2: {action.buy_cb2} пакет(ов) × {optimizer.PACKET_CB2} = {action.buy_cb2 * optimizer.PACKET_CB2} д.е.")
            if action.buy_deposit > 0:
                print(f"  Купить депозиты: {action.buy_deposit} пакет(ов) × {optimizer.PACKET_DEPOSIT} = {action.buy_deposit * optimizer.PACKET_DEPOSIT} д.е.")
            if action.sell_cb1 > 0:
                print(f"  Продать ЦБ1: {action.sell_cb1} пакет(ов) × {optimizer.PACKET_CB1} = {action.sell_cb1 * optimizer.PACKET_CB1} д.е.")
            if action.sell_cb2 > 0:
                print(f"  Продать ЦБ2: {action.sell_cb2} пакет(ов) × {optimizer.PACKET_CB2} = {action.sell_cb2 * optimizer.PACKET_CB2} д.е.")
            if action.sell_deposit > 0:
                print(f"  Продать депозиты: {action.sell_deposit} пакет(ов) × {optimizer.PACKET_DEPOSIT} = {action.sell_deposit * optimizer.PACKET_DEPOSIT} д.е.")
            if (action.buy_cb1 == 0 and action.buy_cb2 == 0 and action.buy_deposit == 0 and
                action.sell_cb1 == 0 and action.sell_cb2 == 0 and action.sell_deposit == 0):
                print("  Бездействие (не изменять портфель)")
        else:
            print("Действие: Нет (последний этап)")
        
        print()
        
        # Применяем действие
        if action is not None:
            current_state = optimizer._apply_action(current_state, action)
        
        # Показываем состояние после действия
        print(f"Состояние после действия:")
        print(f"  ЦБ1: {current_state.cb1:.2f} д.е.")
        print(f"  ЦБ2: {current_state.cb2:.2f} д.е.")
        print(f"  Депозиты: {current_state.deposit:.2f} д.е.")
        print(f"  Свободные средства: {current_state.free:.2f} д.е.")
        print()
        
        # Показываем возможные исходы с вероятностями
        stage_data = stages_data[stage]
        print("Возможные исходы после реализации ситуаций:")
        for situation, prob in sorted(stage_data.probabilities.items(), key=lambda x: -x[1]):
            temp_state = optimizer._apply_situation(current_state, stage_data, situation)
            total = temp_state.cb1 + temp_state.cb2 + temp_state.deposit + temp_state.free
            print(f"  {situation.value} (вероятность {prob:.0%}): итого {total:.2f} д.е.")
        
        # Применяем наиболее вероятную ситуацию для демонстрации траектории
        most_prob_situation = max(stage_data.probabilities.items(), key=lambda x: x[1])[0]
        current_state = optimizer._apply_situation(current_state, stage_data, most_prob_situation)
        
        print(f"\nТраектория (наиболее вероятная ситуация - {most_prob_situation.value}):")
        print(f"  ЦБ1: {current_state.cb1:.2f} д.е.")
        print(f"  ЦБ2: {current_state.cb2:.2f} д.е.")
        print(f"  Депозиты: {current_state.deposit:.2f} д.е.")
        print(f"  Свободные средства: {current_state.free:.2f} д.е.")
        print(f"  Всего: {current_state.cb1 + current_state.cb2 + current_state.deposit + current_state.free:.2f} д.е.")
        print()
    
    print("=" * 80)
    print("ФИНАЛЬНОЕ СОСТОЯНИЕ ПОРТФЕЛЯ")
    print("=" * 80)
    print(f"  ЦБ1: {current_state.cb1:.2f} д.е.")
    print(f"  ЦБ2: {current_state.cb2:.2f} д.е.")
    print(f"  Депозиты: {current_state.deposit:.2f} д.е.")
    print(f"  Свободные средства: {current_state.free:.2f} д.е.")
    print(f"  ИТОГО: {current_state.cb1 + current_state.cb2 + current_state.deposit + current_state.free:.2f} д.е.")
    print()


if __name__ == "__main__":
    main()
