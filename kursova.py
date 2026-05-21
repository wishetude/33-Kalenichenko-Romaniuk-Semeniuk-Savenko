import random
import json
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


# =============================================================
# КЛАС ЕЛЕМЕНТА ІНФРАСТРУКТУРИ
# =============================================================

class Element:
    """
    Представляє один кандидат-елемент ІТ-інфраструктури.
    Поля: назва, вартість, продуктивність, надійність, важливість.
    """

    def __init__(self, name, cost, productivity, reliability, weight):
        self.name         = name
        self.cost         = cost
        self.productivity = productivity
        self.reliability  = reliability
        self.weight       = weight
        self.efficiency   = 0.0      

    def to_dict(self):
        return {
            "name":         self.name,
            "cost":         self.cost,
            "productivity": self.productivity,
            "reliability":  self.reliability,
            "weight":       self.weight
        }

    @staticmethod
    def from_dict(d):
        return Element(
            d["name"],
            d["cost"],
            d["productivity"],
            d["reliability"],
            d["weight"]
        )

    def __str__(self):
        return (
            f"  {self.name:<30} | "
            f"c={self.cost:>9.2f} | "
            f"p={self.productivity:>7.2f} | "
            f"r={self.reliability:.4f} | "
            f"w={self.weight:>4} | "
            f"k={self.efficiency:.6f}"
        )


# =============================================================
# КЛАС ОСОБИНИ ГЕНЕТИЧНОГО АЛГОРИТМУ
# =============================================================

class Individual:
    """
    Булевий вектор генів довжини n.
    genes[j] = 1 — j-й елемент обрано, 0 — ні.
    """

    def __init__(self, genes):
        self.genes        = genes[:]
        self.total_cost   = 0.0
        self.total_weight = 0.0
        self.fitness_1    = 0.0   # F1: сумарна продуктивність
        self.fitness_2    = 0.0   # F2: інтегральна надійність

    def copy(self):
        return deepcopy(self)


# =============================================================
# ДОПОМІЖНИЙ КЛАС ДЛЯ ПОРІВНЯННЯ РЕЗУЛЬТАТІВ ЖА З ПСК
# =============================================================

class _SolutionProxy:
    """
    Тонка обгортка, яка надає інтерфейс Individual
    для результату жадібного алгоритму (словника).
    """

    def __init__(self, f1, f2):
        self.fitness_1 = f1
        self.fitness_2 = f2


# =============================================================
# ОБЧИСЛЕННЯ КОЕФІЦІЄНТА ЕФЕКТИВНОСТІ  
# =============================================================

def calculate_efficiency(elements):
    """
    kj = (p'j + r'j + wj) / cj
    де p'j та r'j — нормовані значення продуктивності та надійності.
    """
    p_vals = [e.productivity for e in elements]
    r_vals = [e.reliability  for e in elements]

    p_max, p_min = max(p_vals), min(p_vals)
    r_max, r_min = max(r_vals), min(r_vals)

    for e in elements:
        p_norm = (
            (e.productivity - p_min) / (p_max - p_min)
            if p_max != p_min else 1.0
        )
        r_norm = (
            (e.reliability - r_min) / (r_max - r_min)
            if r_max != r_min else 1.0
        )
        e.efficiency = (p_norm + r_norm + e.weight) / e.cost


# =============================================================
# ОЦІНЮВАННЯ ОСОБИНИ   
# =============================================================

def evaluate(individual, elements):
    """
    F1(x) = Σ pj·xj
    F2(x) = Σ(rj·wj·xj) / n
    """
    n = len(elements)
    total_cost   = 0.0
    total_weight = 0.0
    f1           = 0.0
    weighted_rel = 0.0

    for i, gene in enumerate(individual.genes):
        if gene == 1:
            e = elements[i]
            total_cost   += e.cost
            total_weight += e.weight
            f1           += e.productivity
            weighted_rel += e.reliability * e.weight

    individual.total_cost   = total_cost
    individual.total_weight = total_weight
    individual.fitness_1    = f1
    individual.fitness_2    = weighted_rel / n if n > 0 else 0.0

    return individual


# =============================================================
# ПЕРЕВІРКА ДОПУСТИМОСТІ РОЗВ'ЯЗКУ
# =============================================================

def is_valid(individual, budget, min_weight):
    return (
        individual.total_cost   <= budget
        and individual.total_weight >= min_weight
    )


# =============================================================
# ПРИНЦИП СПРАВЕДЛИВОГО КОМПРОМІСУ (ПСК) 
# =============================================================

def psk_compare(sol1, sol2, label1="x'", label2="x''"):
    EPS = 1e-9

    f1p = sol1.fitness_1
    f2p = sol1.fitness_2
    f1d = sol2.fitness_1
    f2d = sol2.fitness_2

    f1_min = min(f1p, f1d)
    f2_min = min(f2p, f2d)

    v1 = (f1p - f1d) / f1_min if f1_min > EPS else 0.0
    v2 = (f2p - f2d) / f2_min if f2_min > EPS else 0.0
    total = v1 + v2

    if abs(total) < EPS:
        winner      = 0
        conclusion  = "Оскільки v1+v2 ≈ 0, розв'язки рівнозначні за ПСК"
    elif total > 0:
        winner      = 1
        conclusion  = f"Оскільки v1+v2 > 0, кращим є {label1}"
    else:
        winner      = 2
        conclusion  = f"Оскільки v1+v2 < 0, кращим є {label2}"

    return dict(f1p=f1p, f2p=f2p, f1d=f1d, f2d=f2d,
                f1_min=f1_min, f2_min=f2_min,
                v1=v1, v2=v2, total=total,
                winner=winner, label1=label1, label2=label2,
                conclusion=conclusion)


def better(sol1, sol2):
    return psk_compare(sol1, sol2)['winner'] == 1


def print_psk(psk, indent="  "):
    L1  = psk['label1']
    L2  = psk['label2']
    bar = indent + "─" * 62
    print(f"\n{bar}")
    print(f"{indent}ПРИНЦИП СПРАВЕДЛИВОГО КОМПРОМІСУ (ПСК)")
    print(bar)

    if psk['winner'] == 0 and psk['v1'] == 0.0 and psk['v2'] == 0.0:
        print(f"{indent}{psk['conclusion']}")
        print(bar)
        return

    print(f"\n{indent}{L1}:  "
          f"F1 = {psk['f1p']:.4f}   F2 = {psk['f2p']:.6f}")
    print(f"{indent}{L2}:  "
          f"F1 = {psk['f1d']:.4f}   F2 = {psk['f2d']:.6f}")
    print()

    num1 = psk['f1p'] - psk['f1d']
    print(f"{indent}v1 = (F1({L1}) - F1({L2})) / min{{F1({L1}), F1({L2})}}")
    print(f"{indent}   = ({psk['f1p']:.4f} - {psk['f1d']:.4f}) / {psk['f1_min']:.4f}")
    print(f"{indent}   = {num1:+.4f} / {psk['f1_min']:.4f} = {psk['v1']:+.6f}")
    print()

    num2 = psk['f2p'] - psk['f2d']
    print(f"{indent}v2 = (F2({L1}) - F2({L2})) / min{{F2({L1}), F2({L2})}}")
    print(f"{indent}   = ({psk['f2p']:.6f} - {psk['f2d']:.6f}) / {psk['f2_min']:.6f}")
    print(f"{indent}   = {num2:+.6f} / {psk['f2_min']:.6f} = {psk['v2']:+.6f}")
    print()

    EPS  = 1e-9
    if psk['total'] > EPS:
        sign = ">"
    elif psk['total'] < -EPS:
        sign = "<"
    else:
        sign = "≈ 0,"
    print(f"{indent}v1 + v2 = {psk['v1']:+.6f} + {psk['v2']:+.6f} "
          f"= {psk['total']:+.6f} {sign} 0")
    print()
    print(f"{indent}► {psk['conclusion']}")
    print(bar)


# =============================================================
# АЛГОРИТМ РЕАНІМАЦІЇ   
# =============================================================

def reanimate(individual, elements, budget, min_weight):
    while individual.total_cost > budget:
        selected = [i for i, g in enumerate(individual.genes) if g == 1]
        if not selected:
            break
        worst = min(selected, key=lambda i: elements[i].efficiency)
        individual.genes[worst] = 0
        evaluate(individual, elements)

    while individual.total_weight < min_weight:
        unselected = [i for i, g in enumerate(individual.genes) if g == 0]
        if not unselected:
            individual.genes = [0] * len(individual.genes)
            evaluate(individual, elements)
            return individual

        best = max(unselected, key=lambda i: elements[i].efficiency)

        if individual.total_cost + elements[best].cost <= budget:
            individual.genes[best] = 1
            evaluate(individual, elements)
        else:
            individual.genes = [0] * len(individual.genes)
            evaluate(individual, elements)
            return individual

    return individual


# =============================================================
# ЖАДІБНИЙ АЛГОРИТМ  
# =============================================================

def _psk_best_candidate(candidates, selected_so_far, n_total):
    base_f1 = sum(e.productivity           for e in selected_so_far)
    base_rw = sum(e.reliability * e.weight for e in selected_so_far)

    projections = []
    for e in candidates:
        f1 = base_f1 + e.productivity
        f2 = (base_rw + e.reliability * e.weight) / n_total if n_total > 0 else 0.0
        projections.append((e, f1, f2))

    non_dominated = []
    for i, (ei, f1i, f2i) in enumerate(projections):
        dominated = False
        for j, (ej, f1j, f2j) in enumerate(projections):
            if i == j:
                continue
            sol_i = _SolutionProxy(f1i, f2i)
            sol_j = _SolutionProxy(f1j, f2j)
            if better(sol_j, sol_i):
                dominated = True
                break
        if not dominated:
            non_dominated.append((ei, f1i, f2i))

    if len(non_dominated) == 1:
        return non_dominated[0][0]

    return max(non_dominated, key=lambda t: t[0].efficiency)[0]


def greedy_algorithm(elements, budget, min_weight):
    calculate_efficiency(elements)

    n            = len(elements)
    remaining    = list(elements)
    selected     = []
    total_cost   = 0.0
    total_weight = 0.0

    while remaining:
        candidates = [
            e for e in remaining
            if total_cost + e.cost <= budget
        ]
        if not candidates:
            break

        best = _psk_best_candidate(candidates, selected, n)

        selected.append(best)
        total_cost   += best.cost
        total_weight += best.weight
        remaining.remove(best)

    if total_weight < min_weight:
        return None

    f1 = sum(e.productivity           for e in selected)
    f2 = sum(e.reliability * e.weight for e in selected) / n

    return {
        "selected":     selected,
        "total_cost":   total_cost,
        "total_weight": total_weight,
        "f1":           f1,
        "f2":           f2
    }


# =============================================================
# ГЕНЕРАЦІЯ ПОЧАТКОВОЇ ПОПУЛЯЦІЇ  
# =============================================================

def generate_population(pop_size, elements, budget, min_weight):
    n          = len(elements)
    population = []

    for _ in range(pop_size):
        genes = [1 if random.random() > 0.5 else 0 for _ in range(n)]
        ind   = Individual(genes)
        evaluate(ind, elements)

        if not is_valid(ind, budget, min_weight):
            reanimate(ind, elements, budget, min_weight)

        population.append(ind)

    return population


# =============================================================
# ТУРНІРНИЙ ВІДБІР БАТЬКІВ  
# =============================================================

def tournament_selection(population, k):
    tournament = random.sample(population, min(k, len(population)))
    best       = tournament[0]
    for ind in tournament[1:]:
        if better(ind, best):
            best = ind
    return best


def select_parents(population, k):
    p1 = tournament_selection(population, k)
    attempts = 0
    while True:
        p2 = tournament_selection(population, k)
        if p2.genes != p1.genes or attempts > 10:
            break
        attempts += 1
    return p1, p2


# =============================================================
# ОДНОТОЧКОВЕ СХРЕЩУВАННЯ   
# =============================================================

def crossover(p1, p2):
    n     = len(p1.genes)
    point = random.randint(1, n - 1)
    c1    = Individual(p1.genes[:point] + p2.genes[point:])
    c2    = Individual(p2.genes[:point] + p1.genes[point:])
    return c1, c2


# =============================================================
# МУТАЦІЯ  
# =============================================================

def mutate(individual):
    k = random.randint(0, len(individual.genes) - 1)
    individual.genes[k] = 1 - individual.genes[k]
    return individual


# =============================================================
# ГЕНЕТИЧНИЙ АЛГОРИТМ  
# =============================================================

def genetic_algorithm(
        elements,
        budget,
        min_weight,
        pop_size=20,
        max_gen=100,
        mutation_prob=0.1,
        tournament_size=3,
        show_progress=False):
    calculate_efficiency(elements)

    population = generate_population(pop_size, elements, budget, min_weight)

    best = population[0].copy()
    for ind in population:
        if better(ind, best):
            best = ind.copy()
    
    improvement_log = []

    if show_progress:
        print("\n" + THIN)
        print("  Динаміка покращення")
        print(THIN)
        print(
            f"  Початковий рекорд: "
            f"F1 = {best.fitness_1:.4f}, "
            f"F2 = {best.fitness_2:.6f}, "
            f"вартість = {best.total_cost:.2f}, "
            f"важливість = {best.total_weight:.0f}"
        )
        print(THIN)
        print(
            f"  {'Покоління':>10} | {'F1':>11} | {'F2':>11} | "
            f"{'v1':>10} | {'v2':>10} | {'v1+v2':>10} | Статус"
        )
        print(THIN)

    for gen in range(1, max_gen + 1):
        new_pop = []

        while len(new_pop) < pop_size:

            p1, p2 = select_parents(population, tournament_size)

            c1, c2 = crossover(p1, p2)
            evaluate(c1, elements)
            evaluate(c2, elements)

            if not is_valid(c1, budget, min_weight):
                reanimate(c1, elements, budget, min_weight)
            if not is_valid(c2, budget, min_weight):
                reanimate(c2, elements, budget, min_weight)

            if random.random() < mutation_prob:
                mutate(c1)
                evaluate(c1, elements)
                if not is_valid(c1, budget, min_weight):
                    reanimate(c1, elements, budget, min_weight)

            if random.random() < mutation_prob:
                mutate(c2)
                evaluate(c2, elements)
                if not is_valid(c2, budget, min_weight):
                    reanimate(c2, elements, budget, min_weight)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

        cur_best = population[0]
        for ind in population:
            if better(ind, cur_best):
                cur_best = ind

        if better(cur_best, best):
            psk = psk_compare(
                cur_best,
                best,
                label1="новий рекорд",
                label2="попередній рекорд"
            )

            improvement_log.append({
                "generation": gen,
                "f1": cur_best.fitness_1,
                "f2": cur_best.fitness_2,
                "total_cost": cur_best.total_cost,
                "total_weight": cur_best.total_weight,
                "v1": psk["v1"],
                "v2": psk["v2"],
                "total_v": psk["total"]
            })

            if show_progress:
                print(
                    f"  {gen:>10} | "
                    f"{cur_best.fitness_1:>11.4f} | "
                    f"{cur_best.fitness_2:>11.6f} | "
                    f"{psk['v1']:>+10.6f} | "
                    f"{psk['v2']:>+10.6f} | "
                    f"{psk['total']:>+10.6f} | "
                    f"ПОКРАЩЕННЯ"
                )

            best = cur_best.copy()

    best.improvement_log = improvement_log

    if show_progress:
            print(THIN)
            if improvement_log:
                print(f"  Усього покращень рекорду: {len(improvement_log)}")
                last = improvement_log[-1]
                print(
                    f"  Останнє покращення: покоління {last['generation']}, "
                    f"F1 = {last['f1']:.4f}, "
                    f"F2 = {last['f2']:.6f}, "
                    f"v1 + v2 = {last['total_v']:+.6f}"
                )
            else:
                print("  Нових покращень після початкової популяції не було.")
            print(THIN)

    return best


# =============================================================
# ГЕНЕРАТОР ІНДИВІДУАЛЬНИХ ЗАДАЧ   
# =============================================================

def generate_task(
        n,
        c_range=(100, 1000),
        p_range=(10,  200),
        r_range=(0.80, 0.999),
        w_range=(1,   15),
        kB=0.6,
        kW=0.5):
    elements = []
    sum_c    = 0.0
    sum_w    = 0

    for i in range(n):
        c = round(random.uniform(*c_range), 2)
        p = round(random.uniform(*p_range), 2)
        r = round(random.uniform(*r_range), 4)
        w = random.randint(*w_range)

        elements.append(Element(f"E{i + 1}", c, p, r, w))
        sum_c += c
        sum_w += w

    budget     = round(sum_c * kB, 2)
    min_weight = max(1, int(sum_w * kW))

    return elements, budget, min_weight


# =============================================================
# ЗБЕРЕЖЕННЯ / ЗАВАНТАЖЕННЯ ЗАДАЧІ   (розд. 4.1)
# =============================================================

def save_task(filename, elements, budget, min_weight):
    data = {
        "budget":     budget,
        "min_weight": min_weight,
        "elements":   [e.to_dict() for e in elements]
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_task(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    elements = [Element.from_dict(d) for d in data["elements"]]
    return elements, data["budget"], data["min_weight"]


# =============================================================
# ФОРМАТОВАНИЙ ВИВІД
# =============================================================

SEP  = "=" * 72
THIN = "-" * 72


def _title(text):
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)


def print_elements(elements, budget, min_weight):
    _title("ПОТОЧНІ ДАНІ ЗАДАЧІ")
    print(
        f"  {'Назва':<30} | {'Вартість':>9} | "
        f"{'Прод.':>7} | {'Надійн.':>7} | {'Важл.':>4} | k"
    )
    print(THIN)
    for e in elements:
        print(e)
    print(THIN)
    print(f"  Бюджет B = {budget:.2f}   |   Мін. важливість Wmin = {min_weight}")


def print_solution(label, selected, total_cost, total_weight,
                   f1, f2, elapsed=None):
    print(f"\n{THIN}")
    print(f"  {label}")
    print(THIN)

    if not selected:
        print("  Допустимого розв'язку не знайдено.")
        return

    print(
        f"  {'Назва':<30} | {'Вартість':>9} | "
        f"{'Прод.':>7} | {'Надійн.':>7} | {'Важл.':>4}"
    )
    print(THIN)
    for e in selected:
        print(
            f"  {e.name:<30} | {e.cost:>9.2f} | "
            f"{e.productivity:>7.2f} | {e.reliability:>7.4f} | {e.weight:>4}"
        )
    print(THIN)
    print(f"  Сумарна вартість  : {total_cost:.2f}")
    print(f"  Сумарна важливість: {total_weight:.0f}")
    print(f"  F1 (продуктивність): {f1:.4f}")
    print(f"  F2 (надійність)   : {f2:.6f}")
    if elapsed is not None:
        print(f"  Час виконання     : {elapsed * 1000:.4f} мс")


def print_comparison(gr, ga_sol, elements, t_gr, t_ga):
    _title("ПОРІВНЯННЯ АЛГОРИТМІВ")

    if gr:
        print_solution(
            "ЖАДІБНИЙ АЛГОРИТМ  (x')",
            gr["selected"], gr["total_cost"], gr["total_weight"],
            gr["f1"], gr["f2"], t_gr
        )
    else:
        print(f"\n{THIN}")
        print("  ЖАДІБНИЙ АЛГОРИТМ: допустимого розв'язку не знайдено.")

    selected_ga = [elements[i] for i, g in enumerate(ga_sol.genes) if g == 1]
    if selected_ga:
        print_solution(
            "ГЕНЕТИЧНИЙ АЛГОРИТМ  (x'')",
            selected_ga, ga_sol.total_cost, ga_sol.total_weight,
            ga_sol.fitness_1, ga_sol.fitness_2, t_ga
        )
    else:
        print(f"\n{THIN}")
        print("  ГЕНЕТИЧНИЙ АЛГОРИТМ: допустимого розв'язку не знайдено.")

    if gr and selected_ga:
        gr_proxy = _SolutionProxy(gr["f1"], gr["f2"])
        psk = psk_compare(gr_proxy, ga_sol,
                          label1="ЖА (x')", label2="ГА (x'')")
        print_psk(psk)

    print(THIN)


# =============================================================
# ДОПОМІЖНА ФУНКЦІЯ: рядок прогресу
# =============================================================

def _progress(current, total, prefix="", width=30):
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix}[{bar}] {current}/{total}", end="", flush=True)


# =============================================================
# ЕКСПЕРИМЕНТ 1: ВПЛИВ ПАРАМЕТРА MaxGen   
# =============================================================

def _plot_maxgen_experiment(results, n, R):
    """
    Будує два графіки для експерименту MaxGen:
    1) спільний графік F1 і F2 з двома осями Y;
    2) окремий графік середнього часу виконання.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    maxgen_values = [r['max_gen'] for r in results]
    avg_f1 = [r['avg_f1'] for r in results]
    avg_f2 = [r['avg_f2'] for r in results]
    avg_time = [r['avg_time'] for r in results]

    # ---------- Графік 1: MaxGen -> F1 і F2 ----------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    c_f1 = "#5A73A8"
    c_f2 = "#DF665C"

    line1 = ax1.plot(
        maxgen_values, avg_f1,
        marker='o',
        linestyle='-',
        color=c_f1,
        linewidth=2,
        markersize=7,
        label='Середнє F1'
    )

    ax1.set_xlabel("Кількість поколінь MaxGen", fontsize=12, labelpad=10)
    ax1.set_ylabel("Середнє F1 (продуктивність)", fontsize=12, labelpad=10, color=c_f1)
    ax1.tick_params(axis='y', labelcolor=c_f1)
    ax1.set_xticks(maxgen_values)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        maxgen_values, avg_f2,
        marker='s',
        linestyle='--',
        color=c_f2,
        linewidth=2,
        markersize=7,
        label='Середнє F2'
    )

    ax2.set_ylabel("Середнє F2 (надійність)", fontsize=12, labelpad=10, color=c_f2)
    ax2.tick_params(axis='y', labelcolor=c_f2)

    ax1.set_title(
        f"Вплив MaxGen на середні значення F1 та F2 (n={n}, R={R})",
        fontsize=14,
        pad=20
    )

    ax1.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.5)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    ax1.legend(
        lines,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()
    out = "maxgen_f1_f2.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\n  ✓ Графік F1/F2 для MaxGen збережено: {out}")
    plt.show()

    # ---------- Графік 2: MaxGen -> час ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_time = "#6A9955"

    ax.plot(
        maxgen_values, avg_time,
        marker='o',
        linestyle='-',
        color=c_time,
        linewidth=2,
        markersize=7
    )

    ax.set_title(
        f"Вплив MaxGen на середній час виконання (n={n}, R={R})",
        fontsize=14,
        pad=20
    )
    ax.set_xlabel("Кількість поколінь MaxGen", fontsize=12, labelpad=10)
    ax.set_ylabel("Середній час виконання, мс", fontsize=12, labelpad=10)
    ax.set_xticks(maxgen_values)

    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, value in enumerate(avg_time):
        ax.annotate(
            f"{value:.2f}",
            (maxgen_values[i], avg_time[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            color=c_time
        )

    plt.tight_layout()
    out = "maxgen_time.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"  ✓ Графік часу для MaxGen збережено: {out}")
    plt.show()

def experiment_maxgen(n=20, R=30, pop_size=30,
                      mutation_prob=0.1, tournament_size=3):
    _title(
        f"ЕКСПЕРИМЕНТ 1 — Вплив MaxGen  "
        f"(n={n}, R={R}, pop_size={pop_size}, Pm={mutation_prob})"
    )

    maxgen_values = [10 * n, 20 * n, 50 * n, 100 * n]

    print(f"\n  {'MaxGen':>8} | {'Сер. F1':>11} | "
          f"{'Сер. F2':>11} | {'Сер. час, мс':>14}")
    print(THIN)

    results = []

    for mg in maxgen_values:
        f1_acc = f2_acc = t_acc = 0.0

        for i in range(R):
            _progress(i + 1, R)
            els, B, Wmin = generate_task(n)
            t0  = time.perf_counter()
            sol = genetic_algorithm(
                els, B, Wmin,
                pop_size=pop_size,
                max_gen=mg,
                mutation_prob=mutation_prob,
                tournament_size=tournament_size
            )
            t_acc  += (time.perf_counter() - t0) * 1000
            f1_acc += sol.fitness_1
            f2_acc += sol.fitness_2

        avg_f1 = f1_acc / R
        avg_f2 = f2_acc / R
        avg_time = t_acc / R

        print(f"\r  {mg:>8} | {avg_f1:>11.4f} | "
              f"{avg_f2:>11.6f} | {avg_time:>14.4f}")

        results.append(dict(
            max_gen=mg,
            avg_f1=avg_f1,
            avg_f2=avg_f2,
            avg_time=avg_time
        ))

    print(THIN)

    if results:
        _plot_maxgen_experiment(results, n, R)

# =============================================================
# ЕКСПЕРИМЕНТ 2: ВПЛИВ ПАРАМЕТРА Pm   
# =============================================================

# ГРАФІК: вплив Pm на якість і час роботи ГА

def _plot_pm_experiment(results, n, R, max_gen):
    """
    Будує два графіки для експерименту Pm:
    1) спільний графік F1 і F2 з двома осями Y;
    2) окремий графік середнього часу виконання.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    pm_values = [r['pm'] for r in results]
    avg_f1 = [r['avg_f1'] for r in results]
    avg_f2 = [r['avg_f2'] for r in results]
    avg_time = [r['avg_time'] for r in results]

    # ---------- Графік 1: Pm -> F1 і F2 ----------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    c_f1 = "#5A73A8"
    c_f2 = "#DF665C"

    line1 = ax1.plot(
        pm_values, avg_f1,
        marker='o',
        linestyle='-',
        color=c_f1,
        linewidth=2,
        markersize=7,
        label='Середнє F1'
    )

    ax1.set_xlabel("Ймовірність мутації Pm", fontsize=12, labelpad=10)
    ax1.set_ylabel("Середнє F1 (продуктивність)", fontsize=12, labelpad=10, color=c_f1)
    ax1.tick_params(axis='y', labelcolor=c_f1)
    ax1.set_xticks(pm_values)

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        pm_values, avg_f2,
        marker='s',
        linestyle='--',
        color=c_f2,
        linewidth=2,
        markersize=7,
        label='Середнє F2'
    )

    ax2.set_ylabel("Середнє F2 (надійність)", fontsize=12, labelpad=10, color=c_f2)
    ax2.tick_params(axis='y', labelcolor=c_f2)

    ax1.set_title(
        f"Вплив Pm на середні значення F1 та F2 (n={n}, MaxGen={max_gen}, R={R})",
        fontsize=14,
        pad=20
    )

    ax1.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.5)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    ax1.legend(
        lines,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()
    out = "pm_f1_f2.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\n  ✓ Графік F1/F2 для Pm збережено: {out}")
    plt.show()

    # ---------- Графік 2: Pm -> час ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_time = "#6A9955"

    ax.plot(
        pm_values, avg_time,
        marker='o',
        linestyle='-',
        color=c_time,
        linewidth=2,
        markersize=7
    )

    ax.set_title(
        f"Вплив Pm на середній час виконання (n={n}, MaxGen={max_gen}, R={R})",
        fontsize=14,
        pad=20
    )
    ax.set_xlabel("Ймовірність мутації Pm", fontsize=12, labelpad=10)
    ax.set_ylabel("Середній час виконання, мс", fontsize=12, labelpad=10)
    ax.set_xticks(pm_values)

    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, value in enumerate(avg_time):
        ax.annotate(
            f"{value:.2f}",
            (pm_values[i], avg_time[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            color=c_time
        )

    plt.tight_layout()
    out = "pm_time.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"  ✓ Графік часу для Pm збережено: {out}")
    plt.show()

def experiment_pm(n=20, R=30, pop_size=20, tournament_size=3):
    max_gen = 10 * n

    _title(
        f"ЕКСПЕРИМЕНТ 2 — Вплив Pm  "
        f"(n={n}, MaxGen={max_gen}, R={R}, pop_size={pop_size})"
    )

    pm_values = [0.01, 0.05, 0.1, 0.2, 0.3]

    print(f"\n  {'Pm':>6} | {'Сер. F1':>11} | "
          f"{'Сер. F2':>11} | {'Сер. час, мс':>14}")
    print(THIN)

    results = []

    for pm in pm_values:
        f1_acc = f2_acc = t_acc = 0.0

        for i in range(R):
            _progress(i + 1, R)
            els, B, Wmin = generate_task(n)
            t0  = time.perf_counter()
            sol = genetic_algorithm(
                els, B, Wmin,
                pop_size=pop_size,
                max_gen=max_gen,
                mutation_prob=pm,
                tournament_size=tournament_size
            )
            t_acc  += (time.perf_counter() - t0) * 1000
            f1_acc += sol.fitness_1
            f2_acc += sol.fitness_2

        avg_f1 = f1_acc / R
        avg_f2 = f2_acc / R
        avg_time = t_acc / R

        print(f"\r  {pm:>6.2f} | {avg_f1:>11.4f} | "
              f"{avg_f2:>11.6f} | {avg_time:>14.4f}")

        results.append(dict(
            pm=pm,
            avg_f1=avg_f1,
            avg_f2=avg_f2,
            avg_time=avg_time
        ))

    print(THIN)
    
    if results:
        _plot_pm_experiment(results, n, R, max_gen)

# =============================================================
# ГРАФІК: покращення ГА над ЖА у %
# =============================================================

def _plot_ga_improvement(results, R):
    """
    Будує лінійний графік: на скільки % ГА кращий за ЖА (за критеріями F1 та F2)
    для кожної аналізованої розмірності n. Використовує світлий академічний стиль.
    Зберігає файл ga_vs_gr_improvement.png.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    # Налаштування стилю (академічний, світлий фон, шрифт із зарубками)
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    ns      = [r['n']        for r in results]
    df1     = [r['delta_f1'] for r in results]
    df2     = [r['delta_f2'] for r in results]

    # Створення полотна
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Кольори ліній (схожі на надіслані приклади)
    c_f1 = "#5A73A8"  # Приглушений синій
    c_f2 = "#DF665C"  # Кораловий/Червоний

    # Побудова ліній з маркерами
    ax.plot(ns, df1, marker='o', linestyle='-', color=c_f1, linewidth=2, markersize=7, label='Покращення за F1 (продуктивність), %')
    ax.plot(ns, df2, marker='o', linestyle='-', color=c_f2, linewidth=2, markersize=7, label='Покращення за F2 (надійність), %')

    # Заголовки та підписи осей
    ax.set_title("Залежність покращення якості розв'язку ГА (відносно ЖА) від розмірності задачі", 
                 fontsize=14, pad=20)
    ax.set_xlabel("Розмірність задачі n", fontsize=12, labelpad=10)
    ax.set_ylabel("Покращення ГА над ЖА, %", fontsize=12, labelpad=10)

    # Налаштування осей (залишаємо тільки ліву та нижню лінію, як на фото)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    # Вісь X - встановлюємо точні значення n для позначок
    ax.set_xticks(ns)

    # Сітка (тонкі світло-сірі лінії, переважно горизонтальні)
    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.5)

    # Підписи точних значень відсотків біля точок
    for i, txt in enumerate(df1):
        # Логіка зміщення тексту, щоб написи не накладалися один на одного
        offset = 12 if df1[i] >= df2[i] else -16
        va = 'bottom' if df1[i] >= df2[i] else 'top'
        ax.annotate(f"{txt:.1f}%", (ns[i], df1[i]), textcoords="offset points", 
                    xytext=(0, offset), ha='center', va=va, fontsize=10, color=c_f1)

    for i, txt in enumerate(df2):
        offset = 12 if df2[i] > df1[i] else -16
        va = 'bottom' if df2[i] > df1[i] else 'top'
        ax.annotate(f"{txt:.1f}%", (ns[i], df2[i]), textcoords="offset points", 
                    xytext=(0, offset), ha='center', va=va, fontsize=10, color=c_f2)

    # Легенда знизу по центру, без рамки
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)

    plt.tight_layout()
    out = "ga_vs_gr_improvement.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\n  ✓ Графік збережено: {out}")
    plt.show()

# =============================================================
# ГРАФІК: порівняння ЖА та ГА за часом роботи
# =============================================================

def _plot_time_comparison(results, R):
    """
    Будує графік порівняння середнього часу роботи ЖА та ГА
    для різних розмірностей задачі.
    Зберігає файл time_comparison.png.
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12

    ns = [r['n'] for r in results]
    gr_time = [r['avg_gr_t'] for r in results]
    ga_time = [r['avg_ga_t'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    c_gr = "#5A73A8"
    c_ga = "#DF665C"

    ax.plot(
        ns, gr_time,
        marker='o',
        linestyle='-',
        color=c_gr,
        linewidth=2,
        markersize=7,
        label='Жадібний алгоритм'
    )

    ax.plot(
        ns, ga_time,
        marker='o',
        linestyle='-',
        color=c_ga,
        linewidth=2,
        markersize=7,
        label='Генетичний алгоритм'
    )

    ax.set_title(
        "Порівняння ЖА та ГА за часом роботи",
        fontsize=14,
        pad=20
    )
    ax.set_xlabel("Розмірність задачі n", fontsize=12, labelpad=10)
    ax.set_ylabel("Середній час виконання, мс", fontsize=12, labelpad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    ax.set_xticks(ns)

    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.4, alpha=0.5)

    for i, txt in enumerate(gr_time):
        ax.annotate(
            f"{txt:.2f}",
            (ns[i], gr_time[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            color=c_gr
        )

    for i, txt in enumerate(ga_time):
        ax.annotate(
            f"{txt:.2f}",
            (ns[i], ga_time[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10,
            color=c_ga
        )

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()
    out = "time_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"\n  ✓ Графік часу збережено: {out}")
    plt.show()


# =============================================================
# ЕКСПЕРИМЕНТ 3: ПОРІВНЯННЯ ЖА ТА ГА + ГРАФІК  (розд. 3.3.4)
# =============================================================

def experiment_comparison(R=20, pop_size=20,
                          mutation_prob=0.1, tournament_size=3):
    """
    Порівнює ЖА та ГА за F1, F2 та часом виконання на задачах різної
    розмірності n ∈ {10, 20, 50, 100, 200}.
    Після таблиці будує графік покращення ГА над ЖА у %.
    """
    _title(
        f"ЕКСПЕРИМЕНТ 3 — Порівняння ЖА та ГА  "
        f"(R={R} на кожну розмірність)"
    )

    n_values = [10, 20, 50, 100, 200]

    header = (
        f"  {'n':>5} | "
        f"{'ЖА F1':>10} | {'ЖА F2':>10} | {'ЖА мс':>9} | "
        f"{'ГА F1':>10} | {'ГА F2':>10} | {'ГА мс':>9} | "
        f"{'Δ F1, %':>8} | {'Δ F2, %':>8} | {'Перем. ГА':>10}"
    )
    print(f"\n{header}")
    print("=" * len(header))

    results = []   # для графіка

    for n in n_values:
        max_gen = 10 * n

        gr_f1 = gr_f2 = gr_t = 0.0
        ga_f1 = ga_f2 = ga_t = 0.0
        ga_wins = valid = 0

        print(f"\n  n={n}  ", end="", flush=True)

        for i in range(R):
            _progress(i + 1, R, prefix=f"n={n}  ")
            els, B, Wmin = generate_task(n)
            calculate_efficiency(els)

            # Жадібний
            t0 = time.perf_counter()
            gr = greedy_algorithm(els, B, Wmin)
            gr_elapsed = (time.perf_counter() - t0) * 1000

            # Генетичний
            t0 = time.perf_counter()
            ga = genetic_algorithm(
                els, B, Wmin,
                pop_size=pop_size,
                max_gen=max_gen,
                mutation_prob=mutation_prob,
                tournament_size=tournament_size
            )
            ga_elapsed = (time.perf_counter() - t0) * 1000

            if gr is None:
                continue

            valid   += 1
            gr_f1   += gr["f1"]
            gr_f2   += gr["f2"]
            gr_t    += gr_elapsed
            ga_f1   += ga.fitness_1
            ga_f2   += ga.fitness_2
            ga_t    += ga_elapsed

            gr_proxy = _SolutionProxy(gr["f1"], gr["f2"])
            if better(ga, gr_proxy):
                ga_wins += 1

        if valid == 0:
            print(f"\r  {n:>5} | (допустимих задач не знайдено)")
            continue

        r = valid
        avg_gr_f1 = gr_f1 / r
        avg_gr_f2 = gr_f2 / r
        avg_gr_t  = gr_t  / r
        avg_ga_f1 = ga_f1 / r
        avg_ga_f2 = ga_f2 / r
        avg_ga_t  = ga_t  / r

        delta_f1 = (avg_ga_f1 - avg_gr_f1) / max(abs(avg_gr_f1), 1e-9) * 100
        delta_f2 = (avg_ga_f2 - avg_gr_f2) / max(abs(avg_gr_f2), 1e-9) * 100
        delta_t  = (avg_ga_t  - avg_gr_t)  / max(abs(avg_gr_t),  1e-9) * 100

        print(
            f"\r  {n:>5} | "
            f"{avg_gr_f1:>10.2f} | {avg_gr_f2:>10.6f} | {avg_gr_t:>9.4f} | "
            f"{avg_ga_f1:>10.2f} | {avg_ga_f2:>10.6f} | {avg_ga_t:>9.4f} | "
            f"{delta_f1:>+8.2f} | {delta_f2:>+8.4f} | "
            f"  {ga_wins:>3}/{valid}"
        )

        results.append(dict(
            n=n,
            delta_f1=delta_f1,
            delta_f2=delta_f2,
            delta_t=delta_t,
            avg_gr_t=avg_gr_t,
            avg_ga_t=avg_ga_t,
            ga_wins=ga_wins,
            valid=valid
        ))

    print(f"\n{THIN}")
    print(
        "  Висновок: ГА, як правило, знаходить кращі розв'язки за ПСК,\n"
        "  але повільніший за ЖА. З ростом n різниця у якості зростає."
    )
    print(THIN)

    if results:
        _plot_ga_improvement(results, R)
        _plot_time_comparison(results, R)


# =============================================================
# ДОПОМІЖНІ ФУНКЦІЇ МЕНЮ
# =============================================================

def _input_default(prompt, current, cast=float):
    raw = input(f"  {prompt} (поточне: {current}, Enter = без змін): ").strip()
    if raw:
        try:
            return cast(raw)
        except ValueError:
            print("  ⚠ Некоректне значення. Залишено без змін.")
    return current


def _require_elements(elements):
    if not elements:
        print("\n  ⚠ Спочатку введіть або згенеруйте дані (пункти 1 або 2).")
        return False
    return True


def input_elements_manually():
    count    = int(input("\n  Кількість елементів: "))
    elements = []
    for i in range(count):
        print(f"\n  Елемент #{i + 1}")
        name         = input("    Назва              : ")
        cost         = float(input("    Вартість           : "))
        productivity = float(input("    Продуктивність     : "))
        reliability  = float(input("    Надійність (0..1)  : "))
        weight       = int(input("    Важливість (ціле)  : "))
        elements.append(
            Element(name, cost, productivity, reliability, weight)
        )
    return elements


def print_main_menu():
    print(f"\n{SEP}")
    print("  ГОЛОВНЕ МЕНЮ")
    print(SEP)
    print("  ── ДАНІ ───────────────────────────────────────────────")
    print("   1  Ввести дані вручну")
    print("   2  Згенерувати дані")
    print("   3  Завантажити з файлу  (.json)")
    print("   4  Зберегти у файл      (.json)")
    print("   5  Переглянути поточні дані")
    print()
    print("  ── АЛГОРИТМИ ──────────────────────────────────────────")
    print("   6  Жадібний алгоритм")
    print("   7  Генетичний алгоритм")
    print("   8  Обидва алгоритми + порівняння за ПСК")
    print()
    print("  ── НАЛАШТУВАННЯ ───────────────────────────────────────")
    print("   9  Змінити параметри задачі (B та Wmin)")
    print("  10  Налаштування генетичного алгоритму")
    print()
    print("  ── ЕКСПЕРИМЕНТИ ───────────────────────────────────────")
    print("  11  Дослідження впливу MaxGen")
    print("  12  Дослідження впливу Pm")
    print("  13  Порівняння алгоритмів за точністю та часом + ГРАФІКИ")
    print()
    print("   0  Вихід")
    print(SEP)


# =============================================================
# ГОЛОВНА ФУНКЦІЯ
# =============================================================

def main():
    elements   = []
    budget     = 1600.0
    min_weight = 20

    ga = {
        "pop_size":        50,
        "max_gen":        100,
        "pm":             0.1,
        "tournament_size":  3
    }

    print(f"\n{'*' * 72}")
    print("  Багатокритеріальна задача вибору складу ІТ-інфраструктури")
    print("  Жадібний алгоритм | Генетичний алгоритм")
    print(f"{'*' * 72}")

    while True:
        print_main_menu()
        choice = input("  Ваш вибір: ").strip()

        # ── 1 — РУЧНЕ ВВЕДЕННЯ ─────────────────────────────────
        if choice == "1":
            elements = input_elements_manually()
            calculate_efficiency(elements)
            budget     = _input_default("Бюджет B",              budget,     float)
            min_weight = _input_default("Мінімальна важливість", min_weight, int)
            print(f"\n  ✓ Введено {len(elements)} елементів. "
                  f"B={budget:.2f}, Wmin={min_weight}")

        # ── 2 — ГЕНЕРАЦІЯ ДАНИХ ────────────────────────────────
        elif choice == "2":
            n  = int(input("\n  Кількість елементів n: "))
            kB = _input_default("Коефіцієнт бюджету kB (0<kB<1)", 0.6, float)
            kW = _input_default("Коефіцієнт важливості kW (0<kW<1)", 0.5, float)
            elements, budget, min_weight = generate_task(n, kB=kB, kW=kW)
            calculate_efficiency(elements)
            print(f"\n  ✓ Згенеровано {n} елементів. "
                  f"B={budget:.2f}, Wmin={min_weight}")

        # ── 3 — ЗАВАНТАЖЕННЯ З ФАЙЛУ ───────────────────────────
        elif choice == "3":
            fname = input("\n  Ім'я файлу (напр. task.json): ").strip()
            try:
                elements, budget, min_weight = load_task(fname)
                calculate_efficiency(elements)
                print(f"\n  ✓ Завантажено {len(elements)} елементів. "
                      f"B={budget}, Wmin={min_weight}")
            except FileNotFoundError:
                print(f"\n  ⚠ Файл '{fname}' не знайдено.")
            except Exception as ex:
                print(f"\n  ⚠ Помилка читання: {ex}")

        # ── 4 — ЗБЕРЕЖЕННЯ У ФАЙЛ ──────────────────────────────
        elif choice == "4":
            if not _require_elements(elements):
                continue
            fname = input("\n  Ім'я файлу (напр. task.json): ").strip()
            try:
                save_task(fname, elements, budget, min_weight)
                print(f"\n  ✓ Задачу збережено у '{fname}'")
            except Exception as ex:
                print(f"\n  ⚠ Помилка запису: {ex}")

        # ── 5 — ПЕРЕГЛЯД ДАНИХ ─────────────────────────────────
        elif choice == "5":
            if not _require_elements(elements):
                continue
            print_elements(elements, budget, min_weight)

        # ── 6 — ЖАДІБНИЙ АЛГОРИТМ ─────────────────────────────
        elif choice == "6":
            if not _require_elements(elements):
                continue
            _title("ЖАДІБНИЙ АЛГОРИТМ")
            t0  = time.perf_counter()
            res = greedy_algorithm(elements, budget, min_weight)
            t_el = time.perf_counter() - t0
            if res:
                print_solution(
                    "Результат  (x')",
                    res["selected"], res["total_cost"], res["total_weight"],
                    res["f1"], res["f2"], t_el
                )
            else:
                print("\n  Допустимого розв'язку не знайдено.")

        # ── 7 — ГЕНЕТИЧНИЙ АЛГОРИТМ ───────────────────────────
        elif choice == "7":
            if not _require_elements(elements):
                continue
            _title("ГЕНЕТИЧНИЙ АЛГОРИТМ")
            print(
                f"  pop_size={ga['pop_size']}, max_gen={ga['max_gen']}, "
                f"Pm={ga['pm']}, k={ga['tournament_size']}"
            )
            random.seed(14)
            t0  = time.perf_counter()
            sol = genetic_algorithm(
                elements, budget, min_weight,
                pop_size=ga["pop_size"],
                max_gen=ga["max_gen"],
                mutation_prob=ga["pm"],
                tournament_size=ga["tournament_size"],
                show_progress=True
            )
            t_el = time.perf_counter() - t0
            selected = [elements[i] for i, g in enumerate(sol.genes) if g == 1]
            print_solution(
                "Результат  (x')",
                selected, sol.total_cost, sol.total_weight,
                sol.fitness_1, sol.fitness_2, t_el
            )

        # ── 8 — ОБА АЛГОРИТМИ + ПОРІВНЯННЯ ───────────────────
        elif choice == "8":
            if not _require_elements(elements):
                continue

            t0  = time.perf_counter()
            gr  = greedy_algorithm(elements, budget, min_weight)
            t_gr = time.perf_counter() - t0

            t0  = time.perf_counter()
            ga_sol = genetic_algorithm(
                elements, budget, min_weight,
                pop_size=ga["pop_size"],
                max_gen=ga["max_gen"],
                mutation_prob=ga["pm"],
                tournament_size=ga["tournament_size"],
                show_progress=True
            )
            t_ga = time.perf_counter() - t0

            print_comparison(gr, ga_sol, elements, t_gr, t_ga)

        # ── 9 — ЗМІНИТИ B ТА Wmin ─────────────────────────────
        elif choice == "9":
            budget     = _input_default("Бюджет B",              budget,     float)
            min_weight = _input_default("Мінімальна важливість", min_weight, int)
            print(f"\n  ✓ B={budget:.2f}, Wmin={min_weight}")

        # ── 10 — НАЛАШТУВАННЯ ГА ──────────────────────────────
        elif choice == "10":
            _title("ПАРАМЕТРИ ГЕНЕТИЧНОГО АЛГОРИТМУ")
            ga["pop_size"]        = _input_default("Розмір популяції pop_size",   ga["pop_size"],        int)
            ga["max_gen"]         = _input_default("Кількість поколінь max_gen",  ga["max_gen"],         int)
            ga["pm"]              = _input_default("Ймовірність мутації Pm",      ga["pm"],              float)
            ga["tournament_size"] = _input_default("Розмір турніру k",            ga["tournament_size"], int)
            print(
                f"\n  ✓ pop_size={ga['pop_size']}, max_gen={ga['max_gen']}, "
                f"Pm={ga['pm']}, k={ga['tournament_size']}"
            )

        # ── 11 — ЕКСПЕРИМЕНТ 1: MaxGen ────────────────────────
        elif choice == "11":
            _title("НАЛАШТУВАННЯ ЕКСПЕРИМЕНТУ 1")
            n_ex = _input_default("Розмірність задачі n", 20, int)
            R_ex = _input_default("Кількість повторень R", 30, int)
            experiment_maxgen(
                n=n_ex, R=R_ex,
                pop_size=ga["pop_size"],
                mutation_prob=ga["pm"],
                tournament_size=ga["tournament_size"]
            )

        # ── 12 — ЕКСПЕРИМЕНТ 2: Pm ────────────────────────────
        elif choice == "12":
            _title("НАЛАШТУВАННЯ ЕКСПЕРИМЕНТУ 2")
            n_ex = _input_default("Розмірність задачі n", 20, int)
            R_ex = _input_default("Кількість повторень R", 30, int)
            experiment_pm(
                n=n_ex, R=R_ex,
                pop_size=ga["pop_size"],
                tournament_size=ga["tournament_size"]
            )

        # ── 13 — ЕКСПЕРИМЕНТ 3: Порівняння + ГРАФІК ──────────
        elif choice == "13":
            _title("НАЛАШТУВАННЯ ЕКСПЕРИМЕНТУ 3")
            R_ex = _input_default("Кількість повторень R на кожну n", 20, int)
            experiment_comparison(
                R=R_ex,
                pop_size=ga["pop_size"],
                mutation_prob=ga["pm"],
                tournament_size=ga["tournament_size"]
            )

        # ── 0 — ВИХІД ─────────────────────────────────────────
        elif choice == "0":
            print("\n  Завершення роботи. До побачення!\n")
            break

        else:
            print("\n  ⚠ Невірний вибір. Спробуйте ще раз.")


# =============================================================
if __name__ == "__main__":
    main()