"""
═══════════════════════════════════════════════════════════════════════════════
РГР I — Описание выборки. Оценивание параметров. Доверительные интервалы
Вариант E-10 | Признаки: X1, X2, X3, X4 | n = 200
═══════════════════════════════════════════════════════════════════════════════

Структура скрипта:
  ЭТАП 1 — Первичное описание выборки (вариационный ряд, ECDF, гистограммы,
            числовые характеристики)
  ЭТАП 2 — Предположение о виде закона распределения
  ЭТАП 3 — Оценивание параметров (метод моментов и ММП)
  ЭТАП 4 — Оценка вероятности события P(X > x0)
  ЭТАП 5 — Работа с группированными данными
  ЭТАП 6 — Доверительные интервалы (асимптотические + точные для нормальных)
  ЭТАП 7 — Итоговые выводы (печать в консоль)

Используемые библиотеки:
  numpy         — векторные вычисления, моменты, сортировка
  scipy.stats   — теоретические распределения, квантили t, χ²
  matplotlib    — все графики
  pandas        — чтение CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Настройки графиков ───────────────────────────────────────────────────────
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11,
                     'axes.titleweight': 'bold', 'figure.dpi': 150})
COLORS = ['steelblue', 'mediumseagreen', 'mediumpurple', 'tomato']

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================
df = pd.read_csv("RGR1_E-10_X1-X4 (4).csv")
X1 = df['X1'].values.astype(float)
X2 = df['X2'].values.astype(float)
X3 = df['X3'].values.astype(float)
X4 = df['X4'].values.astype(float)
n = len(X1)   # 200

print("=" * 65)
print(f"Данные загружены: {n} наблюдений, 4 признака (X1–X4)")
print("=" * 65)

# =============================================================================
# ЭТАП 1. ПЕРВИЧНОЕ ОПИСАНИЕ ВЫБОРКИ
# =============================================================================

def describe_column(name, data):
    """Полные числовые характеристики одного признака."""
    xbar = data.mean()
    s2   = np.var(data, ddof=1)             # несмещённая дисперсия
    s    = np.sqrt(s2)
    med  = np.median(data)
    q1   = np.percentile(data, 25)
    q3   = np.percentile(data, 75)
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)             # эксцесс (excess kurtosis)
    return {
        'name': name, 'data': data, 'n': len(data),
        'mean': xbar, 'var': s2, 'std': s,
        'median': med, 'q1': q1, 'q3': q3,
        'min': data.min(), 'max': data.max(),
        'skew': skew, 'kurt': kurt
    }

stats_all = {c: describe_column(c, d)
             for c, d in zip(['X1','X2','X3','X4'], [X1, X2, X3, X4])}

print("\n── ЭТАП 1: Числовые характеристики ──────────────────────────")
header = f"{'Показатель':<22} {'X1':>10} {'X2':>10} {'X3':>10} {'X4':>10}"
print(header)
print("-" * 65)
rows = [
    ('n', 'n'),
    ('Среднее', 'mean'), ('Медиана', 'median'),
    ('Дисперсия s²', 'var'), ('Ст. откл. s', 'std'),
    ('Q1 (25%)', 'q1'), ('Q3 (75%)', 'q3'),
    ('Минимум', 'min'), ('Максимум', 'max'),
    ('Асимметрия', 'skew'), ('Эксцесс', 'kurt'),
]
for label, key in rows:
    vals = [stats_all[c][key] for c in ['X1','X2','X3','X4']]
    fmt  = '{:>10.4f}' if key != 'n' else '{:>10d}'
    row  = f"{label:<22}" + "".join(fmt.format(v) for v in vals)
    print(row)

# ─── Рисунок 1: Гистограммы ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('РГР I · Вариант E-10  |  Гистограммы и подогнанные распределения',
             fontsize=12, fontweight='bold')

for idx, col in enumerate(['X1','X2','X3','X4']):
    r   = stats_all[col]
    ax  = axes[idx // 2, idx % 2]
    ax.hist(r['data'], bins=20, density=True, color=COLORS[idx],
            alpha=0.6, edgecolor='white', label='Эмпирическая')
    ax.axvline(r['mean'],   color='darkorange', lw=2, ls='--',
               label=f"x̄ = {r['mean']:.4f}")
    ax.axvline(r['median'], color='green',      lw=2, ls=':',
               label=f"Me = {r['median']:.4f}")
    ax.set_xlabel(col)
    ax.set_ylabel('Плотность')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

titles = ['X1 — Экспоненциальное со сдвигом',
          'X2 — Равномерное',
          'X3 — Нормальное',
          'X4 — Нормальное']
for ax, t in zip(axes.flat, titles):
    ax.set_title(t)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig1_histograms.png', bbox_inches='tight')
plt.close()
print("\nРис. 1 сохранён: fig1_histograms.png")

# ─── Рисунок 2: Эмпирические функции распределения ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Эмпирические функции распределения (вариационный ряд)', fontsize=12, fontweight='bold')

for idx, col in enumerate(['X1','X2','X3','X4']):
    ax   = axes[idx // 2, idx % 2]
    data = stats_all[col]['data']
    sorted_data = np.sort(data)
    ecdf        = np.arange(1, n + 1) / n
    ax.step(sorted_data, ecdf, color=COLORS[idx], lw=2, label='Fn(x)')
    ax.set_xlabel(col); ax.set_ylabel('Fn(x)')
    ax.set_title(col, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig2_ecdf.png', bbox_inches='tight')
plt.close()
print("Рис. 2 сохранён: fig2_ecdf.png")

# =============================================================================
# ЭТАП 3. ОЦЕНИВАНИЕ ПАРАМЕТРОВ (ММ + ММП)
# =============================================================================

print("\n── ЭТАП 3: Оценивание параметров ────────────────────────────")

# ──── X1: Exp(λ, c)  ─────────────────────────────────────────────────────────
# Метод моментов:
#   E[X] = c + 1/λ  =>  λ = 1/(x̄ - c),  c = min(x) (аппроксимация нижней границы)
#   D[X] = 1/λ²    =>  λ = 1/s
# ММП: c_mle = x_min,  λ_mle = 1/(x̄ - x_min)

xbar1 = X1.mean(); s1 = X1.std(ddof=1)
c_mm  = X1.min()
lam_mm = 1.0 / (xbar1 - c_mm)            # из E[X] = c + 1/λ

c_mle   = X1.min()
lam_mle = 1.0 / (xbar1 - c_mle)         # MM и MLE совпадают при c=x_min

print(f"\nX1 ~ Exp(λ, c)  (сдвинутое экспоненциальное):")
print(f"  Метод моментов: c = {c_mm:.4f},  λ = {lam_mm:.4f}")
print(f"  ММП:            c = {c_mle:.4f}, λ = {lam_mle:.4f}")
print(f"  Проверка MM: E[X] = c + 1/λ = {c_mm + 1/lam_mm:.4f}  (x̄ = {xbar1:.4f})")

# ──── X2: U(a, b)  ───────────────────────────────────────────────────────────
# Метод моментов:  E[X] = (a+b)/2,  D[X] = (b-a)²/12
#   => b-a = sqrt(12)*s,  a = x̄ - sqrt(3)*s,  b = x̄ + sqrt(3)*s
# ММП: a_mle = x_min,  b_mle = x_max

xbar2 = X2.mean(); s2std = X2.std(ddof=1)
a_mm = xbar2 - np.sqrt(3) * s2std
b_mm = xbar2 + np.sqrt(3) * s2std

a_mle2 = X2.min()
b_mle2 = X2.max()

print(f"\nX2 ~ U(a, b)  (равномерное):")
print(f"  Метод моментов: a = {a_mm:.4f},  b = {b_mm:.4f}")
print(f"  ММП:            a = {a_mle2:.4f}, b = {b_mle2:.4f}")
print(f"  Проверка MM: E[X] = (a+b)/2 = {(a_mm+b_mm)/2:.4f}  (x̄ = {xbar2:.4f})")

# ──── X3: N(μ, σ²)  ──────────────────────────────────────────────────────────
# Метод моментов: μ̂ = x̄,  σ̂ = s (ddof=1)
# ММП:            μ̂ = x̄,  σ̂ = s (ddof=0)

mu3_mm  = X3.mean()
sig3_mm = X3.std(ddof=1)
mu3_mle = X3.mean()
sig3_mle = np.std(X3, ddof=0)

print(f"\nX3 ~ N(μ, σ²)  (нормальное):")
print(f"  Метод моментов: μ = {mu3_mm:.4f},  σ = {sig3_mm:.4f},  σ² = {sig3_mm**2:.6f}")
print(f"  ММП:            μ = {mu3_mle:.4f}, σ = {sig3_mle:.4f}, σ² = {sig3_mle**2:.6f}")

# ──── X4: N(μ, σ²)  ──────────────────────────────────────────────────────────
mu4_mm   = X4.mean()
sig4_mm  = X4.std(ddof=1)
mu4_mle  = X4.mean()
sig4_mle = np.std(X4, ddof=0)

print(f"\nX4 ~ N(μ, σ²)  (нормальное):")
print(f"  Метод моментов: μ = {mu4_mm:.4f},  σ = {sig4_mm:.4f},  σ² = {sig4_mm**2:.6f}")
print(f"  ММП:            μ = {mu4_mle:.4f}, σ = {sig4_mle:.4f}, σ² = {sig4_mle**2:.6f}")

# ─── Рисунок 3: Гистограммы с подогнанными распределениями ───────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Подогнанные теоретические распределения', fontsize=12, fontweight='bold')

# X1
ax = axes[0, 0]
ax.hist(X1, bins=25, density=True, color=COLORS[0], alpha=0.55, edgecolor='white')
xp = np.linspace(c_mle, X1.max() + 0.5, 500)
ax.plot(xp, lam_mle * np.exp(-lam_mle * (xp - c_mle)), 'r-', lw=2.5,
        label=f'Exp(λ={lam_mle:.3f}, c={c_mle:.3f})\nМетод MM и ММП совпадают')
ax.set_title('X1 ~ Exp(λ, c)'); ax.set_xlabel('X1'); ax.set_ylabel('f(x)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# X2
ax = axes[0, 1]
ax.hist(X2, bins=15, density=True, color=COLORS[1], alpha=0.55, edgecolor='white')
xp = np.linspace(0, 1.5, 500)
# MM
fp_mm = np.where((xp >= a_mm) & (xp <= b_mm), 1/(b_mm - a_mm), 0)
ax.plot(xp, fp_mm, 'b--', lw=2, label=f'U MM: [{a_mm:.3f}, {b_mm:.3f}]')
# MLE
fp_mle = np.where((xp >= a_mle2) & (xp <= b_mle2), 1/(b_mle2 - a_mle2), 0)
ax.plot(xp, fp_mle, 'r-', lw=2.5, label=f'U MLE: [{a_mle2:.3f}, {b_mle2:.3f}]')
ax.set_title('X2 ~ U(a, b)'); ax.set_xlabel('X2'); ax.set_ylabel('f(x)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# X3
ax = axes[1, 0]
ax.hist(X3, bins=18, density=True, color=COLORS[2], alpha=0.55, edgecolor='white')
xp = np.linspace(-0.15, 0.3, 500)
ax.plot(xp, stats.norm.pdf(xp, mu3_mm, sig3_mm), 'b--', lw=2,
        label=f'N MM: μ={mu3_mm:.4f}, σ={sig3_mm:.4f}')
ax.plot(xp, stats.norm.pdf(xp, mu3_mle, sig3_mle), 'r-', lw=2.5,
        label=f'N MLE: μ={mu3_mle:.4f}, σ={sig3_mle:.4f}')
ax.set_title('X3 ~ N(μ, σ²)'); ax.set_xlabel('X3'); ax.set_ylabel('f(x)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# X4
ax = axes[1, 1]
ax.hist(X4, bins=18, density=True, color=COLORS[3], alpha=0.55, edgecolor='white')
xp = np.linspace(-1.2, 2.5, 500)
ax.plot(xp, stats.norm.pdf(xp, mu4_mm, sig4_mm), 'b--', lw=2,
        label=f'N MM: μ={mu4_mm:.4f}, σ={sig4_mm:.4f}')
ax.plot(xp, stats.norm.pdf(xp, mu4_mle, sig4_mle), 'r-', lw=2.5,
        label=f'N MLE: μ={mu4_mle:.4f}, σ={sig4_mle:.4f}')
ax.set_title('X4 ~ N(μ, σ²)'); ax.set_xlabel('X4'); ax.set_ylabel('f(x)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig3_fitted.png', bbox_inches='tight')
plt.close()
print("Рис. 3 сохранён: fig3_fitted.png")

# =============================================================================
# ЭТАП 4. ОЦЕНКА ВЕРОЯТНОСТИ P(X > x0)
# =============================================================================

print("\n── ЭТАП 4: P(X > x0) ────────────────────────────────────────")

# X1: x0 = среднее (естественный порог)
x0_1   = X1.mean()
p_emp1 = np.mean(X1 > x0_1)
p_mod1 = np.exp(-lam_mle * (x0_1 - c_mle))     # 1 - F(x0) = e^{-λ(x0-c)}
print(f"\nX1: x0 = x̄ = {x0_1:.4f}")
print(f"  Эмпирическая: P_emp = {p_emp1:.4f}  ({int(p_emp1*n)}/{n})")
print(f"  Параметрическая: P_mod = e^(-{lam_mle:.4f}*({x0_1:.4f}-{c_mle:.4f})) = {p_mod1:.4f}")

# X2: x0 = середина [a, b]
x0_2   = (a_mle2 + b_mle2) / 2
p_emp2 = np.mean(X2 > x0_2)
p_mod2 = (b_mle2 - x0_2) / (b_mle2 - a_mle2)
print(f"\nX2: x0 = (a+b)/2 = {x0_2:.4f}")
print(f"  Эмпирическая: P_emp = {p_emp2:.4f}")
print(f"  Параметрическая: P_mod = (b-x0)/(b-a) = {p_mod2:.4f}")

# X3: x0 = μ + σ  (68-95-99.7 правило)
x0_3   = mu3_mle + sig3_mle
p_emp3 = np.mean(X3 > x0_3)
p_mod3 = 1 - stats.norm.cdf(x0_3, mu3_mle, sig3_mle)  # ≈ 0.1587
print(f"\nX3: x0 = μ + σ = {x0_3:.4f}")
print(f"  Эмпирическая: P_emp = {p_emp3:.4f}")
print(f"  Параметрическая: P_mod = 1-Φ(1) ≈ {p_mod3:.4f}")

# X4: x0 = μ
x0_4   = mu4_mle
p_emp4 = np.mean(X4 > x0_4)
p_mod4 = 1 - stats.norm.cdf(x0_4, mu4_mle, sig4_mle)   # = 0.5 по симметрии
print(f"\nX4: x0 = μ = {x0_4:.4f}")
print(f"  Эмпирическая: P_emp = {p_emp4:.4f}")
print(f"  Параметрическая: P_mod = {p_mod4:.4f}")

# =============================================================================
# ЭТАП 5. ГРУППИРОВАННЫЕ ДАННЫЕ
# =============================================================================

print("\n── ЭТАП 5: Группированные данные ────────────────────────────")

def group_and_compute(data, k=9):
    """
    Разбить данные на k равных интервалов, вычислить выборочное среднее
    и дисперсию по формулам для группированных данных.
    """
    edges  = np.linspace(data.min(), data.max(), k + 1)
    mids   = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(data, bins=edges)
    nn = len(data)
    mean_gr = np.sum(counts * mids) / nn
    # Несмещённая дисперсия через формулу для группированных данных
    var_gr  = np.sum(counts * (mids - mean_gr)**2) / (nn - 1)
    std_gr  = np.sqrt(var_gr)
    return mean_gr, std_gr, edges, mids, counts

print(f"\n{'Признак':<8} {'Точн. μ':>10} {'Групп. μ':>10} {'Δμ':>8} "
      f"{'Точн. σ':>10} {'Групп. σ':>10} {'Δσ':>8}")
print("-" * 68)
for col, data in [('X1',X1),('X2',X2),('X3',X3),('X4',X4)]:
    mg, sg, _, _, _ = group_and_compute(data)
    exact_m = data.mean(); exact_s = data.std(ddof=1)
    print(f"{col:<8} {exact_m:>10.4f} {mg:>10.4f} {abs(mg-exact_m):>8.4f} "
          f"{exact_s:>10.4f} {sg:>10.4f} {abs(sg-exact_s):>8.4f}")

# Подробная таблица для X1
print("\nПодробная группировка X1 (k=9 интервалов):")
mg1, sg1, e1, m1, c1 = group_and_compute(X1)
print(f"  {'Интервал':<20} {'Середина':>10} {'Частота':>8} {'Относит.':>10}")
for i in range(len(m1)):
    print(f"  [{e1[i]:.3f}, {e1[i+1]:.3f})  {m1[i]:>10.3f} {c1[i]:>8d} {c1[i]/n:>10.4f}")
print(f"  Итого: {sum(c1):>28d}")
print(f"  Группированное среднее = {mg1:.4f},  σ_гр = {sg1:.4f}")

# ─── Рисунок 4: Группированные данные ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Группированные данные (k = 9 интервалов)', fontsize=12, fontweight='bold')

for idx, (col, data) in enumerate(zip(['X1','X2','X3','X4'],[X1,X2,X3,X4])):
    ax = axes[idx // 2, idx % 2]
    mg, sg, edges, mids, counts = group_and_compute(data)
    widths = edges[1:] - edges[:-1]
    ax.bar(mids, counts / n / widths, width=widths * 0.92,
           color=COLORS[idx], alpha=0.7, edgecolor='white',
           label=f'Сгруп. μ={mg:.4f}, σ={sg:.4f}')
    ax.axvline(data.mean(), color='black', lw=2,   ls='--',
               label=f'Точное x̄ = {data.mean():.4f}')
    ax.axvline(mg,          color='red',   lw=1.8, ls=':',
               label=f'Сгруп.  μ̂ = {mg:.4f}')
    ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel('Плотность')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fig4_grouped.png', bbox_inches='tight')
plt.close()
print("\nРис. 4 сохранён: fig4_grouped.png")

# =============================================================================
# ЭТАП 6. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ
# =============================================================================

print("\n── ЭТАП 6: Доверительные интервалы (α = 0.05, 95%) ─────────")

alpha = 0.05
z95   = stats.norm.ppf(1 - alpha / 2)        # ≈ 1.96
t_crit = stats.t.ppf(1 - alpha / 2, df=n-1)  # t_{0.975, 199}
chi2_lo = stats.chi2.ppf(alpha / 2,     df=n-1)
chi2_hi = stats.chi2.ppf(1 - alpha / 2, df=n-1)

print(f"\nКвантили: z_{{0.975}} = {z95:.4f},  t_{{0.975,199}} = {t_crit:.4f}")
print(f"          χ²_{{0.025,199}} = {chi2_lo:.4f},  χ²_{{0.975,199}} = {chi2_hi:.4f}")

def asymptotic_ci(data):
    """Асимптотический ДИ для МО: x̄ ± z * s / sqrt(n)."""
    xbar = data.mean(); s = data.std(ddof=1)
    se   = s / np.sqrt(len(data))
    return xbar - z95 * se, xbar + z95 * se

def exact_ci_mean(data):
    """Точный ДИ для МО нормального распределения (распределение Стьюдента)."""
    xbar = data.mean(); s = data.std(ddof=1); nn = len(data)
    se   = s / np.sqrt(nn)
    return xbar - t_crit * se, xbar + t_crit * se

def exact_ci_var(data):
    """Точный ДИ для дисперсии нормального распределения (χ²-распределение)."""
    s2 = np.var(data, ddof=1); nn = len(data)
    lo = (nn - 1) * s2 / chi2_hi
    hi = (nn - 1) * s2 / chi2_lo
    return lo, hi

print(f"\nАсимптотические 95% ДИ для μ:")
print(f"  {'Признак':<6} {'μ̂':>10} {'ДИ нижняя':>12} {'ДИ верхняя':>12} {'Ширина':>10}")
print("  " + "-" * 52)
for col, data in [('X1',X1),('X2',X2),('X3',X3),('X4',X4)]:
    lo, hi = asymptotic_ci(data)
    print(f"  {col:<6} {data.mean():>10.4f} {lo:>12.4f} {hi:>12.4f} {hi-lo:>10.4f}")

print(f"\nТочные 95% ДИ для μ (Стьюдент, только X3 и X4 — нормальные):")
for col, data in [('X3',X3),('X4',X4)]:
    lo, hi = exact_ci_mean(data)
    lo_a, hi_a = asymptotic_ci(data)
    print(f"  {col}: Стьюдент  ({lo:.4f}; {hi:.4f}),  ширина = {hi-lo:.4f}")
    print(f"  {col}: Асимптот. ({lo_a:.4f}; {hi_a:.4f}),  ширина = {hi_a-lo_a:.4f}")

print(f"\nТочные 95% ДИ для σ² (χ², только X3 и X4 — нормальные):")
for col, data in [('X3',X3),('X4',X4)]:
    lo, hi = exact_ci_var(data)
    s2 = np.var(data, ddof=1)
    print(f"  {col}: σ² = {s2:.6f}")
    print(f"  {col}: ДИ для σ²: ({lo:.6f}; {hi:.6f})")
    print(f"  {col}: ДИ для σ:  ({np.sqrt(lo):.4f}; {np.sqrt(hi):.4f})")

# ─── Рисунок 5: Визуализация доверительных интервалов ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
names_ci = ['X1\nExp', 'X2\nU(a,b)', 'X3\nN(μ,σ²)', 'X4\nN(μ,σ²)']
all_data  = [X1, X2, X3, X4]
means_ci  = [d.mean() for d in all_data]
ci_lo_all = [asymptotic_ci(d)[0] for d in all_data]
ci_hi_all = [asymptotic_ci(d)[1] for d in all_data]
errs = [[m - l for m,l in zip(means_ci, ci_lo_all)],
        [h - m for h,m in zip(ci_hi_all, means_ci)]]
ax.errorbar(range(4), means_ci, yerr=errs, fmt='o', color='steelblue',
            capsize=10, capthick=2.5, ms=9, lw=2,
            label='Точечная оценка μ̂ ± 95% ДИ')
for i, (lo, hi, m) in enumerate(zip(ci_lo_all, ci_hi_all, means_ci)):
    ax.text(i + 0.12, m, f'[{lo:.3f};\n {hi:.3f}]', fontsize=8, va='center')
ax.set_xticks(range(4)); ax.set_xticklabels(names_ci, fontsize=10)
ax.set_ylabel('μ̂', fontsize=12)
ax.set_title('Асимптотические 95% доверительные интервалы для математического ожидания',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_ci.png', bbox_inches='tight')
plt.close()
print("\nРис. 5 сохранён: fig5_ci.png")

# =============================================================================
# ЭТАП 7. ИТОГОВЫЕ ВЫВОДЫ
# =============================================================================

print("\n" + "=" * 65)
print("ЭТАП 7: ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 65)
print(f"""
Вариант E-10. Проанализированы 4 признака (n=200 наблюдений каждый).

X1 — Признак с правосторонней асимметрией (skew={stats_all['X1']['skew']:.3f}).
  Среднее ({X1.mean():.4f}) существенно превышает медиану ({np.median(X1):.4f}).
  Модель: сдвинутое экспоненциальное Exp(λ, c).
  Параметры ММП: c = {c_mle:.4f}, λ = {lam_mle:.4f}.
  95% ДИ для μ: ({asymptotic_ci(X1)[0]:.4f}; {asymptotic_ci(X1)[1]:.4f}).

X2 — Практически нулевая асимметрия (skew={stats_all['X2']['skew']:.4f}).
  Гистограмма прямоугольной формы. Модель: равномерное U(a, b).
  Параметры ММП: a = {a_mle2:.4f}, b = {b_mle2:.4f}.
  95% ДИ для μ: ({asymptotic_ci(X2)[0]:.4f}; {asymptotic_ci(X2)[1]:.4f}).

X3 — Симметричное распределение (skew={stats_all['X3']['skew']:.4f}),
  колоколообразная форма гистограммы. Модель: нормальное N(μ, σ²).
  Параметры: μ̂ = {mu3_mle:.4f}, σ̂ = {sig3_mle:.4f}.
  Точный ДИ для μ (t): ({exact_ci_mean(X3)[0]:.4f}; {exact_ci_mean(X3)[1]:.4f}).
  Точный ДИ для σ²(χ²): ({exact_ci_var(X3)[0]:.6f}; {exact_ci_var(X3)[1]:.6f}).

X4 — Симметричное двухмодальное распределение (skew={stats_all['X4']['skew']:.4f}).
  Значения концентрируются вблизи -0.2 и +1.6, что соответствует
  смеси двух нормальных законов. В первом приближении аппроксимируется
  N(μ, σ²) с широкой дисперсией.
  Параметры: μ̂ = {mu4_mle:.4f}, σ̂ = {sig4_mle:.4f}.
  Точный ДИ для μ (t): ({exact_ci_mean(X4)[0]:.4f}; {exact_ci_mean(X4)[1]:.4f}).
  Точный ДИ для σ²(χ²): ({exact_ci_var(X4)[0]:.4f}; {exact_ci_var(X4)[1]:.4f}).

Группировка (k=9 интервалов): смещение оценок среднего не превышает
0.06 ед., дисперсии — 0.008 ед., что говорит о достаточной точности
оценок по сгруппированным данным при n=200.
""")

print("Все графики сохранены. Анализ завершён.")
