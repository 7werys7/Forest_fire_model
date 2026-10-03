import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap


EMPTY = 0
TREE = 1
BURNING = 2


ROWS, COLS = 60, 60
TREE_PROB = 0.7
P_GROW = 0.01
F_LIGHTNING = 0.0001
K_THRESHOLD = 1
NEIGHBORHOOD = 4
STEPS = 500
SEED = 42


def init_forest(rows, cols, tree_prob, seed=None):
    rng = np.random.default_rng(seed)
    grid = rng.choice(
        [EMPTY, TREE],
        size=(rows, cols),
        p=[1 - tree_prob, tree_prob]
    )
    return grid.astype(np.uint8)

def count_burning_neighbors(grid, i, j, neighborhood):
    rows, cols = grid.shape
    count = 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            if neighborhood == 4 and abs(di) + abs(dj) != 1:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                if grid[ni, nj] == BURNING:
                    count += 1
    return count

def step(grid, p, f, k, neighborhood):
    rows, cols = grid.shape
    new_grid = grid.copy()
    for i in range(rows):
        for j in range(cols):
            state = grid[i, j]
            if state == BURNING:
                new_grid[i, j] = EMPTY
            elif state == TREE:
                bn = count_burning_neighbors(grid, i, j, neighborhood)
                if bn >= k or np.random.random() < f:
                    new_grid[i, j] = BURNING
            elif state == EMPTY:
                bn = count_burning_neighbors(grid, i, j, neighborhood)
                if bn == 0 and np.random.random() < p:
                    new_grid[i, j] = TREE
    return new_grid

def count_states(g):
    return {
        'empty': int(np.sum(g == EMPTY)),
        'tree': int(np.sum(g == TREE)),
        'burning': int(np.sum(g == BURNING)),
    }

def format_stats_text(step, cur, grown_now, burned_now, total_grown, total_burned, total_cells):
    return (
        f"Шаг: {step}\n"
        f"Пусто:   {cur['empty']:5d}  ({cur['empty']/total_cells*100:5.1f}%)\n"
        f"Деревья: {cur['tree']:5d}  ({cur['tree']/total_cells*100:5.1f}%)\n"
        f"Горит:   {cur['burning']:5d}  ({cur['burning']/total_cells*100:5.1f}%)\n"
        f"---\n"
        f"Выросло за шаг:  {grown_now}\n"
        f"Сгорело за шаг:  {burned_now}\n"
        f"---\n"
        f"Всего выросло:   {total_grown}\n"
        f"Всего сгорело:   {total_burned}"
    )


grid = init_forest(ROWS, COLS, TREE_PROB, SEED)
total_cells = ROWS * COLS


stats = {
    'total_trees_grown': 0,
    'total_trees_burned': 0,
    'history': {'step': [], 'empty': [], 'tree': [], 'burning': []}
}


colors = ['#1a1a1a', '#2e8b57', '#ff4500']
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
title = ax.set_title("Лесной пожар, шаг 0")
ax.axis('off')

cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
cbar.ax.set_yticklabels(['Пусто', 'Дерево', 'Огонь'])

stats_text = ax.text(
    0.02, 0.98, "", transform=ax.transAxes,
    fontsize=10, verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)


initial_cur = count_states(grid)
stats['history']['step'].append(0)
stats['history']['empty'].append(initial_cur['empty'])
stats['history']['tree'].append(initial_cur['tree'])
stats['history']['burning'].append(initial_cur['burning'])

stats_text.set_text(
    format_stats_text(
        step=0,
        cur=initial_cur,
        grown_now=0,
        burned_now=0,
        total_grown=0,
        total_burned=0,
        total_cells=total_cells
    )
)


fig.savefig('forest_fire_initial.png', dpi=150, bbox_inches='tight')
print("Сохранён начальный шаг: forest_fire_initial.png")

prev_grid = grid.copy()


def update(frame):
    global grid, prev_grid

    grid = step(grid, P_GROW, F_LIGHTNING, K_THRESHOLD, NEIGHBORHOOD)

    grown_now = int(np.sum((prev_grid == EMPTY) & (grid == TREE)))
    burned_now = int(np.sum((prev_grid == TREE) & (grid == BURNING)))

    stats['total_trees_grown'] += grown_now
    stats['total_trees_burned'] += burned_now

    prev_grid = grid.copy()

    cur = count_states(grid)

    stats['history']['step'].append(frame + 1)
    stats['history']['empty'].append(cur['empty'])
    stats['history']['tree'].append(cur['tree'])
    stats['history']['burning'].append(cur['burning'])

    im.set_data(grid)
    title.set_text(f"Лесной пожар, шаг {frame + 1}")

    stats_text.set_text(
        format_stats_text(
            step=frame + 1,
            cur=cur,
            grown_now=grown_now,
            burned_now=burned_now,
            total_grown=stats['total_trees_grown'],
            total_burned=stats['total_trees_burned'],
            total_cells=total_cells
        )
    )

    return im, title, stats_text


ani = FuncAnimation(
    fig,
    update,
    frames=STEPS,
    interval=50,
    blit=False,
    repeat=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()



fig.savefig('forest_fire_final.png', dpi=150, bbox_inches='tight')
print("Сохранён конечный шаг: forest_fire_final.png")


final = count_states(grid)

print("=" * 50)
print("ИТОГИ ЭКСПЕРИМЕНТА")
print("=" * 50)
print(f"Размер поля:            {ROWS} x {COLS} = {total_cells} клеток")
print(f"Шагов симуляции:        {STEPS}")
print(f"Начальная плотность:    {TREE_PROB}")
print(f"Вероятность роста p:    {P_GROW}")
print(f"Вероятность молнии f:   {F_LIGHTNING}")
print(f"Порог загорания k:      {K_THRESHOLD}")
print(f"Окрестность:            {NEIGHBORHOOD} соседей")
print("-" * 50)
print("СОСТОЯНИЕ НА КОНЕЦ СИМУЛЯЦИИ:")
print(f"  Пусто:    {final['empty']:5d}  ({final['empty']/total_cells*100:5.1f}%)")
print(f"  Деревья:  {final['tree']:5d}  ({final['tree']/total_cells*100:5.1f}%)")
print(f"  Горит:    {final['burning']:5d}  ({final['burning']/total_cells*100:5.1f}%)")
print("-" * 50)
print("СУММАРНЫЕ СОБЫТИЯ ЗА ВСЮ СИМУЛЯЦИЮ:")
print(f"  Всего деревьев выросло:  {stats['total_trees_grown']}")
print(f"  Всего деревьев сгорело:  {stats['total_trees_burned']}")
print("=" * 50)