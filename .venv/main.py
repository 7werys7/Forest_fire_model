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
NEIGHBORHOOD = 8
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


grid = init_forest(ROWS, COLS, TREE_PROB, SEED)


colors = ['#1a1a1a', '#2e8b57', '#ff4500']
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
title = ax.set_title("Лесной пожар, шаг 0")
ax.axis('off')

cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
cbar.ax.set_yticklabels(['Пусто', 'Дерево', 'Огонь'])


def update(frame):
    global grid
    grid = step(grid, P_GROW, F_LIGHTNING, K_THRESHOLD, NEIGHBORHOOD)
    im.set_data(grid)
    title.set_text(f"Лесной пожар, шаг {frame + 1}")
    return im, title


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