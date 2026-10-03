import numpy as np


EMPTY = 0
TREE = 1
BURNING = 2


def init_forest(rows, cols, tree_prob=0.7, seed=None):

    rng = np.random.default_rng(seed)
    grid = rng.choice(
        [EMPTY, TREE],
        size=(rows, cols),
        p=[1 - tree_prob, tree_prob]
    )
    return grid.astype(np.uint8)


def count_burning_neighbors(grid, i, j, neighborhood=8):

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


def step(grid, p=0.01, f=0.0001, k=1, neighborhood=8):

    rows, cols = grid.shape
    new_grid = grid.copy()

    for i in range(rows):
        for j in range(cols):
            state = grid[i, j]

            if state == BURNING:

                new_grid[i, j] = EMPTY

            elif state == TREE:
                burning_neighbors = count_burning_neighbors(
                    grid, i, j, neighborhood
                )


                if burning_neighbors >= k or np.random.random() < f:
                    new_grid[i, j] = BURNING
                else:
                    new_grid[i, j] = TREE

            elif state == EMPTY:
                burning_neighbors = count_burning_neighbors(
                    grid, i, j, neighborhood
                )


                if burning_neighbors == 0 and np.random.random() < p:
                    new_grid[i, j] = TREE
                else:
                    new_grid[i, j] = EMPTY

    return new_grid


def run_simulation(grid, steps, p=0.01, f=0.0001, k=1, neighborhood=8):

    for _ in range(steps):
        grid = step(grid, p=p, f=f, k=k, neighborhood=neighborhood)
    return grid


if __name__ == "__main__":

    rows, cols = 50, 50

    forest = init_forest(rows, cols, tree_prob=0.7, seed=42)

    forest = run_simulation(
        forest,
        steps=100,
        p=0.01,
        f=0.0001,
        k=1,
        neighborhood=8
    )


    values, counts = np.unique(forest, return_counts=True)
    for v, c in zip(values, counts):
        if v == EMPTY:
            name = "пусто"
        elif v == TREE:
            name = "деревья"
        else:
            name = "горит"
        print(f"{name}: {c}")