* changed: On a distributed mesh `AMG(Pairwise(merges))` runs PETSc's multigrid over XCALibre's own per-rank pairwise aggregates instead of GAMG, and accepts `merges > 1`.
