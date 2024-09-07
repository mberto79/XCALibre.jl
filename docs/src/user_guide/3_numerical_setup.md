# Numerical setup
*super brief summary*

## Discretisation schemes
---

```@example
using XCALibre # hide
using AbstractTrees # hide
import Main.subtypes as subtypes # hide
AbstractTrees.children(d::DataType) = subtypes(d) # hide
print_tree(AbstractScheme) # hide
```

## Linear solvers
---