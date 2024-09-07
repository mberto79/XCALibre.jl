# Numerical setup
*super brief summary*

## Discretisation schemes
---

```@repl
using XCALibre # hide
using AbstractTrees # hide
# import Main.subtypes as subtypes # hide
    using InteractiveUtils # hide
AbstractTrees.children(d::DataType) = Main.subtypes(d) # hide
print_tree(AbstractScheme) # hide
```

## Linear solvers
---