## Causal Discovery Algorithms
```python
template = Callable[[DataFrame, float, ndarray], ndarray]
```

### PC
Inputs:  `ndarray, float, Path | str, int`

Outputs: `tuple[ndarray, ndarray]`

### cuPC
Inputs:  `ndarray, float, Path | str`

Outputs: `tuple[ndarray, ndarray] | None`

### SP-GIES
Inputs:  `DataFrame, Path | str, float, ndarray, bool, list[list[Any]], bool`

Outputs: `ndarray`

***

## Fusion Algorithms
```python
template = Callable[[ndarray, dict], DiGraph]
```

### Fusion Algorithm
Inputs:  `dict[Any, Any], list[ndarray], Any`

Outputs: `Digraph`

### Fusion-Basic Algorithm
Inputs:  `dict[Any, Any], list[ndarray]`

Outputs: `Digraph`

### Screen-Projections Algorithm
Inputs:  `dict[Any, Any], list[ndarray]`

Outputs: `DiGraph`

***

## Partitioning Algorithms
```python
template = Callable[[dict, list[ndarray]], dict]
```

### Expansive Causal Partition Algorithm
Inputs:  `ndarray, dict`

Outputs: `dict`

### Modularity Partition Algorithm
Inputs:  `ndarray, int, int, int | None`

Outputs: `dict`

### Hierarchical Partition Algorithm
Inputs:  `ndarray, float`

Outputs: `dict | None`

### Random Edge Cover Partition
Inputs:  `ndarray, dict`

Outputs: `dict`
