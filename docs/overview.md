# Overview

## Key Terminology

### Superstructure
Given a true graph which forms a DAG, $G=(V,E)$, a superstructure can be understood as
a super-graph, $G'=(V,E')$, where and $E' \supset E$. Below is a visual example of a
true graph and a superstructure.


```mermaid
flowchart
    a1((a))
    a2((a))
    b1((b))
    b2((b))
    c1((c))
    c2((c))
    d1((d))
    d2((d))

    subgraph Superstructure
        a2-->b2
        a2-.->c2
        a2-.->d2
        b2-->c2
        b2-->d2
        c2-->d2
    end

    subgraph Graph
        a1-->b1
        b1-->c1
        b1-->d1
        c1-->d1
    end
```

### Colliders
A collider is a child node that shares two disconnected causal parents.

```mermaid
flowchart TB
    a((a))
    b((b))
    c((c))
    a-->c
    b-->c
```
