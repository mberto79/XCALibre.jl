# Development nodes, hints and ideas

## To do

### Mesh module

- [x] Face centres calculations
- [x] Face area calculations
- [x] Face normals & suitable way to find direction from cell
- [x] Deltas (distance between cells in direction of normal)
- [x] Node interpolation weight functions
- [x] Face interpolation weight functions
- [ ] Separation of boundary and internal nodes, faces and cells (all?)
- [x] Rethink data structure for mesh
- [x] Write mesh to VTK format
- [ ] Extend to 3D
- [ ] Write conversion tool to read grids from OpenFOAM
- [ ] Include boundary information in VTK file for postprocessing

### Solution for Convection/Diffusion equations (FVM)

- [x] Solution for diffusion equation (normal contribution only)
- [x] Solution for convection equation (normal contribution only)
- [x] Solution for convection-diffusion (normal contribution only)
- [x] Implementation of boundary conditions (diffusion)
  - [x] Dirichlech
  - [x] Neumann
  - [ ] Symmetry
  - [ ] Wall
  - [ ] Slip
- [x] Implementation of boundary conditions (convection)
  - [x] Dirichlech
  - [x] Neumann
  - [ ] Symmetry
  - [ ] Wall
  - [ ] Slip
- [x] A simple plotting to check results
- [ ] Extend code for solution including non-orthogonal correction
- [x] Implement second order upwind scheme
- [x] Implement gradient interpolation
- [x] Implement variable interpolation (i.e. allow variable diffusion coefficients)
- [ ] Think about easy, robust and maintainable API for general systems

### Physics

- [x] Implement laminar solver (steady)
- [ ] Implement laminar solver (transient)
- [ ] Implement first RANS turbulence model e.g. $k-\omega$
- [ ] Calculation of y for first cell (can be from mesh.delta?)
- [ ] Calculation of wall normal distance needed for turbulence models
- [ ] Implement wall functions
- [ ] LES solver

### Algorithms

- [x] Steady state SIMPLE
- [ ] Extend to SIMPLEC
- [ ] Transient PISO

### Computing

- [x] Use Julia's SCS sparse matrix format
- [x] Implement linear system solver - IterativeSolvers.jl
- [ ] Allow matrix free methods?
- [ ] Write more efficient matrix build method for sparse matrix
- [ ] Enable threading
- [ ] Enable GPU computation

## Accessing data in structs

```julia
# These extract the x coordinate from "points"

(i->points[i].coords[1]).([1:length(coords);])

[points[i].coords[1] for i ∈ 1:length(coords)]

[points[i].coords[1] for i ∈ [20, 75, 100]]

getproperty.(points, :coords)[1]
```

## Performance of different functions

```julia

function remove_duplicates(vectorList)
    matched = Int32[]
    result = Vector{Int32}[]
    for vector ∈ vectorList
        for idx ∈ 1:length(vectorList)
            if vectorList[idx] == vector
                push!(matched, idx)
            end
        end
        push!(result,matched)
        matched = Int32[]
    end
    return result
end

function remove_duplicates(orientedFaces)
    result = Vector{Int32}[]
    orientedFacesIDs = getproperty.(orientedFaces, :ids)
    for i ∈ 1:length(orientedFaces)
        match = findall(x->x==orientedFacesIDs[i], orientedFacesIDs)
        push!(result, match)
    end
    return result
end
```

# Theory notes

## FVM: Diffusion equation

Discretise the diffusion equation given below:
$$
\nabla \cdot (\Gamma \nabla \phi) = S_\phi
$$

Integrating both sides

$$
\int_V \nabla \cdot (\Gamma \nabla \phi) dV = \int_V S_\phi dV
$$

Assuming the source term is constant within each volume. The right-hand side becomes:

$$
\int_V \nabla \cdot (\Gamma \nabla \phi) dV = \bar{S_\phi} V
$$

The left-hand side can be simplified applying Gauss-Divergence theorem:

$$
\int_S  (\Gamma \nabla \phi) \cdot \hat{n}dA = \bar{S_\phi} V
$$

For a volume made up of discrete faces, we can write

$$
\sum_f  \Gamma_f (\nabla \phi)_f \cdot \hat{n}_f A_f = \bar{S_\phi} V \qquad ((1)
$$

These terms need to be discretised. First take $\Gamma_f$:

$$
\Gamma_f = w_f \Gamma_P + (1 - w_f)\Gamma_1
$$

Here, $w_f$ is an inverse-distance weight function for each face of a cell. In equation (1)
above, the face area, $A_f$, and the unit face-normal vector, $\hat{n}$, are known (i.e. can
be calculated from the mesh/geometry)

TO BE CONTINUED

This can be split into two component (using a local coordinate system), in 2D:

$$
(\nabla \phi)_f \cdot \hat{n}_f = \frac{\phi_1 - \phi_P}{\delta} - J_t
$$

Where, $\delta = \hat{l}_f \cdot \hat{n}_f$. Here $\hat{l}_f$ is the vector that links two
face centres. Therefore, $\delta$ is the face normal distance between two adjacent cells. Now,
$J_t$ is given by:

$$
J_t = \frac{[(\nabla \phi)_f \cdot \hat{t}_f] \hat{t}_f \cdot l}{\delta}
$$

This can be discretised (after some math that I will have to write up eventually):

$$
J_t = \left[ \frac{\phi_{n1,f} - \phi_{n2,f}}{\delta_f |t_f|} \right ] \hat{t}_f \cdot l_f
$$

The final discretised equation (again need to extend in documentation later, but this is
good enough for now so I can get on with the implementation) is:

$$
a_0 \phi_0 + \sum_{n} a_{n,f}\phi_{n} = Q_0
$$

Where, the subscript $n$ denotes neighbour cells and $f$ the common face shared (there is multiplication by -1, hence the negative source term)

$$
a_0 = \sum_n \frac{\Gamma_{n,f} A_{n,f}}{\delta_n}
$$

$$
a_{n} = - \frac{\Gamma_{n,f} A_{n,f}}{\delta_{n,f}}
$$

$$
Q_0 = -\bar{S}_0 V_0 + S_{skew}
$$

## FVM: Convection equation

The convection equation is given by

$$
\nabla \cdot (\rho u \phi) = 0
$$

Integrating over volume

$$
\int_V \nabla \cdot (\rho u \phi) dV = 0
$$

Apply, Gauss-Divergence theorem:

$$
\int_S (\rho u \phi) \cdot \hat{n}A = 0
$$

The surface is made up of discrete faces, therefore:

$$
\sum_f (\rho u \phi)_f \cdot \hat{n}_f A_f = 0
$$

For one cell $P$ and one of its neighbours $N_1$. Assuming the flux, $F = \rho u$ is constant, and using central difference (equally spaced grid - for now) we have:

$$
F A_{f_1} (0.5 \phi_P + 0.5 \phi_{N_1}) \cdot \hat{n}_{f_1} = 0
$$

Extending the above equation to include all neighbours, we have:

$$
\sum_N F A_{f_N} (0.5 \phi_P + 0.5 \phi_{N}) \cdot \hat{n}_{f_N} = 0
$$

Separating cell and neighbour contributions:

$$
\sum_N F A_{f_N} (0.5 \phi_P)\cdot \hat{n}_{f_N} + \sum_N F A_{f_N} (0.5 \phi_{N}) \cdot \hat{n}_{f_N} = 0
$$

This results in the general equation for cell $P$

$$
a_p \phi_P + \sum_N a_N \phi_{N} = 0
$$

Where,

$$
a_p = \sum_N 0.5 F A_{f_N} \cdot \hat{n}_{f_N}
$$

$$
a_N = 0.5 F A_{f_N} \cdot \hat{n}_{f_N}
$$