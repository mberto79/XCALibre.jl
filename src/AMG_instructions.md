### **The AI Agent Prompt**

> **Task:** Implement an Algebraic Multigrid (AMG) linear solver for the `XCALibre.jl` package.
> **Constraints:** > 1. **Zero Third-Party Dependency Rule:** You may only use Base Julia and `KernelAbstractions.jl`. Do NOT rely on vendor-specific libraries or external C/C++ dependencies. Only use packages that are already dependencies in XCALibre.jl.
> 2. **Hardware Agnosticism:** The solver must compile and run on multi-threaded CPUs, NVIDIA, AMD, and Intel GPUs identically via `KernelAbstractions.jl`. If backend specific implementations are needed you can use the relevant extensions in XCALibre.jl located in the ext directory e.g. XCALibre_CUDAExt.jl for CUDA backend.
> 3. **API Integration:** The solver struct and methods must cleanly implement and dispatch into the existing `XCALibre.jl` linear solver API interface.
> **Design Inspiration:** Draw architectural inspiration from NVIDIA's `AMGX` and `AlgebraicMultigrid.jl` (e.g., V-cycles, Smoothed Aggregation or Ruge-Stüben coarsening, and parallel-friendly smoothers like Damped Jacobi/Chebyshev). 
> **Execution:** Follow the 5-step implementation plan provided below. For each step, provide highly optimized, allocation-free (where possible) kernel implementations.

***

### **Implementation Breakdown (5 Steps)**

**Step 1: Core Data Structures & XCALibre API Hookup**
Define the foundational structs. You will need a `MultigridLevel` struct to store the system matrix, prolongation/restriction operators, and smoothers for a single grid level. Then, create the parent `AMGSolver` struct that holds the hierarchy of levels and hooks into `XCALibre.jl`'s specific initialization and `solve!` interface.

**Step 2: Backend-Agnostic Sparse Linear Algebra Kernels**
Since you cannot use vendor sparse libraries, write custom `KernelAbstractions.jl` kernels for standard sparse operations. The most critical component is a highly optimized Sparse Matrix-Vector multiplication (SpMV) kernel (using CSR format to exploit the approach already used in XCALibre.jl - you may use alternative formats for prolongation or restriction if they are more performant) and basic vector BLAS operations (dot products, axpy) that map efficiently to both CPU threads and GPU workgroups.

**Step 3: The Setup Phase (Coarsening & Operators)**
Implement the hierarchy generation. Write parallel-friendly coarsening strategies (for Smoothed Aggregation and classical Ruge-Stüben). Write kernels to build the prolongation operator ($P$), the restriction operator ($R = P^T$), and compute the coarse grid matrix via Galerkin projection ($A_{c} = R A P$). 

**Step 4: Parallel Smoothers & Coarse Solver**
Implement the smoothing algorithms applied at each grid level. Avoid inherently sequential smoothers like Gauss-Seidel. Instead, write `KernelAbstractions.jl` kernels for Damped Jacobi (and implementation already exists in XCALibre.jl Solve/Smoothers it may need to be improved using KernelAbstractions.jl) and Polynomial (Chebyshev) smoothing, which parallelize exceptionally well across GPU threads. For the coarsest level, implement a simple exact solver.

**Step 5: The Multigrid Cycle (Solve Phase)**
Assemble the components from the previous steps into the actual solve iteration (V-cycle and W-cycle). Write the top-level loop that handles the pre-smoothing, calculation of the residual, restriction to the coarser grid, recursive/iterative coarse solve, prolongation of the correction, and post-smoothing. Ensure all memory allocations are handled during the Setup phase so the Solve phase is non-allocating. Since the AMG solver will be used inside iterative loops for solving fluids problems, I would also need a `update!` function to update the underlying sparse matrix and reuse the storage for improved performance.

Note: implement the code as part of the Solve module, inside a new folder `AMG`. When you have an implementation plan ready, save it as a markdown file in the root folder i.e. in /src

API: the user facing API should resemble the current approach, for example:

solvers = (
    U = SolverSetup(
        ...
    ),
    p = SolverSetup(
        solver      = AMG(smoother=Jacobi(), cycle=VCycle(), ...), # USER API 
        convergence = 1e-8,
        relax       = 0.2,
        rtol = 1e-3,
    ),
    ...
)