
export Preconditioner, PreconditionerType
export Jacobi, NormDiagonal #, ILU0 # , LDL
export DILU, DILUprecon
export BoomerAMG, GAMG
export IC0GPU, ILU0GPU

abstract type PreconditionerType end
abstract type LDIVPreconditioner <: PreconditionerType end
abstract type MULPreconditioner <: PreconditionerType end

struct NormDiagonal <: MULPreconditioner end
Adapt.@adapt_structure NormDiagonal

struct Jacobi <: MULPreconditioner end
Adapt.@adapt_structure Jacobi


struct DILU <: LDIVPreconditioner end
Adapt.@adapt_structure DILU

"""
    BoomerAMG(; kwargs...) <: PreconditionerType

HYPRE BoomerAMG via PETSc (`-pc_type hypre`). Distributed meshes only; needs a PETSc with hypre,
which the stock Float64 libraries include (Float32 ones do not). SPD systems only (e.g. pressure).
Use it where [`GAMG`](@ref) converges poorly or the strongest reduction per solve matters; GAMG
is the default recommendation, since it needs no special build and is the better AMG on the GPU.
Like any AMG, results vary slightly with the rank count, and it needs more memory than `Jacobi`.

Defaults are tuned for 3D (`BOOMERAMG_3D_DEFAULTS`): PETSc's own BoomerAMG defaults build a
high-complexity hierarchy in 3D that degrades as the mesh grows. On a GPU with a CUDA-built hypre,
PETSc switches to PMIS coarsening and l1-Jacobi smoothing, so GPU and CPU residuals differ.

`freeze=N` holds the whole preconditioner fixed for `N` solves and rebuilds it from the current
matrix on the `N`th (default 10); in between, Krylov iterates on the current matrix with a
hierarchy built from an older one. In SIMPLE the pressure matrix changes VALUES only, so a stale
hierarchy stays a good preconditioner, and a full hypre rebuild every solve otherwise dominates
runtime. `freeze=1` rebuilds every solve. Longer freezes cut setup cost further but let the
stale hierarchy degrade the pressure residual.

On GPU fields the solver refuses `BoomerAMG()` unless PETSc's hypre reports device execution, since
a host-only hypre crashes there; `device=true` skips that check for a build the query misreads.

Each other keyword `k=v` overrides a default and is passed to PETSc as `-pc_hypre_boomeramg_<k> v`.
Common knobs: `strong_threshold` (0.5-0.7 in 3D), `coarsen_type` ("HMIS"/"PMIS"/"Falgout"),
`interp_type` ("ext+i"/"classical"), `P_max` (interpolation entries per row), `agg_nl` (aggressive-coarsening levels), `relax_type_all`
(smoother, e.g. "SOR/Jacobi"/"Chebyshev"), `grid_sweeps_all` (smoother sweeps). See the PETSc
`-pc_hypre_boomeramg_*` options for the full list; anything else can go through `petsc_options`.
"""
struct BoomerAMG{NT<:NamedTuple} <: PreconditionerType
    opts::NT
    freeze::Int
    device::Bool # skip the hypre device-execution check on GPU fields
end

# 3D CFD-Poisson defaults: low operator complexity that scales with mesh size
const BOOMERAMG_3D_DEFAULTS = (
    strong_threshold = 0.7, coarsen_type = "HMIS", interp_type = "ext+i", P_max = 4,
    agg_nl = 1, agg_num_paths = 2)
BoomerAMG(; freeze=10, device=false, kwargs...) =
    BoomerAMG(merge(BOOMERAMG_3D_DEFAULTS, NamedTuple(kwargs)), freeze, device)

"""
    GAMG(; kwargs...) <: PreconditionerType

PETSc native aggregation AMG (`-pc_type gamg`). Distributed meshes only; needs `using PETSc`
(no hypre build required). SPD systems only (e.g. pressure). The recommended AMG for distributed
pressure solves on large meshes, on CPU and GPU. It needs more memory than `Jacobi` and pays off
only when each rank holds a large partition. The hierarchy is built per partition, so residuals
vary slightly with the rank count; use `Jacobi` where results must match across rank counts exactly.

Defaults set `reuse_interpolation=true`: because the mesh never refines, the pressure matrix
keeps a FIXED sparsity pattern, so GAMG builds the aggregation + prolongation P once and
recomputes only the coarse operators (RAP) and smoothers each solve — the hierarchy stays
numerically current at a fraction of a full setup. `freeze=N` additionally holds the whole
preconditioner fixed for `N` solves, skipping even that update, and rebuilds it on the `N`th
(default 25; `freeze=1` updates every solve). Because the coefficient drift between rebuilds is
small, the freeze leaves the pressure residual essentially unchanged while removing most of the
setup cost.

Each keyword `k=v` overrides a default and is passed as `-pc_gamg_<k> v`, e.g.
`GAMG(threshold=0.02, square_graph=1)`. See the PETSc `-pc_gamg_*` options.
"""
struct GAMG{NT<:NamedTuple} <: PreconditionerType
    opts::NT
    freeze::Int
end

# reuse_interpolation valid only for SAME_NONZERO_PATTERN — guaranteed here (no mesh refinement)
const GAMG_DEFAULTS = (reuse_interpolation = true,)
GAMG(; freeze=25, kwargs...) = GAMG(merge(GAMG_DEFAULTS, NamedTuple(kwargs)), freeze)

struct IC0GPU <: MULPreconditioner end
Adapt.@adapt_structure IC0GPU

struct ILU0GPU <: MULPreconditioner end
Adapt.@adapt_structure ILU0GPU

struct Preconditioner{T,M,P,S}
    A::M
    P::P
    storage::S
end
function Adapt.adapt_structure(to, itp::Preconditioner{T,M,Pr,S}) where {T,M,Pr,S}
    A = Adapt.adapt(to, itp.A)
    P = Adapt.adapt(to, itp.P)
    storage = Adapt.adapt(to, itp.storage) 
    Preconditioner{T,typeof(A),typeof(P),typeof(storage)}(A,P,storage)
end

is_ldiv(precon::Preconditioner{T,M,P,S}) where {T,M,P,S} = T <: LDIVPreconditioner

Preconditioner{PT}(A) where {PT<:BoomerAMG} = error(
    "BoomerAMG runs through PETSc on distributed meshes only; use Jacobi/DILU/etc for serial runs")

Preconditioner{PT}(A) where {PT<:GAMG} = error(
    "GAMG runs through PETSc on distributed meshes only; use Jacobi/DILU/etc for serial runs")

Preconditioner{NormDiagonal}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(F, m), backend)
    P = diagonal_operator(S)
    Preconditioner{NormDiagonal,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{Jacobi}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(F, m), backend)
    P = diagonal_operator(S)
    Preconditioner{Jacobi,typeof(A),typeof(P),typeof(S)}(A,P,S)
end


struct DILUprecon{M,V,VI}
    A::M
    D::V
    Di::VI
end
Adapt.@adapt_structure DILUprecon

Preconditioner{DILU}(A::SparseXCSR{N,F,I}) where {N,F,I} = begin
    m, n = size(A)
    m == n || throw("Matrix not square")
    Acsr = parent(A)
    D = zeros(F, m)
    Di = zeros(I, m)
    diagonal_indices!(Di, Acsr)
    S = DILUprecon(Acsr, D, Di)
    P = S
    Preconditioner{DILU,typeof(Acsr),typeof(P),typeof(S)}(Acsr,P,S)
end
