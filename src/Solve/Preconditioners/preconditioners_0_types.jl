
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

# struct LDL <: MULPreconditioner end
# Adapt.@adapt_structure LDL

# struct ILU0 <: MULPreconditioner end
# Adapt.@adapt_structure ILU0

struct DILU <: LDIVPreconditioner end
Adapt.@adapt_structure DILU

"""
    BoomerAMG(; kwargs...) <: PreconditionerType

HYPRE BoomerAMG via PETSc (`-pc_type hypre`). Distributed meshes only; requires a PETSc
build configured with `--download-hypre`. No transpose solve (use on SPD systems, e.g.
pressure). Recommended pressure preconditioner for distributed runs.

Defaults are tuned for 3D (`BOOMERAMG_3D_DEFAULTS`): hypre's own defaults are 2D-tuned and
build a high-complexity hierarchy in 3D that degrades as the mesh grows.

`reuse` rebuilds the AMG hierarchy only every `reuse` solves (default 10). In SIMPLE the
pressure matrix changes VALUES only, so a hierarchy stays a good preconditioner for many
iterations; rebuilding it every solve otherwise dominates runtime. `reuse=1` rebuilds every solve.

Each other keyword `k=v` overrides a default and is passed to PETSc as `-pc_hypre_boomeramg_<k> v`.
Common knobs: `strong_threshold` (0.5-0.7 in 3D), `coarsen_type` ("HMIS"/"PMIS"/"Falgout"),
`interp_type` ("ext+i"/"classical"), `agg_nl` (aggressive-coarsening levels), `relax_type_all`
(smoother, e.g. "SOR/Jacobi"/"Chebyshev"), `grid_sweeps_all` (smoother sweeps). See the PETSc
`-pc_hypre_boomeramg_*` options for the full list; anything else can go through `petsc_options`.
"""
struct BoomerAMG{NT<:NamedTuple} <: PreconditionerType
    opts::NT
    reuse::Int
end

# 3D CFD-Poisson defaults: low operator complexity that scales with mesh size
const BOOMERAMG_3D_DEFAULTS = (
    strong_threshold = 0.7, coarsen_type = "HMIS", interp_type = "ext+i",
    agg_nl = 1, agg_num_paths = 2)
BoomerAMG(; reuse=10, kwargs...) = BoomerAMG(merge(BOOMERAMG_3D_DEFAULTS, NamedTuple(kwargs)), reuse)

"""
    GAMG(; kwargs...) <: PreconditionerType

PETSc native aggregation AMG (`-pc_type gamg`). Distributed meshes only; needs `using PETSc`
(no hypre build required). SPD systems only (e.g. pressure).

Defaults set `reuse_interpolation=true`: because the mesh never refines, the pressure matrix
keeps a FIXED sparsity pattern, so GAMG builds the aggregation + prolongation P once and
recomputes only the coarse operators (RAP) and smoothers each solve — the hierarchy stays
numerically current at a fraction of a full setup. `reuse` additionally freezes the whole PC
for `reuse` solves via KSPSetReusePreconditioner (default 1 = let GAMG's cheap per-solve rebuild run).

Each keyword `k=v` overrides a default and is passed as `-pc_gamg_<k> v`, e.g.
`GAMG(threshold=0.02, square_graph=1)`. See the PETSc `-pc_gamg_*` options.
"""
struct GAMG{NT<:NamedTuple} <: PreconditionerType
    opts::NT
    reuse::Int
end

# reuse_interpolation valid only for SAME_NONZERO_PATTERN — guaranteed here (no mesh refinement)
const GAMG_DEFAULTS = (reuse_interpolation = true,)
GAMG(; reuse=1, kwargs...) = GAMG(merge(GAMG_DEFAULTS, NamedTuple(kwargs)), reuse)

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
    P = opDiagonal(S)
    Preconditioner{NormDiagonal,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

Preconditioner{Jacobi}(A::AbstractSparseArray{F,I}) where {F,I} = begin
    backend = get_backend(A)
    m, n = size(A)
    m == n || throw("Matrix not square")
    S = _convert_array!(zeros(F, m), backend)
    P = opDiagonal(S)
    Preconditioner{Jacobi,typeof(A),typeof(P),typeof(S)}(A,P,S)
end

# Preconditioner{LDL}(A::AbstractSparseArray{F,I}) where {F,I} = begin
#     m, n = size(A)
#     m == n || throw("Matrix not square")
#     S = zeros(F, m)
#     # P = similar(A)
#     # triu!(P)
#     # P  = opLDL(P)
#     # # P  = opLDL(P)
#     P  = opLDL(A)
#     Preconditioner{LDL,typeof(A),typeof(P),typeof(S)}(A,P,S)
# end

# Preconditioner{ILU0}(A::AbstractSparseArray{F,I}) where {F,I} = begin
#     m, n = size(A)
#     m == n || throw("Matrix not square")
#     S = ilu0(A)
#     P  = LinearOperator(
#         F, m, n, false, false, (y, v) -> ldiv!(y, S, v)
#         )
#     Preconditioner{ILU0,typeof(A),typeof(P),typeof(S)}(A,P,S)
# end

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
