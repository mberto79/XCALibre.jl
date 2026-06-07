# Phase 4 fixtures: real unstructured FVM Laplacian matrices spanning the row-length-variance
# axis. SELL-P targets irregular row lengths (prompt §3); poisson3d is structured (uniform 7-nnz)
# so it cannot show a SELL-P win. We discretise a conduction Laplacian on real meshes (2D tri/quad,
# 3D tet, polyhedral OF) to get the genuine mesh connectivity row-length distribution. The matrix
# VALUES are real Laplacian coefficients; BCs are not applied (they touch diagonal/source only, not
# the off-diagonal sparsity that drives SELL-P), so the structure is faithful for the SpMV bench.

using XCALibre
using SparseArrays, LinearAlgebra, Statistics
using XCALibre.Solve: _rowptr, _colval, _nzval, _amg_matrix, _m
using KernelAbstractions

# Discretise -Laplacian(rDf, T) on `mesh` and return the assembled operator as AMGMatrixCSR.
function laplace_matrix(mesh)
    backend = CPU(); workgroup = 1024
    model = Physics(time=Steady(), solid=Solid{Uniform}(k=1.0),
                    energy=Energy{Conduction}(), domain=mesh)
    src = ScalarField(mesh)
    initialise!(model.energy.T, 0.0)
    config = Configuration(
        solvers=(T=SolverSetup(solver=Cg(), preconditioner=Jacobi(), convergence=1e-8,
                               relax=1.0, rtol=1e-4, atol=1e-5),),
        schemes=(T=Schemes(laplacian=Linear),),
        runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=Hardware(backend=backend, workgroup=workgroup),
        boundaries=(T=(),))
    T_eqn = (
        - XCALibre.Laplacian{Linear}(model.solid.rDf, model.energy.T)
        == - XCALibre.Source(src)
    ) → ScalarEquation(model.energy.T, config.boundaries.T)
    prev = KernelAbstractions.zeros(backend, _get_float(mesh), length(mesh.cells))
    XCALibre.Discretise.discretise!(T_eqn, prev, config)
    return _amg_matrix(_A(T_eqn))
end

# Read a UNV/FOAM mesh and build its Laplacian operator.
function fixture_from_mesh(reader, file; scale=1.0)
    mesh = reader(file; scale=scale)
    return laplace_matrix(adapt(CPU(), mesh))
end

# Row-length distribution of a CSR matrix: drives the SELL-P adopt/reject verdict.
# Reports nnz/row mean,max,std,CV plus mean within-32-row-slice padding overhead (the actual
# SELL-P cost: padded_slice_nnz / true_slice_nnz - 1).
function rowlen_stats(A; slice=32)
    rp = _rowptr(A); n = _m(A)
    len = Int[rp[i+1] - rp[i] for i in 1:n]
    cv = std(len) / mean(len)
    nslices = cld(n, slice)
    padded = 0; true_nnz = 0
    for s in 1:nslices
        lo = (s-1)*slice + 1; hi = min(s*slice, n)
        smax = maximum(@view len[lo:hi])
        padded += smax * slice
        true_nnz += sum(@view len[lo:hi])
    end
    pad_overhead = padded / true_nnz - 1
    return (; n, nnz=sum(len), mean=mean(len), max=maximum(len),
            min=minimum(len), std=std(len), cv, pad_overhead)
end
