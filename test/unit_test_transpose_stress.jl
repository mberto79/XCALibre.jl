using XCALibre
using LinearAlgebra
using StaticArrays
using Test

# div!(phi::VectorField, Γf, tensor, BCs, config) computes ∇·(Γ tensor). The viscous stress
# source of the solvers, ∇·(ν dev2((∇U)ᵀ)), is that operator applied to Dev2(T(gradU)); its
# component i is ∂_j(ν ∂U_j/∂x_i) - ⅔ ∂_i(ν ∇·U). For U = G x with tr(G) = 0 and ν = ν0 + c x it
# is c (G_xx, G_xy, G_xz), i.e. c ∂U_x/∂x_i, and zero for constant ν. With ∇U = G in every cell
# and ν exact at the faces, the face fluxes are exact, so the source is exact in every cell.

const TS = XCALibre.Solvers

ts_grids = pkgdir(XCALibre, "examples/0_GRIDS")

function stress_setup(TF; grid="quad40.unv")
    mesh = UNV2D_mesh(joinpath(ts_grids, grid), scale=0.001, float_type=TF)
    hardware = Hardware(backend=CPU(), workgroup=64)
    names = [b.name for b ∈ mesh.boundaries]
    BCs = assign(region=mesh, (
        U = [Dirichlet(names[1], [1.0, 0.0, 0.0]), Zerogradient(names[2]),
             Wall(names[3], [0.0, 0.0, 0.0]), Wall(names[4], [0.0, 0.0, 0.0])],
    ))
    config = Configuration(
        solvers=(U=SolverSetup(solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-8, relax=1.0),),
        schemes=(U=Schemes(),), runtime=Runtime(iterations=1, write_interval=-1, time_step=1),
        hardware=hardware, boundaries=BCs)
    mesh, config
end

# U = G x with G_ij = ∂U_i/∂x_j: a = ∂U_x/∂x, b = ∂U_x/∂y, d = ∂U_y/∂x, divergence free
const ts_a, ts_b, ts_d = 0.3, 1.7, -0.9
ts_G(TF) = SMatrix{3,3,TF}(ts_a, ts_d, 0, ts_b, -ts_a, 0, 0, 0, 0)

function stress_fields(mesh, TF, c)
    U = VectorField(mesh)
    for i ∈ eachindex(mesh.cells)
        x, y, _ = mesh.cells[i].centre
        U.x.values[i] = ts_a*x + ts_b*y
        U.y.values[i] = ts_d*x - ts_a*y
    end
    gradU = Grad{Gauss}(U)
    for i ∈ eachindex(mesh.cells)
        gradU.result[i] = ts_G(TF)
    end
    nu = FaceScalarField(mesh)
    for f ∈ eachindex(mesh.faces)
        nu.values[f] = 1 + c*mesh.faces[f].centre[1]
    end
    U, gradU, nu
end

@testset "Explicit viscous stress ∇·(ν dev2((∇U)ᵀ))" begin
    for TF ∈ (Float64, Float32)
        mesh, config = stress_setup(TF)
        c = TF(2.5)
        U, gradU, nu = stress_fields(mesh, TF, c)
        source = VectorField(mesh)
        TS.transpose_stress!(source, nu, gradU, config.boundaries.U, config)

        @test eltype(source.x.values) === TF
        tol = TF === Float64 ? 1e-9 : 2e-3
        @test maximum(abs.(source.x.values .- c*ts_a)) < tol
        @test maximum(abs.(source.y.values .- c*ts_b)) < tol
        @test maximum(abs.(source.z.values)) < tol

        # constant viscosity: the term vanishes for a divergence-free field
        U, gradU, nu = stress_fields(mesh, TF, zero(TF))
        TS.transpose_stress!(source, nu, gradU, config.boundaries.U, config)
        @test maximum(abs.(source.x.values)) < tol
        @test maximum(abs.(source.y.values)) < tol
    end

    # The cell gradient uses the same index convention, G_ij = ∂U_i/∂x_j (cells away from the
    # boundary, where the Gauss gradient of a linear field is exact on this uniform mesh)
    mesh, config = stress_setup(Float64)
    U, gradU, nu = stress_fields(mesh, Float64, 1.0)
    Uf = FaceVectorField(mesh)
    grad!(gradU, Uf, U, config.boundaries.U, 0.0, config)
    interior = [i for i ∈ eachindex(mesh.cells) if all(0.05 .< mesh.cells[i].centre[1:2] .< 0.95)]
    @test maximum(norm(gradU[i] - ts_G(Float64)) for i ∈ interior) < 1e-9
end

@testset "Divergence of a tensor field, div!(::VectorField, Γf, tensor, BCs, config)" begin
    # Linear tensor field A + B x + C y on a graded mesh: (∇·T)_i = Σ_j ∂_j T_ij = B_i1 + C_i2.
    # Linear interpolation with the mesh weights reproduces it at every internal face, so cells
    # with only internal faces get it exactly; equal weights would not on this mesh.
    mesh, config = stress_setup(Float64; grid="flatplate_2D_lowRe.unv")
    nb = length(mesh.boundary_cellsID)
    @test maximum(abs(f.weight - 0.5) for f ∈ mesh.faces[nb+1:end]) > 0.01

    A = @SMatrix [1.0 2.0 0.0; -1.0 0.5 0.0; 0.0 0.0 0.0]
    B = @SMatrix [3.0 -2.0 0.0; 4.0 1.0 0.0; 0.0 0.0 0.0]
    C = @SMatrix [-1.0 5.0 0.0; 2.0 -3.0 0.0; 0.0 0.0 0.0]
    tensor = TensorField(mesh)
    for i ∈ eachindex(mesh.cells)
        x, y, _ = mesh.cells[i].centre
        tensor[i] = A + B*x + C*y
    end
    Γ = FaceScalarField(mesh)
    Γ.values .= 1
    phi = VectorField(mesh)
    div!(phi, Γ, tensor, config.boundaries.U, config)

    expected = B[:, 1] + C[:, 2]
    interior = setdiff(eachindex(mesh.cells), mesh.boundary_cellsID)
    err = maximum(norm(phi[i] - expected) for i ∈ interior)
    @test err < 1e-6*norm(expected)

    # Constant tensor: the fluxes through all faces of a cell sum to zero, so leaving out the
    # slip and symmetry faces leaves exactly minus their flux, -A n A_f / V, and zero elsewhere
    mesh, config = stress_setup(Float64)
    names = [b.name for b ∈ mesh.boundaries]
    BCs = assign(region=mesh, (
        U = [Dirichlet(names[1], [1.0, 0.0, 0.0]), Zerogradient(names[2]),
             Symmetry(names[3]), Slip(names[4])],
    ))
    tensor = TensorField(mesh)
    for i ∈ eachindex(mesh.cells)
        tensor[i] = A
    end
    Γ = FaceScalarField(mesh)
    Γ.values .= 1
    phi = VectorField(mesh)
    div!(phi, Γ, tensor, BCs.U, config)

    reference = zeros(SVector{3,Float64}, length(mesh.cells))
    for b ∈ mesh.boundaries[3:4], fID ∈ b.IDs_range
        f = mesh.faces[fID]
        cID = f.ownerCells[1]
        reference[cID] -= A*f.normal*f.area/mesh.cells[cID].volume
    end
    @test maximum(norm(phi[i] - reference[i]) for i ∈ eachindex(mesh.cells)) < 1e-9
end
