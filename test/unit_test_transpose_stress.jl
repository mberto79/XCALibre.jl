using XCALibre
using LinearAlgebra
using StaticArrays
using Test

# The explicit stress source is ∇·(ν dev2((∇U)ᵀ)), component i = ∂_j(ν ∂U_j/∂x_i) - ⅔ ∂_i(ν ∇·U).
# For U = G x with tr(G) = 0 and ν = ν0 + c x, it is c (G_xx, G_xy, G_xz), i.e. c ∂U_x/∂x_i,
# and zero for constant ν. With ∇U = G in every cell and ν exact at the faces, the face fluxes
# are exact, so the finite-volume divergence is exact in every cell.

const TS = XCALibre.Solvers

ts_grids = pkgdir(XCALibre, "examples/0_GRIDS")

function stress_setup(TF)
    mesh = UNV2D_mesh(joinpath(ts_grids, "quad40.unv"), scale=0.001, float_type=TF)
    hardware = Hardware(backend=CPU(), workgroup=64)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Zerogradient(:outlet),
             Wall(:bottom, [0.0, 0.0, 0.0]), Wall(:top, [0.0, 0.0, 0.0])],
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
        source = TS.stress_source(mesh)
        fluxes = TS.stress_fluxes(mesh, true)
        TS.transpose_stress!(source, fluxes, nu, gradU, config.boundaries.U, config)

        @test eltype(source.x.values) === TF
        @test eltype(fluxes[1].values) === TF
        tol = TF === Float64 ? 1e-9 : 2e-3
        @test maximum(abs.(source.x.values .- c*ts_a)) < tol
        @test maximum(abs.(source.y.values .- c*ts_b)) < tol
        @test maximum(abs.(source.z.values)) < tol

        # constant viscosity: the term vanishes for a divergence-free field
        U, gradU, nu = stress_fields(mesh, TF, zero(TF))
        TS.transpose_stress!(source, fluxes, nu, gradU, config.boundaries.U, config)
        @test maximum(abs.(source.x.values)) < tol
        @test maximum(abs.(source.y.values)) < tol

        # switched off: no work space, source untouched
        @test TS.stress_fluxes(mesh, false) === nothing
        fill!(source.x.values, 7)
        TS.transpose_stress!(source, nothing, nu, gradU, config.boundaries.U, config)
        @test all(==(7), source.x.values)
    end

    # The cell gradient uses the same index convention, G_ij = ∂U_i/∂x_j (cells away from the
    # boundary, where the Gauss gradient of a linear field is exact on this uniform mesh)
    mesh, config = stress_setup(Float64)
    U, gradU, nu = stress_fields(mesh, Float64, 1.0)
    Uf = FaceVectorField(mesh)
    grad!(gradU, Uf, U, config.boundaries.U, 0.0, config)
    interior = [i for i ∈ eachindex(mesh.cells) if all(0.05 .< mesh.cells[i].centre[1:2] .< 0.95)]
    @test maximum(norm(gradU[i] - ts_G(Float64)) for i ∈ interior) < 1e-9

    # Slip and symmetry faces carry no explicit stress flux
    fx, fy, fz = TS.stress_fluxes(mesh, true)
    U, gradU, nu = stress_fields(mesh, Float64, 1.0)
    BCs = assign(region=mesh, (
        U = [Dirichlet(:inlet, [1.0, 0.0, 0.0]), Zerogradient(:outlet),
             Symmetry(:bottom), Slip(:top)],
    ))
    TS.explicit_shear_stress!(fx, fy, fz, nu, gradU, BCs.U, config)
    for name ∈ (:bottom, :top)
        r = mesh.boundaries[findfirst(b -> b.name == name, mesh.boundaries)].IDs_range
        @test all(iszero, fx.values[r]) && all(iszero, fy.values[r])
    end
end
