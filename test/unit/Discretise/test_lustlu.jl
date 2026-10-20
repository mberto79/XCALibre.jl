using XCALibre
using Test
using StaticArrays
using LinearAlgebra

# LUSTLU: LUST matrix plus the explicit gradient correction of its upwind quarter (as LinearUpwind)

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
backend = CPU(); workgroup = 1024
hardware = Hardware(backend=backend, workgroup=workgroup)
config = Configuration(
    solvers=nothing, schemes=nothing, boundaries=nothing, hardware=hardware,
    runtime=Runtime(iterations=1, write_interval=-1, time_step=1))

u = SVector(1.0, 0.6, 0.0)
lustlu_flux(mesh) = begin
    Uf = FaceVectorField(mesh)
    for fID ∈ eachindex(mesh.faces)
        Uf.x[fID], Uf.y[fID], Uf.z[fID] = u
    end
    mdotf = FaceScalarField(mesh)
    flux!(mdotf, Uf, config)
    mdotf
end

lustlu_residual(A, x, b) = begin
    nzval, colval, rowptr = get_sparse_fields(A)
    [sum(nzval[k]*x[colval[k]] for k ∈ rowptr[i]:rowptr[i+1]-1) - b[i] for i ∈ eachindex(b)]
end

lustlu_interior(mesh) = setdiff(1:length(mesh.cells), unique(mesh.boundary_cellsID))

trig = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "trig40.unv")))
quad = adapt(backend, UNV2D_mesh(joinpath(grids_dir, "quad40.unv")))

# the linear share is exact for a linear field only where the face centre lies between the cell
# centres (structured grid); on triangles the correction is checked against LinearUpwind's
@testset "LUSTLU: second-order consistency for a linear field" begin
    res = map((quad, trig)) do mesh
        mdotf = lustlu_flux(mesh)
        a = SVector(0.3e-3, -0.7e-3, 0.0)
        phi = ScalarField(mesh)
        phi.values .= [a⋅c + 2.0 for c ∈ mesh.cells.centre]
        gradphi = VectorField(mesh)
        for i ∈ eachindex(mesh.cells)
            gradphi.x[i], gradphi.y[i], gradphi.z[i] = a
        end
        r = map((LUSTLU, LUST, LinearUpwind, Upwind)) do scheme
            eqn = (
                Divergence{scheme}(mdotf, phi, gradphi) == Source(ConstantScalar(0.0))
            ) → ScalarEquation(phi, ())
            discretise!(eqn, phi, config)
            lustlu_residual(eqn.equation.A, phi.values, eqn.equation.b)
        end
        (r, (u⋅a) .* mesh.cells.volume)
    end
    (r, exact) = res[1]; interior = lustlu_interior(quad)
    @test isapprox(r[1][interior], exact[interior]; rtol=1e-10)
    (r, exact) = res[2]
    @test maximum(abs, r[1] .- r[2]) > 0
    @test r[1] .- r[2] ≈ 0.25 .* (r[3] .- r[4])
end

@testset "LUSTLU: vector field, exact for a linear velocity" begin
    mesh = quad
    mdotf = lustlu_flux(mesh)
    interior = lustlu_interior(mesh)
    volume = mesh.cells.volume
    G = @SMatrix [0.2e-3 -0.5e-3 0.0; 0.4e-3 0.1e-3 0.0; 0.0 0.0 0.0]
    U = VectorField(mesh)
    gradU = TensorField(mesh)
    for i ∈ eachindex(mesh.cells)
        U.x[i], U.y[i], U.z[i] = G*mesh.cells.centre[i] + SVector(1.0, 2.0, 0.0)
        gradU[i] = G
    end
    exact = G*u
    eqn = (Divergence{LUSTLU}(mdotf, U, gradU) == Source(VectorField(mesh))) → VectorEquation(U, ())
    discretise!(eqn, U, config)
    (; A0, bx, by) = eqn.equation
    @test isapprox(lustlu_residual(A0, U.x.values, bx)[interior], exact[1] .* volume[interior]; rtol=1e-10)
    @test isapprox(lustlu_residual(A0, U.y.values, by)[interior], exact[2] .* volume[interior]; rtol=1e-10)
end

@testset "LUSTLU: matrix and boundary rows are those of LUST, zero gradient gives LUST" begin
    mesh = quad
    mdotf = lustlu_flux(mesh)
    BCs = assign(region=mesh, (U = [
        Dirichlet(:inlet, [1.0, 0.0, 0.0]), Wall(:bottom, [0.0, 0.0, 0.0]),
        Zerogradient(:outlet), Symmetry(:top)],))
    U = VectorField(mesh)
    U.x.values .= [sin(c[1]/200) for c in mesh.cells.centre]
    U.y.values .= [cos(c[2]/300) for c in mesh.cells.centre]
    gradU = TensorField(mesh) # zero
    assembled = map((LUST, LUSTLU)) do scheme
        eqn = (Divergence{scheme}(mdotf, U, gradU) == Source(VectorField(mesh))) → VectorEquation(U, BCs.U)
        discretise!(eqn, U, config)
        out = map((XDir(), YDir())) do dir
            update_equation!(eqn, config)
            apply_boundary_conditions!(eqn, BCs.U, dir, 0.0, config)
            copy(get_sparse_fields(eqn.equation.A)[1]), copy(dir isa XDir ? eqn.equation.bx : eqn.equation.by)
        end
        out
    end
    for d in 1:2
        @test assembled[1][d][1] ≈ assembled[2][d][1] # nzval
        @test assembled[1][d][2] ≈ assembled[2][d][2] # rhs
    end
end

@testset "LUSTLU traits and restrictions" begin
    @test XCALibre.Discretise.is_second_order(LUSTLU)
    @test !XCALibre.Discretise.is_second_order(LUST)
    mesh = quad
    phi = ScalarField(mesh)
    term = Divergence{LUSTLU}(lustlu_flux(mesh), phi, VectorField(mesh))
    @test XCALibre.Discretise.has_face_source(typeof(term))
    termLUST = Divergence{LUST}(lustlu_flux(mesh), phi)
    @test !XCALibre.Discretise.has_face_source(typeof(termLUST))
    @test XCALibre.Discretise.convected_face_value(Divergence{LUST}(), 1.0, 3.0, 0.5, 1.0) == 1.75
end
