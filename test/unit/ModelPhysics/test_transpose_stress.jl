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
    hardware = Hardware(backend=CPU(), workgroup=1024)
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
        TS.transpose_stress!(source, nu, gradU, config.boundaries.U, zero(TF), config)

        @test eltype(source.x.values) === TF
        tol = TF === Float64 ? 1e-9 : 2e-3
        @test maximum(abs.(source.x.values .- c*ts_a)) < tol
        @test maximum(abs.(source.y.values .- c*ts_b)) < tol
        @test maximum(abs.(source.z.values)) < tol

        # constant viscosity: the term vanishes for a divergence-free field
        U, gradU, nu = stress_fields(mesh, TF, zero(TF))
        TS.transpose_stress!(source, nu, gradU, config.boundaries.U, zero(TF), config)
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

@testset "Explicit viscous stress with a cell viscosity (interpolated product)" begin
    # With ν = 1 + c x as a cell field the face product ν dev2((∇U)ᵀ) interpolated with the
    # mesh weights is still exact for constant ∇U on this uniform mesh, so the source is again
    # c (G_xx, G_xy, G_xz); boundary faces use the face viscosity
    for TF ∈ (Float64, Float32)
        mesh, config = stress_setup(TF)
        c = TF(2.5)
        U, gradU, nu = stress_fields(mesh, TF, c)
        nuc = ScalarField(mesh)
        for i ∈ eachindex(mesh.cells)
            nuc.values[i] = 1 + c*mesh.cells[i].centre[1]
        end
        source = VectorField(mesh)
        cell_mueff = TS.cell_nueff(nuc, nothing) # laminar: ν alone
        @test cell_mueff[7] == nuc.values[7]
        TS.transpose_stress!(source, nu, gradU, config.boundaries.U, zero(TF), config; cell_mueff)
        @test eltype(source.x.values) === TF
        tol = TF === Float64 ? 1e-9 : 2e-3
        # boundary cells: the boundary-face gradient takes the (uniform) BC values, not the linear field
        interior = setdiff(eachindex(mesh.cells), mesh.boundary_cellsID)
        @test maximum(abs.(source.x.values[interior] .- c*ts_a)) < tol
        @test maximum(abs.(source.y.values[interior] .- c*ts_b)) < tol
    end

    # Internal faces carry w Γ[P] T[P] + (1 - w) Γ[N] T[N] (the divergence of the face-interpolated product),
    # not Γf (w T[P] + (1 - w) T[N]): checked against a direct sum with a jumping Γ
    mesh, config = stress_setup(Float64; grid="flatplate_2D_lowRe.unv")
    ncells = length(mesh.cells)
    Γc = ScalarField(mesh)
    Γc.values .= [isodd(i) ? 1.0 : 50.0 for i ∈ 1:ncells]
    tensor = TensorField(mesh)
    for i ∈ 1:ncells
        x, y, _ = mesh.cells[i].centre
        tensor[i] = @SMatrix [1+x 2y 0; -y 0.5+x*y 0; 0 0 0]
    end
    Γf = FaceScalarField(mesh)
    Γf.values .= 1
    phi = VectorField(mesh)
    div!(phi, Γf, tensor, (), config; Γc=Γc)
    expected = zeros(SVector{3,Float64}, ncells)
    nb = length(mesh.boundary_cellsID)
    for fID ∈ nb+1:length(mesh.faces)
        f = mesh.faces[fID]
        P, N = f.ownerCells
        flux = (f.weight*Γc[P]*tensor[P] + (1 - f.weight)*Γc[N]*tensor[N])*f.normal*f.area
        expected[P] += flux/mesh.cells[P].volume
        expected[N] -= flux/mesh.cells[N].volume
    end
    @test maximum(norm(phi[i] - expected[i]) for i ∈ 1:ncells) < 1e-9*maximum(norm, expected)

    # Compressible/multiphase cell μ_eff = ρ(ν + ν_t)
    rho = ScalarField(mesh); rho.values .= 2.0
    nut = ScalarField(mesh); nut.values .= 0.25
    μ = TS.cell_mueff(rho, ConstantScalar(0.5), (; nut))
    @test μ[3] == 2.0*(0.5 + 0.25)
    @test TS.cell_mueff(rho, FaceScalarField(mesh), (; nut)) === nothing
end

@testset "Boundary-face gradient of DirichletFunction, RotatingWall and Outlet" begin
    # Every fixed-value condition gets (U_b - U_c)/δ, a DirichletFunction evaluated at the face and `time`:
    # one returning Dirichlet or Wall values reproduces them to the bit in the transpose stress and the
    # viscous work (before, it kept the owner-cell gradient and differed by O(1)); Outlet is zero gradient.
    for TF ∈ (Float64, Float32)
        mesh = UNV2D_mesh(joinpath(ts_grids, "quad40.unv"), scale=0.001, float_type=TF)
        n = [b.name for b ∈ mesh.boundaries] # inlet, outlet, bottom, top
        bcs(U) = assign(region=mesh, (U=U,)).U
        config = stress_setup(TF)[2]
        reference = bcs([Dirichlet(n[1], [1.0, 0.0, 0.0]), Zerogradient(n[2]),
            Wall(n[3], [0.0, 0.0, 0.0]), Wall(n[4], [0.0, 0.0, 0.0])])
        functions = bcs([DirichletFunction(n[1], (x, t, i) -> SVector{3}(1.0, 0.0, 0.0)), Outlet(n[2]),
            DirichletFunction(n[3], (x, t, i) -> SVector{3}(0.0, 0.0, 0.0)),
            RotatingWall(n[4]; centre=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0], rpm=0.0)])

        U = VectorField(mesh)
        for i ∈ eachindex(mesh.cells)
            x, y, _ = mesh.cells[i].centre
            U.x.values[i] = sin(4x)*cos(6y) + 3y
            U.y.values[i] = x*y
        end
        nuc = ScalarField(mesh); nuc.values .= [1 + 3c.centre[1] + 5c.centre[2]^2 for c ∈ mesh.cells]
        nuf = FaceScalarField(mesh); nuf.values .= [1 + 3f.centre[1] + 5f.centre[2]^2 for f ∈ mesh.faces]
        mdotf = FaceScalarField(mesh); mdotf.values .= [f.normal[1] for f ∈ mesh.faces]
        function terms(BCs)
            gradU = Grad{Gauss}(U); Uf = FaceVectorField(mesh)
            grad!(gradU, Uf, U, BCs, zero(TF), config)
            source = VectorField(mesh)
            TS.transpose_stress!(source, nuf, gradU, BCs, zero(TF), config; cell_mueff=TS.cell_nueff(nuc, nothing))
            work = ScalarField(mesh)
            XCALibre.ModelPhysics.viscous_work!(work, nuf, nuc, U, Uf, gradU, mdotf, BCs, zero(TF), config)
            source, work
        end
        sA, wA = terms(reference)
        sB, wB = terms(functions)
        @test sB.x.values == sA.x.values && sB.y.values == sA.y.values && sB.z.values == sA.z.values
        @test wB.values == wA.values
        @test eltype(sB.x.values) === TF && eltype(wB.values) === TF
    end

    # A sheared, time-dependent inlet profile, against the face flux computed by hand: with a uniform cell
    # gradient G only the inlet faces change between two inlet conditions, so the difference of the sources
    # is the difference of their fluxes ν_f (M_b' - ⅔ tr(M_b) I) n A/V with M_b = G + (s - G n) n'.
    for TF ∈ (Float64, Float32)
        mesh, config = stress_setup(TF)
        n = [b.name for b ∈ mesh.boundaries]
        profile(x, t, i) = SVector{3}((1 + t)*4x[2]*(1 - x[2]), 0.1*t*x[2], 0.0)
        others = (Zerogradient(n[2]), Wall(n[3], [0.0, 0.0, 0.0]), Wall(n[4], [0.0, 0.0, 0.0]))
        sheared = assign(region=mesh, (U=[DirichletFunction(n[1], profile), others...],)).U
        fixed = assign(region=mesh, (U=[Dirichlet(n[1], [0.0, 0.0, 0.0]), others...],)).U
        U, gradU, nu = stress_fields(mesh, TF, TF(2.5))
        nuc = ScalarField(mesh); nuc.values .= 1
        G = ts_G(TF)
        tol = TF === Float64 ? 1e-9 : 1e-3
        for t ∈ (TF(0), TF(0.75))
            s1, s0 = VectorField(mesh), VectorField(mesh)
            TS.transpose_stress!(s1, nu, gradU, sheared, t, config; cell_mueff=TS.cell_nueff(nuc, nothing))
            TS.transpose_stress!(s0, nu, gradU, fixed, t, config; cell_mueff=TS.cell_nueff(nuc, nothing))
            expected = zeros(SVector{3,TF}, length(mesh.cells))
            for (k, fID) ∈ enumerate(mesh.boundaries[1].IDs_range)
                f = mesh.faces[fID]; cID = f.ownerCells[1]
                for (sign, Ub) ∈ ((1, profile(f.centre, t, k)), (-1, zero(SVector{3,TF})))
                    Mb = G + ((Ub - U[cID])/f.delta - G*f.normal)*f.normal'
                    Tb = Mb' - TF(2)/3*tr(Mb)*I
                    expected[cID] += sign*nu[fID]*(Tb*f.normal)*f.area/mesh.cells[cID].volume
                end
            end
            scale = maximum(norm, expected)
            @test scale > 0
            @test maximum(norm(s1[i] - s0[i] - expected[i]) for i ∈ eachindex(mesh.cells)) < tol*scale
        end
    end
end

@testset "Periodic faces: conservative transpose stress, tensor divergence and viscous work" begin
    # Periodic faces take the internal-face flux of the pair (interpolated product of the two cells), so
    # the contributions of the two faces of every pair cancel and the periodic patches add nothing to the
    # domain integral, although they change the cell values; with the owner-cell value they did not cancel.
    for TF ∈ (Float64, Float32)
        mesh = UNV2D_mesh(joinpath(ts_grids, "quad40.unv"), scale=0.001, float_type=TF)
        n = [b.name for b ∈ mesh.boundaries]
        periodic = construct_periodic(mesh, CPU(), n[3], n[4])
        BCs = assign(region=mesh, (U=[Zerogradient(n[1]), Zerogradient(n[2]), periodic...],)).U
        pair = Tuple(bc for bc ∈ BCs if bc isa Union{Periodic,PeriodicParent})
        config = stress_setup(TF)[2]
        U = VectorField(mesh)
        gradU = Grad{Gauss}(U)
        for i ∈ eachindex(mesh.cells)
            x, y, _ = mesh.cells[i].centre
            U.x.values[i] = sin(6y) + x
            U.y.values[i] = x*y
            gradU.result[i] = SMatrix{3,3,TF}(1, y, 0, 6cos(6y), x, 0, 0, 0, 0)
        end
        nuc = ScalarField(mesh); nuc.values .= [1 + 3c.centre[2]^2 for c ∈ mesh.cells]
        nuf = FaceScalarField(mesh); nuf.values .= [1 + 3f.centre[2]^2 for f ∈ mesh.faces]
        V = [c.volume for c ∈ mesh.cells]
        tol = TF === Float64 ? 1e-10 : 1e-4

        s_pair, s_int = VectorField(mesh), VectorField(mesh)
        TS.transpose_stress!(s_pair, nuf, gradU, pair, zero(TF), config; cell_mueff=TS.cell_nueff(nuc, nothing))
        TS.transpose_stress!(s_int, nuf, gradU, (), zero(TF), config; cell_mueff=TS.cell_nueff(nuc, nothing))
        d = [s_pair[i] - s_int[i] for i ∈ eachindex(mesh.cells)]
        @test maximum(norm, d) > 0.1
        @test norm(sum(d .* V)) < tol*maximum(norm, d)*sum(V)

        # generic tensor divergence with a face coefficient only
        tensor = TensorField(mesh)
        for i ∈ eachindex(mesh.cells); tensor[i] = gradU.result[i]; end
        p_pair, p_int = VectorField(mesh), VectorField(mesh)
        div!(p_pair, nuf, tensor, pair, config)
        div!(p_int, nuf, tensor, (), config)
        d = [p_pair[i] - p_int[i] for i ∈ eachindex(mesh.cells)]
        @test maximum(norm, d) > 0.1
        @test norm(sum(d .* V)) < tol*maximum(norm, d)*sum(V)

        # viscous work of the pair: equal and opposite forces with the same upwind velocity
        Uf = FaceVectorField(mesh)
        mdotf = FaceScalarField(mesh); mdotf.values .= [f.normal[2] for f ∈ mesh.faces]
        w_pair, w_int = ScalarField(mesh), ScalarField(mesh)
        XCALibre.ModelPhysics.viscous_work!(w_pair, nuf, nuc, U, Uf, gradU, mdotf, pair, zero(TF), config)
        XCALibre.ModelPhysics.viscous_work!(w_int, nuf, nuc, U, Uf, gradU, mdotf, (), zero(TF), config)
        d = w_pair.values .- w_int.values
        @test maximum(abs, d) > 0.1
        @test abs(sum(d .* V)) < tol*maximum(abs, d)*sum(V)
    end
end

@testset "PISO with LES: DirichletFunction walls reproduce Wall to the bit" begin
    # End to end through the solver's call of transpose_stress!: a DirichletFunction returning zero on
    # the walls gives the fields of Wall exactly (before the fix the explicit stress at the wall faces
    # kept the owner-cell gradient and the fields differed by ~1e-5 after 20 steps)
    mesh = UNV2D_mesh(joinpath(ts_grids, "quad40.unv"), scale=0.001)
    n = [b.name for b ∈ mesh.boundaries]
    inflow(x, t, i) = SVector{3}(1.0 + 0.1sin(50t), 0.0, 0.0)
    function piso_fields(walls)
        model = Physics(time=Transient(), fluid=Fluid{Incompressible}(nu=1e-3), turbulence=LES{WALE}(),
            energy=Energy{Isothermal}(), domain=mesh)
        BCs = assign(region=mesh, (
            U = [DirichletFunction(n[1], inflow), Zerogradient(n[2]), walls...],
            p = [Zerogradient(n[1]), Dirichlet(n[2], 0.0), Zerogradient(n[3]), Zerogradient(n[4])],
            nut = [Dirichlet(n[1], 0.0), Zerogradient(n[2]), Dirichlet(n[3], 0.0), Dirichlet(n[4], 0.0)]))
        sch = Schemes(time=CrankNicolson, divergence=LUST, gradient=Gauss)
        setup(solver) = SolverSetup(solver=solver, preconditioner=Jacobi(), convergence=1e-10, relax=1.0,
            rtol=0.0, atol=1e-12)
        config = Configuration(solvers=(U=setup(Bicgstab()), p=setup(Cg())), schemes=(U=sch, p=sch),
            boundaries=BCs, runtime=Runtime(iterations=20, time_step=1e-3, write_interval=-1),
            hardware=Hardware(backend=CPU(), workgroup=1024))
        initialise!(model.momentum.U, [1.0, 0.0, 0.0]); initialise!(model.momentum.p, 0.0)
        initialise!(model.turbulence.nut, 0.0)
        run!(model, config)
        (; U, p) = model.momentum
        U.x.values, U.y.values, p.values
    end
    zero_wall(x, t, i) = SVector{3}(0.0, 0.0, 0.0)
    walls = piso_fields((Wall(n[3], [0.0, 0.0, 0.0]), Wall(n[4], [0.0, 0.0, 0.0])))
    functions = piso_fields((DirichletFunction(n[3], zero_wall), DirichletFunction(n[4], zero_wall)))
    @test all(walls .== functions)
end
