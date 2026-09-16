# Shared streamtube case. The exact potential-flow solution for a uniform inlet is U ≡ [1,0,0].
function potential_flow_case()
    side_patches = [:top, :side1, :side2, :bottom]
    mesh = UNV3D_mesh(
        joinpath(
            pkgdir(XCALibre), "examples", "0_GRIDS",
            "3d_streamtube_1.0x0.1x0.1_0.08mm.unv",
        ),
        scale=1.0,
    )
    model = Physics(
        time=Steady(),
        fluid=Fluid{Incompressible}(nu=1e-3),
        turbulence=RANS{Laminar}(),
        energy=Energy{Isothermal}(),
        domain=mesh,
    )
    boundaries = assign(
        region=mesh,
        (
            U=[
                Dirichlet(:inlet, [1.0, 0.0, 0.0]),
                Zerogradient(:outlet),
                Slip.(side_patches)...,
            ],
            p=[
                Zerogradient(:inlet),
                Dirichlet(:outlet, 0.0),
                Slip.(side_patches)...,
            ],
        ),
    )
    schemes = (U=Schemes(), p=Schemes(gradient=Gauss))
    solvers = (
        U=SolverSetup(
            solver=Bicgstab(), preconditioner=Jacobi(),
            convergence=1e-8, relax=1.0,
        ),
        p=SolverSetup(
            solver=Cg(), preconditioner=Jacobi(),
            convergence=1e-8, relax=1.0, rtol=1e-10, atol=1e-14,
        ),
    )
    config = Configuration(
        solvers=solvers,
        schemes=schemes,
        runtime=Runtime(iterations=1, time_step=1, write_interval=-1),
        hardware=Hardware(backend=CPU(), workgroup=256),
        boundaries=boundaries,
    )
    return model, config, mesh, side_patches
end

@testset "potential flow closes non-orthogonal continuity" begin
    model, config, mesh, side_patches = potential_flow_case()

    initialise!(model.momentum.U, [1.0, 0.0, 0.0])
    model.momentum.U.y.values[300] = 3.0
    result = potential_flow!(model, config; ncorrectors=10)

    divergence = ScalarField(mesh)
    div!(divergence, result.flux, config)
    volumes = getproperty.(mesh.cells, :volume)
    continuity_error = sum(abs.(divergence.values).*volumes)/sum(volumes)

    @test result.residual < 1e-8
    @test continuity_error < 1e-9
    @test all(isfinite, model.momentum.U.x.values)
    @test all(isfinite, model.momentum.U.y.values)
    @test all(isfinite, model.momentum.U.z.values)

    # reconstruct! zeroes any cell it finds rank deficient, which the assertions above cannot
    # see: they read Uf, which the boundary conditions overwrite, and the flux rather than U.
    @test !any(iszero, model.momentum.U.x.values)
    @test mean(model.momentum.U.x.values) ≈ 1.0 atol=1e-2

    inlet = only(filter(boundary -> boundary.name == :inlet, mesh.boundaries))
    for fID in inlet.IDs_range
        @test model.momentum.Uf[fID] ≈ SVector(1.0, 0.0, 0.0)
    end
    for name in side_patches
        patch = only(filter(boundary -> boundary.name == name, mesh.boundaries))
        for fID in patch.IDs_range
            @test dot(model.momentum.Uf[fID], mesh.faces[fID].normal) ≈ 0.0 atol=1e-12
        end
    end
end

# A field that is already the potential-flow solution must come back unchanged. This is the only
# check on the reconstructed U that is exact rather than a tolerance on a perturbed field.
@testset "potential flow preserves the exact solution" begin
    model, config, _, _ = potential_flow_case()

    initialise!(model.momentum.U, [1.0, 0.0, 0.0])
    result = potential_flow!(model, config; ncorrectors=10)

    @test result.residual < 1e-8
    @test maximum(abs, model.momentum.U.x.values .- 1.0) < 1e-10
    @test maximum(abs, model.momentum.U.y.values) < 1e-10
    @test maximum(abs, model.momentum.U.z.values) < 1e-10
end
