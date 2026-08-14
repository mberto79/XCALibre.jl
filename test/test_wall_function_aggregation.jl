@testset "wall functions aggregate multiply incident faces" begin
    mesh = FOAM3D_mesh(
        joinpath(pkgdir(XCALibre), "test", "grids", "OF_cavity_hex", "polyMesh"),
        scale=0.001,
    )
    model = Physics(
        time=Steady(),
        fluid=Fluid{Incompressible}(nu=1.0e-7),
        turbulence=RANS{KOmega}(),
        energy=Energy{Isothermal}(),
        domain=mesh,
    )
    BCs = assign(
        region=mesh,
        (
            k=[KWallFunction(:walls), KWallFunction(:top)],
            omega=[OmegaWallFunction(:walls), OmegaWallFunction(:top)],
        ),
    )
    config = (
        hardware=Hardware(backend=CPU(), workgroup=64),
    )

    initialise!(model.momentum.U, [2.0, 3.0, 4.0])
    initialise!(model.momentum.Uf, [0.0, 0.0, 0.0])
    initialise!(model.turbulence.k, 0.5)
    initialise!(model.turbulence.nut, 0.05)

    moving_bc = BCs.k[2]
    for fID in moving_bc.IDs_range
        cID = mesh.boundary_cellsID[fID]
        model.momentum.Uf[fID] = model.momentum.U[cID]
    end

    n_cells = length(mesh.cells)
    expected_production = zeros(n_cells)
    expected_omega = zeros(n_cells)
    wall_area = zeros(n_cells)
    incident_faces = zeros(Int, n_cells)
    incident_patches = [Set{Int}() for _ in 1:n_cells]

    for (patch_index, (k_bc, omega_bc)) in enumerate(zip(BCs.k, BCs.omega))
        for fID in k_bc.IDs_range
            cID = mesh.boundary_cellsID[fID]
            face = mesh.faces[fID]
            (; area, delta, normal) = face
            nu = model.fluid.nu[cID]
            k = model.turbulence.k[cID]
            U = model.momentum.U[cID]
            Uw = model.momentum.Uf[fID]

            (; kappa, cmu, E, yPlusLam) = k_bc.value
            yplus = XCALibre.ModelPhysics.y_plus(k, nu, delta, cmu)
            nutw = XCALibre.ModelPhysics.nut_wall(nu, yplus, kappa, E)
            u_star = cmu^0.25*sqrt(k)
            dUdy = u_star/(kappa*delta)
            relative_velocity = U - Uw
            tangential_speed = norm(
                relative_velocity - (relative_velocity⋅normal)*normal,
            )
            production = yplus > yPlusLam ?
                (nu + nutw)*tangential_speed/delta*dUdy : zero(nu)

            (; kappa, beta1, cmu, yPlusLam) = omega_bc.value
            omega_vis = XCALibre.ModelPhysics.ω_vis(nu, delta, beta1)
            omega_log = XCALibre.ModelPhysics.ω_log(k, delta, cmu, kappa)
            omega = yplus > yPlusLam ? omega_log : omega_vis

            expected_production[cID] += area*production
            expected_omega[cID] += area*omega
            wall_area[cID] += area
            incident_faces[cID] += 1
            push!(incident_patches[cID], patch_index)
        end
    end

    wall_cells = findall(>(0.0), wall_area)
    corner_cells = filter(
        cID -> incident_faces[cID] > 1 && length(incident_patches[cID]) > 1,
        wall_cells,
    )
    @test !isempty(corner_cells)
    expected_production[wall_cells] ./= wall_area[wall_cells]
    expected_omega[wall_cells] ./= wall_area[wall_cells]

    production = ScalarField(mesh)
    initialise!(production, -1.0)
    XCALibre.ModelPhysics.correct_production!(
        production,
        BCs.k,
        model,
        nothing,
        config,
    )
    @test production.values[wall_cells] ≈ expected_production[wall_cells]

    stationary_cells = Set(mesh.boundary_cellsID[fID] for fID in BCs.k[1].IDs_range)
    moving_cells = Set(mesh.boundary_cellsID[fID] for fID in moving_bc.IDs_range)
    moving_only_cells = collect(setdiff(moving_cells, stationary_cells))
    @test !isempty(moving_only_cells)
    @test all(iszero, production.values[moving_only_cells])

    first_result = copy(production.values)
    initialise!(production, -1.0)
    XCALibre.ModelPhysics.correct_production!(
        production,
        reverse(BCs.k),
        model,
        nothing,
        config,
    )
    @test production.values ≈ first_result

    omega_source = ScalarField(mesh)
    omega_eqn = (
        Laplacian{Linear}(ConstantScalar(1.0), model.turbulence.omega) ==
        Source(omega_source)
    ) → ScalarEquation(model.turbulence.omega, BCs.omega)
    nzval = _nzval(_A(omega_eqn))
    b = _b(omega_eqn, nothing)
    fill!(nzval, 2.0)
    fill!(b, -1.0)
    XCALibre.ModelPhysics.constrain_equation!(omega_eqn, BCs.omega, model, config)

    rowptr = _rowptr(_A(omega_eqn))
    colval = _colval(_A(omega_eqn))
    @test b[wall_cells] ≈ expected_omega[wall_cells]
    for cID in wall_cells
        row = rowptr[cID]:(rowptr[cID + 1] - 1)
        diagonal = spindex(rowptr, colval, cID, cID)
        @test nzval[diagonal] == 1.0
        @test all(nzi -> nzi == diagonal || iszero(nzval[nzi]), row)
    end
end
