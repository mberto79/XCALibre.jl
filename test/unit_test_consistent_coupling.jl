# Uniform Mach 0.84 stream through an empty box (exact steady solution) at relaxation (0.6, 0.3):
# both couplings hold it on the shipped 5x5x5 box; on a 20x10x10 box the default coupling amplifies
# round-off until the run diverges, while the consistent coupling holds it.

# Write a uniform nx×ny×nz hexahedral box [0, lx]×[0, ly]×[0, lz] in the format read by
# `FOAM3D_mesh`, with patches inlet/outlet (x = 0, lx), bottom/top (y) and front/back (z).
function write_hex_box(dir; n=(20, 10, 10), l=(2.0, 1.0, 1.0))
    nx, ny, nz = n
    point(i, j, k) = i + (nx + 1)*(j + (ny + 1)*k) # 0-based
    cell(i, j, k) = i + nx*(j + ny*k)
    xface(i, j, k) = (point(i, j, k), point(i, j+1, k), point(i, j+1, k+1), point(i, j, k+1))
    yface(i, j, k) = (point(i, j, k), point(i, j, k+1), point(i+1, j, k+1), point(i+1, j, k))
    zface(i, j, k) = (point(i, j, k), point(i+1, j, k), point(i+1, j+1, k), point(i, j+1, k))
    faces, owner, neighbour = NTuple{4,Int}[], Int[], Int[]
    for k ∈ 0:nz-1, j ∈ 0:ny-1, i ∈ 0:nx-1 # internal faces, upper-triangular order
        c = cell(i, j, k)
        i < nx - 1 && (push!(faces, xface(i+1, j, k)); push!(owner, c); push!(neighbour, cell(i+1, j, k)))
        j < ny - 1 && (push!(faces, yface(i, j+1, k)); push!(owner, c); push!(neighbour, cell(i, j+1, k)))
        k < nz - 1 && (push!(faces, zface(i, j, k+1)); push!(owner, c); push!(neighbour, cell(i, j, k+1)))
    end
    patches = Tuple{String,Int,Int}[]
    function patch!(name, cells_faces)
        start = length(faces)
        for (f, c) ∈ cells_faces
            push!(faces, f); push!(owner, c)
        end
        push!(patches, (name, length(faces) - start, start))
    end
    patch!("inlet", ((reverse(xface(0, j, k)), cell(0, j, k)) for k ∈ 0:nz-1 for j ∈ 0:ny-1))
    patch!("outlet", ((xface(nx, j, k), cell(nx-1, j, k)) for k ∈ 0:nz-1 for j ∈ 0:ny-1))
    patch!("bottom", ((reverse(yface(i, 0, k)), cell(i, 0, k)) for k ∈ 0:nz-1 for i ∈ 0:nx-1))
    patch!("top", ((yface(i, ny, k), cell(i, ny-1, k)) for k ∈ 0:nz-1 for i ∈ 0:nx-1))
    patch!("front", ((reverse(zface(i, j, 0)), cell(i, j, 0)) for j ∈ 0:ny-1 for i ∈ 0:nx-1))
    patch!("back", ((zface(i, j, nz), cell(i, j, nz-1)) for j ∈ 0:ny-1 for i ∈ 0:nx-1))

    header(class, object) =
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class $class;\n    object $object;\n}\n\n"
    function write_list(name, class, items)
        open(joinpath(dir, name), "w") do io
            print(io, header(class, name), length(items), "\n(\n")
            foreach(item -> println(io, item), items)
            print(io, ")\n")
        end
    end
    points = ["($(l[1]*i/nx) $(l[2]*j/ny) $(l[3]*k/nz))" for k ∈ 0:nz for j ∈ 0:ny for i ∈ 0:nx]
    write_list("points", "vectorField", points)
    write_list("faces", "faceList", ["4($(join(f, ' ')))" for f ∈ faces])
    write_list("owner", "labelList", owner)
    write_list("neighbour", "labelList", neighbour)
    write_list("boundary", "polyBoundaryMesh",
        ["$name\n{\n    type patch;\n    nFaces $nf;\n    startFace $sf;\n}" for (name, nf, sf) ∈ patches])
    return dir
end

function uniform_stream_box(mesh, consistent; iterations=60, inlet=:x_min, outlet=:x_max,
        sides=(:y_min, :y_max, :z_min, :z_max))
    gamma, cp, Pr = 1.4, 1005.0, 0.72
    R = cp*(1 - 1/gamma)
    T_inf, p_inf = 255.56, 80600.0
    U_inf = 0.84*sqrt(gamma*R*T_inf)
    velocity = [U_inf, 0.0, 0.0]

    model = Physics(
        time = Steady(),
        fluid = Fluid{Compressible}(nu=1.5e-5, cp=cp, gamma=gamma, Pr=Pr),
        turbulence = RANS{Laminar}(),
        energy = Energy{SensibleEnthalpy}(Tref=0.0),
        domain = mesh
        )

    enthalpy = Enthalpy(cp=cp, Tref=0.0)
    boundaries = assign(region=mesh, (
        U = [Dirichlet(inlet, velocity), Zerogradient(outlet), [Symmetry(s) for s ∈ sides]...],
        p = [Zerogradient(inlet), Dirichlet(outlet, p_inf), [Symmetry(s) for s ∈ sides]...],
        he = [
            FixedTemperature(inlet, T=T_inf, enthalpy), Zerogradient(outlet),
            [Symmetry(s) for s ∈ sides]...]
        ))

    solver(relax, rtol) = SolverSetup(
        solver=Bicgstab(), preconditioner=Jacobi(), convergence=1e-12, relax=relax,
        rtol=rtol, atol=1e-14)
    solvers = (U=solver(0.6, 1e-1), p=solver(0.3, 1e-6), he=solver(0.3, 1e-1))
    upwind = Schemes(divergence=Upwind, gradient=Gauss)
    config = Configuration(
        solvers=solvers, schemes=(U=upwind, p=upwind, he=upwind), boundaries=boundaries,
        runtime=Runtime(iterations=iterations, write_interval=-1, time_step=1),
        hardware=Hardware(backend=CPU(), workgroup=workgroupsize(mesh)))

    initialise!(model.momentum.U, velocity)
    initialise!(model.momentum.p, p_inf)
    initialise!(model.energy.T, T_inf)

    run!(model, config; progress=false, consistent=consistent)

    (; U, p) = model.momentum
    dU = maximum(abs.(U.x.values .- U_inf))/U_inf
    dp = maximum(abs.(p.values .- p_inf))/p_inf
    return dU, dp
end

unv_box = UNV3D_mesh(pkgdir(XCALibre, "examples/0_GRIDS/3d_box_1000x1000x1000mm_5.unv"), scale=0.001)
for consistent ∈ (false, true)
    dU, dp = uniform_stream_box(unv_box, consistent)
    @test dU < 1e-8
    @test dp < 1e-8
end

hex_box = FOAM3D_mesh(write_hex_box(mktempdir()); scale=1.0)
hex_patches = (inlet=:inlet, outlet=:outlet, sides=(:bottom, :top, :front, :back))
dU, dp = uniform_stream_box(hex_box, true; iterations=40, hex_patches...)
@test dU < 1e-8
@test dp < 1e-8
dU, dp = uniform_stream_box(hex_box, false; iterations=40, hex_patches...) # diverges
@test !(max(dU, dp) < 1e-6) # also true for NaN

import Accessors

# Incompressible laminar backward-facing step: the consistent coupling converges without pressure
# under-relaxation to the solution of the default coupling (relaxation 0.8, 0.2).
include(joinpath(@__DIR__, "distributed", "psimple_case.jl"))

function bfs_simple(consistent, p_relax, iterations)
    model, config = incompressible_case(bfs_mesh(), bfs_bcs; iterations=iterations)
    p_solver = Accessors.setproperties(config.solvers.p, relax=p_relax)
    config = Accessors.@set config.solvers = (U=config.solvers.U, p=p_solver)
    simple!(model, config; consistent=consistent, progress=false)
    return Array(model.momentum.U.x.values), Array(model.momentum.p.values)
end

Ux_ref, p_ref = bfs_simple(false, 0.2, 1500)
Ux, p = bfs_simple(true, 1.0, 800)
@test maximum(abs.(Ux .- Ux_ref)) < 1e-5
@test maximum(abs.(p .- p_ref)) < 1e-5
