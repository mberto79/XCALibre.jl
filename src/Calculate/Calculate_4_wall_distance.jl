export wall_distance!
export AbstractWallDistance, MeshWave, Poisson

abstract type AbstractWallDistance end

"""
    MeshWave <: AbstractWallDistance

Geometric wall distance (default). Each cell carries the location of its nearest wall point
and the Euclidean distance to it. Cells touching a wall node start from the exact closest
point on the wall faces through their nodes; sweeps over the internal faces then let a cell
take a neighbour's wall point whenever that point is closer, until no cell changes. Each sweep only reads face
neighbours, so it runs unchanged on distributed meshes (one halo exchange per sweep) and the
result does not depend on the partitioning.

### Example
- `RANS{KOmegaSST}(walls=(:wall,), wall_distance=MeshWave())`
"""
struct MeshWave <: AbstractWallDistance end

"""
    Poisson(; iterations=1000) <: AbstractWallDistance

Wall distance from the solution φ of ∇²φ = -1 (φ = 0 on walls), y = -|∇φ| + √(|∇φ|² + 2φ).
It needs `solvers.y` and `schemes.y` in the configuration. Exact for a single plane wall, it
loses accuracy away from walls, e.g. upstream of a leading edge.

### Example
- `RANS{KOmegaSST}(walls=(:wall,), wall_distance=Poisson(iterations=2000))`
"""
Base.@kwdef struct Poisson <: AbstractWallDistance
    iterations::Int = 1000
end

"""
    wall_distance!(model, walls, config; method=MeshWave())

Compute the wall distance `model.turbulence.y` to the patches named in `walls` with `method`
(`MeshWave()` or `Poisson()`). Returns a new `Configuration` whose boundaries include the
conditions of `y`.
"""
wall_distance!(model, walls, config; method::AbstractWallDistance=MeshWave()) =
    wall_distance!(method, model, walls, config)

# config with the boundary conditions of y attached (used downstream, e.g. by the output writer)
function wall_distance_config(mesh, walls, config)
    (; solvers, schemes, runtime, hardware, postprocess) = config
    BCs = wall_distance_BCs(mesh, walls, config)
    wallBCs = assign(region=mesh, (y = [BCs...],))
    updated_boundaries = (; config.boundaries..., y = wallBCs.y)
    Configuration(
        schemes=schemes, solvers=solvers, runtime=runtime,
        hardware=hardware, postprocess=postprocess, boundaries=updated_boundaries
        )
end

function wall_distance!(method::Poisson, model, walls, config)
    (; iterations) = method
    mesh = model.domain
    is_report_rank(mesh) && @info "Calculating wall distance (Poisson)..."
    (; y) = model.turbulence
    new_config = wall_distance_config(mesh, walls, config)
    (; solvers, schemes, boundaries) = new_config
    
    phi = ScalarField(mesh)

    phi_eqn = (
        -Laplacian{schemes.y.laplacian}(ConstantScalar(1.0), phi) 
        == 
        Source(ConstantScalar(1.0))
    # ) → ScalarEquation(phi, wallBCs.y) # wallBCs are used when setting up BCs for the user
    ) → ScalarEquation(phi, boundaries.y)

    # @reset phi_eqn.preconditioner = set_preconditioner(
    #     solvers.y.preconditioner, phi_eqn, wallBCs.y, config)

    # Krylov preconditioner/workspace are serial-only (distributed solves through PETSc PCs)
    if !is_distributed_mesh(mesh)
        @reset phi_eqn.preconditioner = set_preconditioner(solvers.y.preconditioner, phi_eqn)
        @reset phi_eqn.solver = _workspace(solvers.y.solver, _b(phi_eqn), _index_type(_A(phi_eqn)))
    end
    distributed = is_distributed_mesh(mesh)
    phi_deqn = wrap_eqn(phi_eqn, mesh, solvers.y, config; label="y")
    phi_eqn = unwrap_eqn(phi_deqn)

    TF = _get_float(mesh)

    phiGrad = Grad{schemes.y.gradient}(phi)
    phif = FaceScalarField(mesh)
    # grad!(phiGrad, phif, phi, wallBCs.y, zero(TF), config) # assuming time=0
    grad!(phiGrad, phif, phi, boundaries.y, zero(TF), config) # assuming time=0

    n_cells = length(mesh.cells)
    prev = similar(phi.values)
    # R_phi = ones(TF, iterations)

    for iteration ∈ 1:iterations
        @. prev = phi.values
        discretise!(phi_eqn, phi, config)
        # apply_boundary_conditions!(phi_eqn, wallBCs.y, nothing, 0.0, config) # wrong BCs!!
        # apply_boundary_conditions!(phi_eqn, wallBCs.y, nothing, 0.0, config)
        apply_boundary_conditions!(phi_eqn, boundaries.y, nothing, 0.0, config)

        distributed || update_preconditioner!(phi_eqn.preconditioner, mesh, config)
        # implicit_relaxation!(phi_eqn, phi.values, solvers.y.relax, nothing, config)
        phi_res = solve_system!(phi_deqn, solvers.y, phi, nothing, config)
        explicit_relaxation!(phi, prev, solvers.y.relax, config)

        if phi_res < solvers.y.convergence 
            is_report_rank(mesh) && @info "Wall distance converged in $iteration iterations ($phi_res)"
            break
        elseif iteration == iterations
            is_report_rank(mesh) && @info "Wall distance calculation did not converge ($phi_res)"
        end
    end
    
    # explicit_relaxation! overwrote ghost phi after the last solve synced it
    sync!(phi, mesh, config)
    # grad!(phiGrad, phif, phi, wallBCs.y, zero(TF), config) # assuming time=0
    grad!(phiGrad, phif, phi, boundaries.y, zero(TF), config) # assuming time=0
    normal_distance!(y, phi, phiGrad, config)
    # y.values .= phi.values

    BCs = wallBCs = phi = phi_eqn = phiGrad = phif = prev = nothing
    GC.gc()

    new_config
end

function normal_distance!(y, phi, phiGrad, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    ndrange = length(phi.values)
    kernel! = _sized(_normal_distance!, backend, workgroup, ndrange)
    kernel!(y, phi, phiGrad)
    KernelAbstractions.synchronize(backend)
end

function wall_distance_BCs(mesh, walls, config)
    boundaries_cpu = get_boundaries(mesh.boundaries)
    boundary_names = map(boundary -> boundary.name, boundaries_cpu)
    wall_names = collect(Symbol.(walls))
    missing_walls = setdiff(wall_names, boundary_names)
    isempty(missing_walls) || error("Wall distance patches not found in mesh: $(Tuple(missing_walls)). Available patches: $(Tuple(boundary_names))")

    empty_names = empty_boundary_names(boundaries_cpu, config.boundaries)
    matched_walls = intersect(wall_names, boundary_names)
    isempty(matched_walls) && error("Wall distance needs at least one wall patch")
    warn_omitted_velocity_walls(boundaries_cpu, config.boundaries, matched_walls)

    BCs = []
    for boundary ∈ boundaries_cpu
        boundary_name = boundary.name
        if boundary_name ∈ matched_walls
            push!(BCs, Dirichlet(boundary_name, 0.0))
        elseif boundary_name ∈ empty_names
            push!(BCs, Empty(boundary_name))
        else
            push!(BCs, Extrapolated(boundary_name))
        end
    end
    BCs
end

function empty_boundary_names(boundaries_cpu, boundaries)
    empty_names = Symbol[]
    for field_BCs ∈ boundaries
        for BC ∈ field_BCs
            if typeof(BC) <: Empty
                push!(empty_names, boundaries_cpu[BC.ID].name)
            end
        end
    end
    unique(empty_names)
end

function warn_omitted_velocity_walls(boundaries_cpu, boundaries, wall_names)
    hasproperty(boundaries, :U) || return nothing

    omitted = Symbol[]
    for BC ∈ boundaries.U
        if typeof(BC) <: Union{Wall,RotatingWall}
            name = boundaries_cpu[BC.ID].name
            name ∈ wall_names || push!(omitted, name)
        end
    end

    isempty(omitted) || @warn "Velocity wall patches omitted from wall distance calculation: $(Tuple(unique(omitted))). Add them to turbulence walls if they are physical walls."
    nothing
end

@kernel function _normal_distance!(y, phi, phiGrad)
    i = @index(Global)

    gradMag_raw = norm(phiGrad.result[i])
    gradMag = isfinite(gradMag_raw) ? gradMag_raw : zero(gradMag_raw)
    radicand_raw = gradMag^2 + 2*phi.values[i]
    radicand = isfinite(radicand_raw) ? max(radicand_raw, zero(radicand_raw)) : zero(radicand_raw)
    y.values[i] = max(-gradMag + sqrt(radicand), zero(gradMag))
end

function wall_distance!(::MeshWave, model, walls, config)
    mesh = model.domain
    is_report_rank(mesh) && @info "Calculating wall distance (MeshWave)..."
    (; y) = model.turbulence
    new_config = wall_distance_config(mesh, walls, config)

    origin = VectorField(mesh)
    y_new = ScalarField(mesh)
    origin_new = VectorField(mesh)
    seed_wall_points!(y, origin, mesh, walls)
    sync!((y, origin), mesh, config)

    sweeps = 0
    changed = true
    while changed
        sweeps += 1
        wall_point_sweep!(y_new, origin_new, y, origin, config)
        sync!((y_new, origin_new), mesh, config)
        changed = global_any(!same_values(y, y_new) || !same_values(origin, origin_new), mesh)
        copyto!(y.values, y_new.values)
        copyto!(origin.x.values, origin_new.x.values)
        copyto!(origin.y.values, origin_new.y.values)
        copyto!(origin.z.values, origin_new.z.values)
    end
    TF = _get_float(mesh)
    global_any(any(==(typemax(TF)), y.values), mesh) &&
        error("MeshWave wall distance: some cells are not connected to a wall patch $(Tuple(walls))")
    is_report_rank(mesh) && @info "Wall distance converged in $sweeps sweeps"
    new_config
end

same_values(a::ScalarField, b::ScalarField) = a.values == b.values
same_values(a::VectorField, b::VectorField) =
    a.x.values == b.x.values && a.y.values == b.y.values && a.z.values == b.z.values

# Every cell touching a wall node starts from the exact closest point over the wall faces through
# its nodes; all other cells start unset (y = typemax). On a distributed mesh some of those faces
# belong to another rank: `remote_wall_faces` supplies them, so the seed is the same as on the
# undivided mesh. One pass on the host, copied to the backend.
function seed_wall_points!(y, origin, mesh, walls)
    TF = _get_float(mesh)
    boundaries_cpu = get_boundaries(mesh.boundaries)
    cell_centre = Array(mesh.cell_centre)
    face_centre = Array(mesh.face_centre)
    face_nodes_range = Array(mesh.face_nodes_range)
    face_nodes = Array(mesh.face_nodes)
    node_coords = Array(mesh.node_coords)
    node_cells_range = Array(mesh.node_cells_range)
    node_cells = Array(mesh.node_cells)

    wall_names = collect(Symbol.(walls))
    wall_faces = Int[]
    for boundary ∈ boundaries_cpu
        boundary.name ∈ wall_names && append!(wall_faces, boundary.IDs_range)
    end

    # candidate wall faces (centre, node coordinates) and the local nodes they touch
    centres = eltype(node_coords)[]
    points = Vector{eltype(node_coords)}[]
    node_faces = Dict{Int,Vector{Int}}()
    for fID ∈ wall_faces
        push!(centres, face_centre[fID])
        push!(points, node_coords[face_nodes[face_nodes_range[fID]]])
        for n ∈ face_nodes[face_nodes_range[fID]]
            push!(get!(node_faces, n, Int[]), length(centres))
        end
    end
    remote = remote_wall_faces(mesh, wall_faces)
    if !isempty(remote)
        coord_node = Dict(node_coords[n] => n for n ∈ eachindex(node_coords))
        for (fc, pts) ∈ remote
            push!(centres, fc)
            push!(points, pts)
            for x ∈ pts
                n = get(coord_node, x, 0)
                n == 0 || push!(get!(node_faces, n, Int[]), length(centres))
            end
        end
    end

    n_cells = length(cell_centre)
    yv = fill(typemax(TF), n_cells)
    pv = fill(zero(eltype(cell_centre)), n_cells)
    for (n, faceIDs) ∈ node_faces, cID ∈ node_cells[node_cells_range[n]]
        c = cell_centre[cID]
        for k ∈ faceIDs
            p = closest_point_face(c, centres[k], eachindex(points[k]), points[k])
            d = norm(c - p)
            if closer(d, p, yv[cID], pv[cID])
                yv[cID] = d
                pv[cID] = p
            end
        end
    end
    copyto!(y.values, yv)
    copyto!(origin.x.values, getindex.(pv, 1))
    copyto!(origin.y.values, getindex.(pv, 2))
    copyto!(origin.z.values, getindex.(pv, 3))
    nothing
end

# Wall faces of other ranks that touch nodes shared with this rank, as (centre, node coordinates).
# Serial meshes have none; Distribute extends this for DistributedMesh.
remote_wall_faces(mesh, wall_faces) = Tuple{eltype(mesh.node_coords),Vector{eltype(mesh.node_coords)}}[]

# One Jacobi sweep: each cell keeps the closest of its own and its face neighbours' wall points.
# Reads only the previous state, so the result does not depend on cell order or partitioning.
function wall_point_sweep!(y_new, origin_new, y, origin, config)
    (; cell_centre, cell_faces_range, cell_neighbours) = y.mesh
    xcal_foreach(y_new, config) do i
        c = cell_centre[i]
        d = y[i]
        p = origin[i]
        for j ∈ cell_faces_range[i]
            nID = cell_neighbours[j]
            dn = y[nID]
            dn == typemax(dn) && continue
            pn = origin[nID]
            dc = norm(c - pn)
            if closer(dc, pn, d, p)
                d = dc
                p = pn
            end
        end
        y_new[i] = d
        origin_new[i] = p
    end
end

# strict order on (distance, point): equal distances are resolved by the point coordinates, so
# the chosen wall point is the same whatever order neighbours are visited in
@inline closer(d, p, dbest, pbest) =
    d < dbest || (d == dbest && (p[1] < pbest[1] || (p[1] == pbest[1] &&
        (p[2] < pbest[2] || (p[2] == pbest[2] && p[3] < pbest[3])))))

# Closest point to x on a wall face with centre fc and nodes node_coords[ids]: a segment for 2D
# meshes (two nodes), otherwise the polygon split into triangles about fc
function closest_point_face(x, fc, ids, node_coords)
    length(ids) == 2 && return closest_point_segment(x, node_coords[ids[1]], node_coords[ids[2]])
    best = node_coords[ids[1]]
    dbest = norm(x - best)
    for j ∈ eachindex(ids)
        a = node_coords[ids[j]]
        b = node_coords[ids[mod1(j + 1, length(ids))]]
        q = closest_point_triangle(x, fc, a, b)
        dq = norm(x - q)
        if dq < dbest
            dbest = dq
            best = q
        end
    end
    best
end

function closest_point_segment(x, a, b)
    ab = b - a
    t = clamp(((x - a)⋅ab)/(ab⋅ab), zero(eltype(ab)), one(eltype(ab)))
    a + t*ab
end

# Closest point on triangle (a, b, c) to x (Ericson, Real-Time Collision Detection, §5.1.5)
function closest_point_triangle(x, a, b, c)
    ab = b - a; ac = c - a; ax = x - a
    d1 = ab⋅ax; d2 = ac⋅ax
    (d1 ≤ 0 && d2 ≤ 0) && return a
    bx = x - b
    d3 = ab⋅bx; d4 = ac⋅bx
    (d3 ≥ 0 && d4 ≤ d3) && return b
    vc = d1*d4 - d3*d2
    (vc ≤ 0 && d1 ≥ 0 && d3 ≤ 0) && return a + d1/(d1 - d3)*ab
    cx = x - c
    d5 = ab⋅cx; d6 = ac⋅cx
    (d6 ≥ 0 && d5 ≤ d6) && return c
    vb = d5*d2 - d1*d6
    (vb ≤ 0 && d2 ≥ 0 && d6 ≤ 0) && return a + d2/(d2 - d6)*ac
    va = d3*d6 - d5*d4
    (va ≤ 0 && (d4 - d3) ≥ 0 && (d5 - d6) ≥ 0) && return b + (d4 - d3)/((d4 - d3) + (d5 - d6))*(c - b)
    denom = 1/(va + vb + vc)
    v = vb*denom; w = vc*denom
    a + ab*v + ac*w
end
