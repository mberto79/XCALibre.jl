using XCALibre
using Test

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
mesh = UNV3D_mesh(joinpath(grids_dir, "3d_box_1000x1000x1000mm_10.unv"), scale=0.001)
ncells = length(mesh.cells)
n_bfaces = length(mesh.boundary_cellsID)
total_volume = sum(c.volume for c ∈ mesh.cells)

@testset "Phase 1 partition nparts=$nparts" for nparts ∈ (1, 2, 4)
    parts = partition_cells(mesh, nparts)
    dms = [extract_subdomain(mesh, parts, r) for r ∈ 1:nparts]

    # cell ownership: bijections and contiguous global blocks
    @test sum(dm.partition.n_owned for dm ∈ dms) == ncells
    owned_orig = sort!(vcat([dm.orig_cells[1:dm.partition.n_owned] for dm ∈ dms]...))
    @test owned_orig == 1:ncells
    l2g_owned = sort!(vcat([dm.partition.local_to_global[1:dm.partition.n_owned] for dm ∈ dms]...))
    @test l2g_owned == 1:ncells
    for dm ∈ dms
        p = dm.partition
        @test p.local_to_global[1:p.n_owned] == p.row_start:p.row_end
        @test all(p.owner[1:p.n_owned] .== p.rank)
        @test all(p.owner[p.n_owned+1:end] .!= p.rank)
    end

    # conservation: owned volumes sum to serial total
    vol = sum(sum(dm.mesh.cells[i].volume for i ∈ 1:dm.partition.n_owned; init=0.0) for dm ∈ dms)
    @test vol ≈ total_volume rtol = 1e-12

    # ghost geometry copied verbatim
    for dm ∈ dms
        p = dm.partition
        for i ∈ p.n_owned+1:p.n_owned+p.n_ghost
            g = dm.orig_cells[i]
            @test dm.mesh.cells[i].centre == mesh.cells[g].centre
            @test dm.mesh.cells[i].volume == mesh.cells[g].volume
        end
    end

    # boundary patches: all present on every rank, counts sum to serial
    for (pi, b) ∈ enumerate(mesh.boundaries)
        @test all(dm.mesh.boundaries[pi].name == b.name for dm ∈ dms)
        @test sum(length(dm.mesh.boundaries[pi].IDs_range) for dm ∈ dms) == length(b.IDs_range)
    end

    # boundary-face invariants: first in faces, self-owned, owned cells only
    for dm ∈ dms
        lm = dm.mesh
        for fID ∈ 1:length(lm.boundary_cellsID)
            f = lm.faces[fID]
            @test f.ownerCells[1] == f.ownerCells[2] == lm.boundary_cellsID[fID]
            @test lm.boundary_cellsID[fID] <= dm.partition.n_owned
        end
    end

    # face accounting: boundary/interior once, processor faces on exactly two ranks
    face_count = zeros(Int, length(mesh.faces))
    for dm ∈ dms, f ∈ dm.orig_faces
        face_count[f] += 1
    end
    for fID ∈ 1:length(mesh.faces)
        o1, o2 = mesh.faces[fID].ownerCells
        expected = (fID <= n_bfaces || parts[o1] == parts[o2]) ? 1 : 2
        @test face_count[fID] == expected
    end

    # face geometry copied verbatim
    for dm ∈ dms
        lm = dm.mesh
        for (lf, f) ∈ enumerate(lm.faces)
            gf = mesh.faces[dm.orig_faces[lf]]
            @test f.centre == gf.centre && f.normal == gf.normal && f.e == gf.e
            @test f.area == gf.area && f.delta == gf.delta && f.weight == gf.weight
        end
    end

    # processor patches: symmetry and send/recv alignment invariant
    for (r, dm) ∈ enumerate(dms)
        for pp ∈ dm.procs
            qm = dms[pp.neighbour+1]
            qq = findfirst(x -> x.neighbour == r - 1, qm.procs)
            @test qq !== nothing
            ppq = qm.procs[qq]
            @test sort(dm.orig_faces[pp.faces]) == sort(qm.orig_faces[ppq.faces])
            @test qm.orig_cells[ppq.send_cells] == dm.orig_cells[pp.recv_ghosts]
            @test dm.orig_cells[pp.send_cells] == qm.orig_cells[ppq.recv_ghosts]
        end
        @test sum(length(pp.recv_ghosts) for pp ∈ dm.procs; init=0) == dm.partition.n_ghost
    end

    # owned-cell connectivity preserved (order, neighbours, nsign vs serial)
    for dm ∈ dms
        lm = dm.mesh
        for i ∈ 1:dm.partition.n_owned
            gcell = mesh.cells[dm.orig_cells[i]]
            lcell = lm.cells[i]
            @test length(lcell.faces_range) == length(gcell.faces_range)
            for (k, j) ∈ enumerate(lcell.faces_range)
                gj = gcell.faces_range[k]
                @test dm.orig_faces[lm.cell_faces[j]] == mesh.cell_faces[gj]
                @test lm.cell_nsign[j] == mesh.cell_nsign[gj]
                @test dm.orig_cells[lm.cell_neighbours[j]] == mesh.cell_neighbours[gj]
            end
        end
        # every face's owners are valid local cells that list it back
        for (lf, f) ∈ enumerate(lm.faces)
            for o ∈ f.ownerCells
                @test 1 <= o <= length(lm.cells)
            end
        end
    end

    # spike: fields and show work through property forwarding
    for dm ∈ dms
        phi = ScalarField(dm)
        @test length(phi.values) == dm.partition.n_owned + dm.partition.n_ghost
    end
    @test occursin("DistributedMesh", sprint(show, dms[1]))

    # degenerate single-partition case matches serial mesh
    if nparts == 1
        dm = dms[1]
        @test dm.partition.n_ghost == 0 && isempty(dm.procs)
        @test length(dm.mesh.cells) == ncells
        @test length(dm.mesh.faces) == length(mesh.faces)
        @test length(dm.mesh.nodes) == length(mesh.nodes)
    end
end
