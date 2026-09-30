using XCALibre
using LinearAlgebra
using StaticArrays
using Test

# `bound!` and the `MFaceBased` limiter update cell values that neighbouring work items also
# read. Both must give the same result for any workgroup size and thread count, equal to a
# plain serial reference loop.

config_for(workgroup) = (hardware = Hardware(backend=CPU(), workgroup=workgroup),)

@testset "$grid" for grid ∈ ("quad40.unv", "trig40.unv")
    mesh = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), grid), scale=0.001)
    (; cells, cell_faces, cell_neighbours, faces) = mesh
    ncells = length(cells)
    workgroups = (1, 3, 16, ncells)
    repeats = 10

    # oscillating field with patches of adjacent negative cells
    xs = [c.centre[1] for c ∈ cells]
    ys = [c.centre[2] for c ∈ cells]
    phi0 = @. sin(9000xs)*cos(7000ys) + 0.2

    @testset "bound! is a Jacobi update" begin
        mzero = eps(Float64)
        function bound_cell(v, i)
            nb = [cell_neighbours[fi] for fi ∈ cells[i].faces_range]
            average = sum(max(v[j], mzero) for j ∈ nb)/length(nb)
            signbit(v[i]) ? max(average, mzero) : max(v[i], mzero)
        end
        reference = [bound_cell(phi0, i) for i ∈ 1:ncells]

        # non-vacuous: adjacent negative cells exist, so an in-place sweep differs from it
        @test any(phi0[i] < 0 && phi0[cell_neighbours[fi]] < 0
            for i ∈ 1:ncells for fi ∈ cells[i].faces_range)
        inplace = copy(phi0)
        for i ∈ 1:ncells
            inplace[i] = bound_cell(inplace, i)
        end
        @test inplace != reference

        field = ScalarField(mesh)
        work = similar(field.values)
        @test all(1:repeats) do _
            all(workgroups) do workgroup
                field.values .= phi0
                fill!(work, NaN)
                XCALibre.ModelPhysics.bound!(field, work, config_for(workgroup))
                field.values == reference
            end
        end
        field.values .= phi0
        XCALibre.ModelPhysics.bound!(field, config_for(3))
        @test field.values == reference
    end

    @testset "MFaceBased limiter is independent of the schedule" begin
        grad0 = [SVector(4000*sin(3000xs[i]), 3000*cos(5000ys[i]), 0.0) for i ∈ 1:ncells]
        reference = copy(grad0)
        for i ∈ 1:ncells, fi ∈ cells[i].faces_range
            FP, FN = phi0[i], phi0[cell_neighbours[fi]]
            δmax, δmin = max(FP, FN) - FP, min(FP, FN) - FP
            d = faces[cell_faces[fi]].centre - cells[i].centre
            fval = reference[i]⋅d
            if fval > δmax
                reference[i] += d*(δmax - fval)/(d⋅d)
            elseif fval < δmin
                reference[i] += d*(δmin - fval)/(d⋅d)
            end
        end
        @test count(reference .!= grad0) > ncells ÷ 10 # the limiter is active

        phi = ScalarField(mesh)
        phi.values .= phi0
        ∇phi = Grad{Gauss}(phi)
        function limited(workgroup)
            ∇phi.result.x.values .= getindex.(grad0, 1)
            ∇phi.result.y.values .= getindex.(grad0, 2)
            ∇phi.result.z.values .= getindex.(grad0, 3)
            limit_gradient!(MFaceBased(mesh), ∇phi, phi, config_for(workgroup))
            [∇phi[i] for i ∈ 1:ncells]
        end
        serial = limited(ncells)
        scale = maximum(norm, grad0)
        @test all(norm(serial[i] - reference[i]) <= 1e-12*scale for i ∈ 1:ncells)
        # bitwise identical for every schedule
        @test all(limited(workgroup) == serial for _ ∈ 1:repeats for workgroup ∈ workgroups)
    end
end
