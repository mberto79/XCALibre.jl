using XCALibre
using StaticArrays
using Test

# Turbulence kernels must compute in the float type of the mesh: Float64 literals and
# coefficients promote Float32 arithmetic, so results are no longer Float32 computations.

const MP = XCALibre.ModelPhysics

mesh32 = UNV2D_mesh(joinpath(pkgdir(XCALibre, "examples/0_GRIDS"), "quad40.unv"),
    scale=0.001, float_type=Float32)

@testset "bound! averages neighbours in the field's float type" begin
    (; cells, cell_neighbours) = mesh32
    neighbours(i) = (cell_neighbours[fi] for fi ∈ cells[i].faces_range)
    hardware = Hardware(backend=CPU(), workgroup=64)
    config = (hardware=hardware,)

    # Negative cells have no negative neighbour, so the result does not depend on update order.
    v0 = Float32[1 + (sin(7.3f0*i)^2)*1000 for i ∈ eachindex(cells)]
    negative = falses(length(cells))
    for i ∈ eachindex(cells)
        any(negative[j] for j ∈ neighbours(i)) || (negative[i] = true; v0[i] = -1)
    end

    bounded(T) = map(eachindex(v0)) do i
        v0[i] > 0 && return v0[i]
        s = zero(T)
        n = 0
        for j ∈ neighbours(i); s += max(T(v0[j]), T(eps(Float32))); n += 1; end
        Float32(s/n)
    end
    ref32, ref64 = bounded(Float32), bounded(Float64)
    @test count(negative) > 0
    @test ref32 != ref64 # the check below can tell the two accumulations apart

    phi = ScalarField(mesh32)
    phi.values .= v0
    MP.bound!(phi, config)
    @test eltype(phi.values) === Float32
    @test phi.values == ref32
end

@testset "Wall-function helpers return the input float type" begin
    k, nu, y, cmu, kappa, E = 1f-2, 1f-5, 1f-3, 9f-2, 41f-2, 98f-1
    @test (@inferred MP.y_plus(k, nu, y, cmu)) isa Float32
    @test (@inferred MP.ω_log(k, y, cmu, kappa)) isa Float32
    @test (@inferred MP.nut_wall(nu, 30f0, kappa, E)) isa Float32
    @test MP.y_plus(1e-2, 1e-5, 1e-3, 0.09) == 0.09^0.25*1e-3*sqrt(1e-2)/1e-5
end

@testset "Tensor wrappers keep the gradient's float type" begin
    gradU = [SMatrix{3,3,Float32}(1:9)]
    S = StrainRate(gradU, nothing, nothing, nothing)
    @test eltype(S[1]) === Float32
    @test eltype(Vorticity(nothing, gradU)[1]) === Float32
    @test eltype(Dev(S)[1]) === Float32
end

@testset "Model coefficients take the mesh float type" begin
    for turbulence ∈ (
        RANS{KOmega}(), RANS{KOmegaSST}(walls=(:wall,)), LES{Smagorinsky}(), LES{KEquation}())
        coeffs = turbulence(mesh32).coeffs
        @test all(c -> c isa Float32, values(coeffs))
    end
end
