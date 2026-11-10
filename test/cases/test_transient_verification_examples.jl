using XCALibre
using Test

# Runs heat_decay.jl of examples/transient_verification and checks the temporal orders it prints,
# so the shipped examples keep running and match their README. The schemes and the PISO shear wave
# are tested in unit/Discretise/test_crank_nicolson.jl. The script runs in its own module, keeping Main clean.

function run_transient_example(file)
    m = Module(Symbol(splitext(file)[1]))
    Base.include(m, pkgdir(XCALibre, "examples", "transient_verification", file))
    return Base.invokelatest(getproperty, m, :results) # bound after this method was compiled
end

@testset "heat_decay.jl" begin
    (; errors, orders) = run_transient_example("heat_decay.jl")
    # finest pair of time steps, against the exact semi-discrete solution
    @test orders["Euler"][end] ≈ 1 atol=0.1
    @test orders["CrankNicolson"][end] ≈ 2 atol=0.1
    @test errors["CrankNicolson"][end] < errors["Euler"][end]/10
    @test errors["CrankNicolson"][end] < errors["CrankNicolson(0.9)"][end] < errors["Euler"][end]
end
