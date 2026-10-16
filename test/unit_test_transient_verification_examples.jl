using XCALibre
using Test

# Runs two scripts of examples/transient_verification and checks the temporal orders they print,
# so the shipped examples keep running and match their README (the schemes themselves are tested
# in unit_test_crank_nicolson.jl). Each script runs in its own module, keeping Main clean.

function run_transient_example(file)
    m = Module(Symbol(splitext(file)[1]))
    Base.include(m, pkgdir(XCALibre, "examples", "transient_verification", file))
    return Base.invokelatest(getproperty, m, :results) # bound after this method was compiled
end

@testset "$file" for file ∈ ("heat_decay.jl", "shear_wave_piso.jl")
    (; errors, orders) = run_transient_example(file)
    # finest pair of time steps; both cases compare with the exact semi-discrete solution
    @test orders["Euler"][end] ≈ 1 atol=0.1
    @test orders["CrankNicolson"][end] ≈ 2 atol=0.1
    @test errors["CrankNicolson"][end] < errors["Euler"][end]/10
end
