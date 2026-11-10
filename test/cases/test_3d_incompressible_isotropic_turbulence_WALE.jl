# Validation of the LES WALE model: decaying homogeneous isotropic turbulence (CBC)
#
# A periodic cube of side L = 0.09·2π m (32³ cells) is initialised with a synthetic,
# divergence-free turbulent velocity field matching the CBC spectrum at tU₀/M = 42, then
# left to decay. The simulated energy spectra at t = 0.28 s and 0.66 s (tU₀/M = 98 and 171)
# are compared with experiment up to the grid Nyquist wavenumber.
#
# - Initial field: random Fourier modes following Saad et al. (2017).
# - Energy spectrum: shell-averaged FFT, based on OpenFOAM's energySpectrum utility.
#
# References
#   Comte-Bellot, G., & Corrsin, S. (1971). Simple Eulerian time correlation of full- and
#     narrow-band velocity signals in grid-generated, 'isotropic' turbulence. Journal of
#     Fluid Mechanics, 48(2), 273–337. https://doi.org/10.1017/S0022112071001599
#   Saad, T., Cline, D., Stoll, R., & Sutherland, J. C. (2017). Scalable tools for generating
#     synthetic isotropic turbulence with arbitrary spectra. AIAA Journal, 55(1), 327–331.
#     https://doi.org/10.2514/1.J055230

using XCALibre
using DelimitedFiles
# Experimental spectrum (CBC at tU₀/M = 42), interpolated in log-log space
# First row skipped: E42 = 0 there, and log(0) = -Inf would break the interpolation
CBC = readdlm(pkgdir(XCALibre, "test/CBC_Data/CBC_data.csv"), ',', Float64; header=true)[1]

κexp  = Float64.(CBC[2:end, 1])
Eexp  = Float64.(CBC[2:end, 2])
logEk = linear_interpolation(log.(κexp), log.(Eexp))

# M random unit vectors, uniformly distributed on the sphere
function random_unit_vectors(rng, M)
    ϕ = 2π .* rand(rng, M)
    t = 2 .* rand(rng, M) .- 1
    θ = acos.(t)
    return SVector.(sin.(θ) .* cos.(ϕ), sin.(θ) .* sin.(ϕ), cos.(θ))
end

function calculate_turbulent_velocity(Xc, N_divisions, L, M, κexp, logEk; rng = MersenneTwister(1234))
    Δx = L / N_divisions
    κ  = range(max(2π / L, κexp[1]), π / Δx; length = M)
    Δκ = κ[2] - κ[1]
    q  = @. sqrt(exp(logEk(log(κ))) * Δκ)            # qₘ = √(E(κₘ)Δκ)

    K = random_unit_vectors(rng, M)                 # wavevector directions

    # modified wavenumbers (collocated grid, central-difference divergence: sin(κΔx)/Δx)
    k = Vector{SVector{3, Float64}}(undef, M)
    for m in 1:M
        k[m] = sin.(κ[m] * K[m] * Δx) ./ Δx
    end

    ζ = random_unit_vectors(rng, M)
    σ = normalize.(ζ .× k)                          # σ = ζ × k / |ζ × k|
    Ψ = π .* rand(rng, M) .- π / 2                  # Ψ ~ U(-π/2, π/2)

    # velocity at every mesh point
    U = Vector{SVector{3, Float64}}(undef, length(Xc))
    for i in eachindex(Xc)
        x = Xc[i]
        u = SVector(0.0, 0.0, 0.0)
        for j in 1:M
            u += 2 * q[j] * cos(κ[j] * (K[j] ⋅ x) + Ψ[j]) * σ[j]
        end
        U[i] = u
    end
    return getindex.(U, 1), getindex.(U, 2), getindex.(U, 3)
end


grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "box32x32x32.unv"

mesh_file = joinpath(grids_dir, grid)
mesh = UNV3D_mesh(mesh_file, scale=0.001)

backend = CPU(); workgroup = 1024;

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

#parameters
L = 0.09  * 2π
M = 5000  # Number of modes in fourier series 
nu = 1.5e-5
#extract all the cell centre coordinates to calculate the initial velocity field \bm{u(x)}
x = [cell.centre for cell ∈ mesh.cells]
Ncells = Int(round((length(mesh.cells))^(1/3),digits=5))
timestep = 0.01


model = Physics(
    time = Transient(),
    fluid = Fluid{Incompressible}(nu=nu),
    turbulence = LES{WALE}(; C=0.5),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )


periodic1 = construct_periodic(mesh, backend, :top, :bottom)
periodic2 = construct_periodic(mesh, backend, :front, :back)
periodic3 = construct_periodic(mesh, backend, :right, :left)

BCs= assign(
    region = mesh_dev,
    (
        U = [
            periodic1...,
            periodic2...,
            periodic3...
        ],
        p = [
            periodic1...,
            periodic2...,
            periodic3...
        ],
        nut = [
            periodic1...,
            periodic2...,
            periodic3...
        ],

    )
)


divergence = XCALibre.Linear 
schemes = (
    # # transient schemes
    U = Schemes(time=Euler, divergence=divergence, gradient=Gauss),
    p = Schemes(gradient=Gauss)

)

solvers = (
    U = SolverSetup(
        solver      = Bicgstab(), 
        preconditioner = Jacobi(),
        convergence = 1e-7,
        atol = 1e-6,
        relax=1

    ),
    p = SolverSetup(
        solver      = Cg(),
        preconditioner = Jacobi(),
        convergence = 1e-7,
        atol = 1e-6,
        relax=1

    )
)
iters1 = round(Int, 0.28/timestep)          # 28 for dt = 0.01 (tU₀/M = 42 → 98)
iters2 = round(Int, (0.66-0.28)/timestep)   # 38 for dt = 0.01 (tU₀/M = 98 → 171)
runtime = Runtime(iterations=iters1, time_step=timestep,  write_interval=-1) 

config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)

GC.gc(true)
#initialise the velocity with the synthetic turbulent velocity field
Uc, Vc, Wc = calculate_turbulent_velocity(x,Ncells,L,M, κexp, logEk)
model.momentum.U.x.values .= adapt(backend, Uc)
model.momentum.U.y.values .= adapt(backend, Vc)
model.momentum.U.z.values .= adapt(backend, Wc)
initialise!(model.momentum.p, 0.0)
initialise!(model.turbulence.nut,2*nu)

#save t=0 velocity and coords 
U = model.momentum.U
Ux = Array(U.x.values)
Uy = Array(U.y.values)
Uz = Array(U.z.values)


V = [[Ux[i], Uy[i], Uz[i]] for i in eachindex(U)]

residuals = run!(model, config, output=OpenFOAM();pref = 0.0)


U = model.momentum.U
Ux = Array(U.x.values)
Uy = Array(U.y.values)
Uz = Array(U.z.values)

V1 = [[Ux[i], Uy[i], Uz[i]] for i in eachindex(U)]


runtime = Runtime(iterations=iters2, time_step=timestep, write_interval=-1) 

config = Configuration(solvers=solvers, schemes=schemes, runtime=runtime, hardware=hardware, boundaries=BCs)
run!(model, config, output=OpenFOAM();pref = 0.0)

U = model.momentum.U
Ux = Array(U.x.values)
Uy = Array(U.y.values)
Uz = Array(U.z.values)

V2 = [[Ux[i], Uy[i], Uz[i]] for i in eachindex(U)]


function spectra_from_sim(x,V)
    Nx = Int(round(length(x)^(1/3), digits=3))

    x0 = minimum(getindex.(x, 1));  xmax = maximum(getindex.(x, 1))
    y0 = minimum(getindex.(x, 2));  ymax = maximum(getindex.(x, 2))
    z0 = minimum(getindex.(x, 3));  zmax = maximum(getindex.(x, 3))

    Lx = xmax - x0
    Ly = ymax - y0
    Lz = zmax - z0


    i = round.(Int, (getindex.(x, 1) .- x0) ./ Lx .* (Nx - 1)) .+ 1
    j = round.(Int, (getindex.(x, 2) .- y0) ./ Ly .* (Nx - 1)) .+ 1   
    k = round.(Int, (getindex.(x, 3) .- z0) ./ Lz .* (Nx - 1)) .+ 1   

    Ux = Array{Float64}(undef, Nx, Nx, Nx)
    Uy = Array{Float64}(undef, Nx, Nx, Nx)
    Uz = Array{Float64}(undef, Nx, Nx, Nx)

    for n in eachindex(x)
        Ux[i[n], j[n], k[n]] = V[n][1]
        Uy[i[n], j[n], k[n]] = V[n][2]
        Uz[i[n], j[n], k[n]] = V[n][3]
    end

    kw = round.(Int, fftfreq(Nx) .* Nx)
    κmax = Nx ÷ 2                       
    κmax_corner  = round(Int, sqrt(3) * κmax)
    Uhat_x = fft(Ux)
    Uhat_y = fft(Uy)
    Uhat_z = fft(Uz)

    En = zeros(Float64, κmax_corner + 1)       

    for ii in 1:Nx, jj in 1:Nx, kk in 1:Nx
        κ_bin = round(Int, sqrt(kw[ii]^2 + kw[jj]^2 + kw[kk]^2))
        κ_bin > κmax_corner && continue              
        energy = 0.5 * (abs2(Uhat_x[ii,jj,kk]) +
                        abs2(Uhat_y[ii,jj,kk]) +
                        abs2(Uhat_z[ii,jj,kk])) / Nx^6
        En[κ_bin + 1] += energy
    end

    # cell centres span L - Δx, so rescale by Nx/(Nx-1) to recover the box length L
    κNorm = 2π / (max(Lx, Ly, Lz) * Nx / (Nx - 1))
    κ     = (0:κmax_corner) .* κNorm
    En  ./= κNorm
    return En, κ
end
# full CBC table (read above); zero entries are filtered in spectrum_error
κdata = CBC[:,1]
E42  = CBC[:,2]
E98  = CBC[:,3]
E171 = CBC[:,4]


En, κ = spectra_from_sim(x,V)
En1, κ1 = spectra_from_sim(x,V1)
En2, κ2 = spectra_from_sim(x,V2)


# Compare the simulated spectrum with experiment for κ up to Nyquist (π/Δx).
# The experimental data is interpolated in log-log space onto the simulated κ bins.
# Returns:
#   ε_log - RMS of log10(E_sim/E_exp), i.e. how far the spectrum shape is off (in decades)
#   ε_k   - relative error in the kinetic energy resolved over the same κ range
function spectrum_error(κsim, Esim, κexp, Eexp, κmax)
    valid  = Eexp .> 0                                  # zero entries can't be logged
    κe, Ee = Float64.(κexp[valid]), Float64.(Eexp[valid])
    logEe  = linear_interpolation(log.(κe), log.(Ee))

    mask = (κe[1] .<= κsim .<= min(κmax, κe[end])) .& (Esim .> 0)
    κs, Es = κsim[mask], Esim[mask]
    Ei = exp.(logEe.(log.(κs)))

    ε_log = sqrt(sum(abs2, log10.(Es ./ Ei)) / length(Es))

    trapz(x, y) = sum((x[2:end] .- x[1:end-1]) .* (y[2:end] .+ y[1:end-1])) / 2
    ε_k = abs(trapz(κs, Es) - trapz(κs, Ei)) / trapz(κs, Ei)

    return ε_log, ε_k
end

κNyq = π / (L / Ncells)

ε_log1, ε_k1 = spectrum_error(κ1, En1, κdata, E98,  κNyq)
ε_log2, ε_k2 = spectrum_error(κ2, En2, κdata, E171, κNyq)
println("t = 0.28 s: ε_log = $ε_log1, ε_k = $ε_k1")
println("t = 0.66 s: ε_log = $ε_log2, ε_k = $ε_k2")

# tolerances: ~1.5x the reference run on the 32³ grid (C = 0.5, dt = 0.01, M = 5000)
#   reference: t = 0.28 s → ε_log = 0.052, ε_k = 0.0016
#              t = 0.66 s → ε_log = 0.064, ε_k = 0.096
tol_log = 0.10      # ~26% RMS pointwise in E(κ)
tol_k   = 0.15


@testset "WALE decaying isotropic turbulence (CBC)" begin
    @test sum(En2) < sum(En1) < sum(En)                 
    @test ε_log1 < tol_log
    @test ε_log2 < tol_log
    @test ε_k1 < tol_k
    @test ε_k2 < tol_k
end