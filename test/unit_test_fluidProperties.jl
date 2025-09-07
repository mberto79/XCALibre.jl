using XCALibre
using DelimitedFiles

## 4 pressure test points for H2 and N2:

# H2: [1, 3, 10, 12.858] bar
# N2: [1, 5, 15, 33.958] bar

H2_para_pressure_configs = [1.0, 3.0, 10.0, 12.858]
H2_normal_pressure_configs = [1.0, 3.0, 10.0, 12.964]
N2_pressure_configs = [1.0, 5.0, 15.0, 33.958]

H2_para_T_crit = 32.938
H2_normal_T_crit = 33.145
N2_T_crit = 126.192


## Temperature ranges:

# H2: [14 to 40 K with 0.25 K step]
# H2: [40 to 100 K with 1 K step]
# H2: [100 to 500 K with 10 K step]

# N2: [64 to 90 K with 0.25 K step]
# N2: [90 to 150 K with 1 K step]
# N2: [150 to 500 K with 10 K step]

H2_para_instance = HelmholtzEnergy(name=H2_para())
H2_instance = HelmholtzEnergy(name=H2())
N2_instance = HelmholtzEnergy(name=N2())

density_max_error = 0.001 # 0.1% deviation is allowed
viscosity_max_error = 0.02 # 2% deviation is allowed
them_cond_max_error = 0.15 # typically within 1-15% deviation, but at p_crit becomes highly inaccurate!
them_cond_max_crit_error = 0.4 # typically within 1-10% deviation, but at p_crit becomes highly inaccurate!


### APPROACH

# Compare data for 4 pressure test points for each fluid
# For each property (density, viscosity, conductivity), store highest deviation
# Select highest deviating test point
# Make sure deviation is less than max allowed for each of the three properties - then the test is passed


# Choose which branch of the EOS to use (liquid/vapor/supercritical)
function branch_index(T::F, Tcrit::F, Tsat::F) where {F<:AbstractFloat}
    if T > Tcrit
        return 1             # supercritical, only use index 1 since both densities are equal
    elseif T >= Tsat
        return 2               # vapor branch
    else
        return 1             # liquid branch
    end
end


# Read NIST data
function read_baseline_csv(filename)
    raw = readdlm(filename, ',', header=true)
    data, header = raw   # numeric values and column names

    cols = Dict{String, Vector{Float64}}()
    for (i, name) in enumerate(header)
        cols[string(name)] = data[:, i]
    end

    return cols
end

function relerr(model_value, reference_value)
    diff = abs(model_value - reference_value) # absolute diff

    scale = max(abs(reference_value), eps()) #  avoid division by zero by using eps()

    return diff / scale
end



function compare_model_with_baseline(eos, Tcrit, p_bar, baseline_file)
    baseline = read_baseline_csv(baseline_file)

    temps = baseline["Temperature_K"]
    density_ref = baseline["Density_kg_m3"]
    viscosity_ref = baseline["Viscosity_uPa_s"] .* 1e-6     # convert to appropriate units
    k_ref = baseline["ThermalConductivity_W_mK"]

    P = p_bar * 1e5  # convert bars to actual pressure

    # Track max errors for each property
    # Tuple layout: (max_error, Temp, computed_value, ref_value)
    max_errs = Dict(
        :rho => (0.0, NaN, NaN, NaN),
        :mu  => (0.0, NaN, NaN, NaN),
        :k   => (0.0, NaN, NaN, NaN)
    )

    for (i,T) in enumerate(temps)
        result = eos(T, P)

        idx = branch_index(T, Tcrit, result.T_sat) # pick the corrent branch of the output values

        rho_model = result.rho[idx]
        mu_model  = result.mu[idx]
        k_model   = result.k[idx]

        # Compute relative errors
        rho_err = relerr(rho_model, density_ref[i])
        mu_err  = relerr(mu_model, viscosity_ref[i])
        k_err   = relerr(k_model, k_ref[i])

        # Update max errors if current point is worse
        if rho_err > max_errs[:rho][1]
            max_errs[:rho] = (rho_err, T, rho_model, density_ref[i])
        end
        if mu_err > max_errs[:mu][1]
            max_errs[:mu] = (mu_err, T, mu_model, viscosity_ref[i])
        end
        if k_err > max_errs[:k][1]
            max_errs[:k] = (k_err, T, k_model, k_ref[i])
        end
    end

    println("[Fluid Properties] Results for pressure of $(p_bar) bar")
    println("Density    : max rel. error = $(round(max_errs[:rho][1]*100, digits=3))% at T=$(max_errs[:rho][2]) K | COMPUTED=$(max_errs[:rho][3]) | NIST=$(max_errs[:rho][4])")
    println("Viscosity  : max rel. error = $(round(max_errs[:mu][1]*100, digits=3))% at T=$(max_errs[:mu][2]) K | COMPUTED=$(max_errs[:mu][3]) | NIST=$(max_errs[:mu][4])")
    println("Therm.Cond.: max rel. error = $(round(max_errs[:k][1]*100, digits=3))% at T=$(max_errs[:k][2]) K | COMPUTED=$(max_errs[:k][3]) | NIST=$(max_errs[:k][4])\n\n")

    # CHECK TOLERANCES
    @test max_errs[:rho][1] <= density_max_error
    @test max_errs[:mu][1] <= viscosity_max_error

    if p_bar !== 33.958
        @test max_errs[:k][1] <= them_cond_max_error
    else
        @test max_errs[:k][1] <= them_cond_max_crit_error # Gets inaccurate at crit pressure for Nitrogen!
    end
end


# pressure_tag(p) = replace(string(p), "." => "-") ### false positives with this

function pressure_tag(p) # handle dot names issue
    if p in (12.858, 33.958, 12.964)   # the only cases with a dot in filenames
        return replace(string(p), "." => "-")   # e.g. 12.858 -> 12-858
    else
        return string(Int(p))  # e.g. 1.0 -> 1
    end
end


# Run H2 (parahydrogen) tests
for p in H2_para_pressure_configs
    filename = "NIST_H2_Results_P$(pressure_tag(p))_bar.csv"
    println("Testing H2 (parahydrogen) at $(p) bar → $(filename)")
    compare_model_with_baseline(
        H2_para_instance,
        H2_para_T_crit,
        p,
        joinpath(@__DIR__, "Fluids_NIST_Data", filename)
    )
end

# Run H2 (normal hydrogen) tests
for p in H2_normal_pressure_configs
    filename = "NIST_H2_NORMAL_Results_P$(pressure_tag(p))_bar.csv"
    println("Testing H2 (normal hydrogen) at $(p) bar → $(filename)")
    compare_model_with_baseline(
        H2_instance,
        H2_normal_T_crit,
        p,
        joinpath(@__DIR__, "Fluids_NIST_Data", filename)
    )
end

# Run N2 tests
for p in N2_pressure_configs
    filename = "NIST_N2_Results_P$(pressure_tag(p))_bar.csv"
    println("Testing N2 at $(p) bar → $(filename)")
    compare_model_with_baseline(
        N2_instance,
        N2_T_crit,
        p,
        joinpath(@__DIR__, "Fluids_NIST_Data", filename)
    )
end