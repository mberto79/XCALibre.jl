# =============================================================================
#  Rung 2.7 - are the tabulated derivatives consistent with the tabulated rho?
# =============================================================================
#
#  Plan: dev_notes_LH2_validation_plan.md, Stage 2.
#
#  WHY THIS IS NOW A PRECONDITION, NOT A NICETY
#
#  `RealFluid` tabulates rho, psi and beta as THREE INDEPENDENT tables, each
#  walked separately from the Helmholtz EOS. Nothing forces them to agree with
#  one another once discretised onto a bilinear grid. That was tolerable while
#  they were used in different places, and it is not any more:
#
#      psi_s = psi_T - beta^2*T/(rho*cp)
#
#  the isentropic correction that `multiphase_thermo_acoustic = :implicit` puts
#  into the pressure equation's time coefficient, mixes all three in one
#  expression. If `psi` is not the derivative of the `rho` the solver transports,
#  the correction is wrong by exactly that discrepancy - and it is wrong in the
#  coefficient that sets the acoustic response, which is where rung 2.5 showed
#  errors become instabilities.
#
#  Seconds to run, no CFD, so it goes before RealFluid reaches the duct.
#
#  WHAT IS CHECKED, on-branch, against the SAME table:
#
#      psi(p,T)   ==  (1/rho)*(d rho/dp)_T          [1/Pa]
#      beta(p,T)  == -(1/rho)*(d rho/dT)_p          [1/K]
#
#  by central difference on the same grid, through `table_lookup` rather than
#  against the underlying EOS. `unit_test_property_tables.jl` already pins the
#  tables to the EOS; what is unverified is whether the three INTERPOLATED fields
#  the solver actually consumes agree with each other.
#
#  ACCEPTANCE is convergence, not a magnitude. A central difference carries its
#  own truncation error, so the mismatch cannot be zero; what it must do is FALL
#  as the table is refined. One that does not fall is a formulation error in the
#  builder, and no refinement will remove it.
#
#  Peng-Robinson is the control: rho, psi and beta all come from ONE cubic,
#  analytically, so they are consistent by construction. That is what "consistent"
#  looks like - independently of whether the cubic is any good for hydrogen, which
#  it is not (see the header of 2_peng_robinson.jl).
# =============================================================================

using XCALibre
using Test
using Printf

# Saturated LH2 operating range of the Tatsumoto case: 0.4-1.1 MPa against a
# critical pressure of 1.2964 MPa.
const P_RANGE = (0.30e6, 1.25e6)

# The liquid table must stop below T_c = 33.145 K. Above it there is no liquid
# root, both branches collapse onto the single supercritical one, and the table
# contains a step change - `build_property_tables` refuses such a table outright,
# which is correct, and worth knowing before it bites mid-run.
t_range(branch) = branch === :liquid ? (20.0, 32.5) : (20.0, 34.0)

"""
    on_branch_points(sat, branch, tr; n, margin)

Sample points where `branch` physically exists, keeping `margin` clear of the
saturation line so a central-difference stencil cannot straddle it.

ON BRANCH ONLY, AND WHY. A tabulated branch is a function of state only where
that phase EXISTS: liquid below `T_sat(p)`, vapour above it. Outside that,
`build_property_tables` continues the branch metastably for a few K and then
falls back to the saturation line - deliberately, because the mixture blend
evaluates BOTH phases in EVERY cell, so a single-phase liquid cell still asks for
a vapour density.

In that fallback region the three tables are NOT derivatives of a common `rho`
and cannot be: `rho` is pinned to the saturation value while `psi` and `beta` are
whatever the builder last computed. Checking their consistency there is
meaningless - an earlier version of this file did exactly that and reported
relative errors of 1e15, which was the metric failing, not the tables.

What matters off-branch is not consistency but that the SOLVER survives it, which
is what the `0.1*kappa_T` floor in `_update_psi_isentropic!` is for. It is
measured separately below.
"""
function on_branch_points(sat, branch, tr; n=7, margin=1.0)
    pts = Tuple{Float64,Float64}[]
    for i in 1:n
        p = P_RANGE[1] + (P_RANGE[2] - P_RANGE[1])*(i/(n + 1))
        T_sat = saturation_temperature(sat, p)
        lo, hi = branch === :liquid ? (tr[1] + margin, T_sat - margin) :
                                      (T_sat + margin, tr[2] - margin)
        hi > lo || continue
        for j in 1:n
            push!(pts, (p, lo + (hi - lo)*(j/(n + 1))))
        end
    end
    return pts
end

"""The complement: states where the phase does not exist and the table is a
metastable continuation or the saturation-line fallback."""
function off_branch_points(sat, branch, tr; n=7, margin=1.0)
    pts = Tuple{Float64,Float64}[]
    for i in 1:n
        p = P_RANGE[1] + (P_RANGE[2] - P_RANGE[1])*(i/(n + 1))
        T_sat = saturation_temperature(sat, p)
        lo, hi = branch === :liquid ? (T_sat + margin, tr[2]) : (tr[1], T_sat - margin)
        hi > lo || continue
        for j in 1:n
            push!(pts, (p, lo + (hi - lo)*(j/(n + 1))))
        end
    end
    return pts
end

# Symmetric relative difference. Normalising by one side alone makes the metric
# explode wherever that side passes through zero, which says nothing about the
# tables.
reldiff(a, b) = abs(a - b)/max(abs(a), abs(b), eps())

function derivative_mismatch(rho_tab, psi_tab, beta_tab, pts; dp, dT)
    worst_psi = 0.0; worst_beta = 0.0
    sum_psi = 0.0; sum_beta = 0.0; nsamp = 0
    for (p, T) in pts
        rho = table_lookup(rho_tab, p, T)
        rho > 0 || continue
        drdp = (table_lookup(rho_tab, p + dp, T) - table_lookup(rho_tab, p - dp, T))/(2dp)
        e_psi = reldiff(table_lookup(psi_tab, p, T), drdp/rho)
        drdT = (table_lookup(rho_tab, p, T + dT) - table_lookup(rho_tab, p, T - dT))/(2dT)
        e_beta = reldiff(table_lookup(beta_tab, p, T), -drdT/rho)
        worst_psi = max(worst_psi, e_psi); worst_beta = max(worst_beta, e_beta)
        sum_psi += e_psi^2; sum_beta += e_beta^2; nsamp += 1
    end
    return (worst_psi = worst_psi, worst_beta = worst_beta,
            rms_psi = sqrt(sum_psi/max(nsamp, 1)),
            rms_beta = sqrt(sum_beta/max(nsamp, 1)), nsamples = nsamp)
end

# How hard the isentropic correction works, as a fraction of `psi`. On-branch it
# equals `1 - 1/gamma` and must stay below 1; off-branch no such bound applies.
function isentropic_weight(rho_tab, psi_tab, beta_tab, cp_tab, pts)
    worst = 0.0; at = (0.0, 0.0); n = 0
    for (p, T) in pts
        rho = table_lookup(rho_tab, p, T); rho > 0 || continue
        psi = table_lookup(psi_tab, p, T); psi > 0 || continue
        cp = table_lookup(cp_tab, p, T); cp > 0 || continue
        beta = table_lookup(beta_tab, p, T)
        w = beta^2*T/(rho*cp)/psi
        n += 1
        if w > worst
            worst = w; at = (p, T)
        end
    end
    return (weight = worst, at = at, n = n)
end

@testset "2.7 tabulated derivative consistency" begin

    println("\n", "="^78)
    println(" Rung 2.7 - do tabulated psi and beta match d(rho) of the same table?")
    println("="^78)

    SAT = build_saturation_curve(H2(), p=P_RANGE, T=(19.0, 120.0), np=201, nT=201)

    for branch in (:liquid, :vapour)
        tr = t_range(branch)
        pts = on_branch_points(SAT, branch, tr)
        @printf("\n%s H2  -  %d on-branch sample points\n",
                uppercase(string(branch)), length(pts))
        @printf("  %-11s %12s %12s %12s %12s\n",
                "grid", "worst psi", "rms psi", "worst beta", "rms beta")

        results = map((41, 81, 161)) do ng
            rf = RealFluid(H2(), branch; p=P_RANGE, T=tr, np=ng, nT=ng, verbose=false)
            dp = (P_RANGE[2] - P_RANGE[1])/(ng - 1)
            dT = (tr[2] - tr[1])/(ng - 1)
            m = derivative_mismatch(rf.rho.rho, rf.rho.psi, rf.beta.beta, pts; dp=dp, dT=dT)
            @printf("  %4dx%-6d %12.3e %12.3e %12.3e %12.3e\n",
                    ng, ng, m.worst_psi, m.rms_psi, m.worst_beta, m.rms_beta)
            (ng = ng, m...)
        end

        println("\n  convergence under refinement (must FALL - see header):")
        for i in 1:length(results)-1
            c, f = results[i], results[i+1]
            r = (f.ng - 1)/(c.ng - 1)
            @printf("    %3d -> %3d (r=%.1f):  psi %6.2f   beta %6.2f\n", c.ng, f.ng, r,
                    log(c.rms_psi/f.rms_psi)/log(r), log(c.rms_beta/f.rms_beta)/log(r))
        end

        @test results[end].nsamples > 0
        @test results[end].rms_psi  < results[1].rms_psi
        @test results[end].rms_beta < results[1].rms_beta
        # Loose on purpose: this catches a transposed or mis-scaled table, which
        # shows up as O(1) or worse, not a few percent of interpolation error.
        @test results[end].rms_psi  < 0.05
        @test results[end].rms_beta < 0.05

        rf = RealFluid(H2(), branch; p=P_RANGE, T=tr, np=161, nT=161, verbose=false)
        on  = isentropic_weight(rf.rho.rho, rf.rho.psi, rf.beta.beta, rf.cp.cp, pts)
        off = isentropic_weight(rf.rho.rho, rf.rho.psi, rf.beta.beta, rf.cp.cp,
                                off_branch_points(SAT, branch, tr))
        println("\n  isentropic correction  beta^2*T/(rho*cp) / psi:")
        @printf("    ON  branch (%3d pts): max %8.4f at p = %.2f MPa, T = %.1f K -> gamma = %.2f\n",
                on.n, on.weight, on.at[1]/1e6, on.at[2], 1/max(1 - on.weight, 1e-9))
        @printf("    OFF branch (%3d pts): max %8.4f   (metastable / saturation fallback)\n",
                off.n, off.weight)
        @test on.n > 0
        @test 0.0 <= on.weight < 1.0
    end

    println("\n", "-"^78)
    println(" Peng-Robinson control - one cubic, so all three are consistent by")
    println(" construction. This is what a consistent set looks like.")
    println("-"^78)
    for branch in (:liquid, :vapour)
        pr = PengRobinson(H2(), branch=branch)
        worst = 0.0
        for (p, T) in on_branch_points(SAT, branch, t_range(branch))
            dp = branch === :liquid ? 1.0e2 : 1.0e3
            rho = XCALibre.ModelPhysics.pr_density(pr, p, T)
            r1  = XCALibre.ModelPhysics.pr_density(pr, p + dp, T)
            r0  = XCALibre.ModelPhysics.pr_density(pr, p - dp, T)
            (isfinite(rho) && rho > 0 && isfinite(r1) && isfinite(r0)) || continue
            worst = max(worst, reldiff(phase_compressibility(pr, p, T),
                                       ((r1 - r0)/(2dp))/rho))
        end
        @printf("  %-8s worst psi mismatch against its own d(rho)/dp: %.3e\n", branch, worst)
        @test worst < 1e-3
    end
    println()
end
