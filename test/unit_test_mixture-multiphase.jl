using XCALibre

# Mean of non-zero entries (ignores untouched cells).
function mean_nonzero(v)
    s, n = 0.0, 0
    @inbounds for x in v
        if abs(x) > 1e-14
            s += x; n += 1
        end
    end
    n == 0 ? 0.0 : s / n
end

# Schiller-Naumann drag factor (Re<1000)
sn_factor(Re) = max(Re < 1000 ? 1 + 0.15*Re^0.687 : 0.0183*Re, 1.0)

# Closed-form Stokes terminal velocity
stokes_terminal(rho_d, rho_c, d, mu_c, g) = abs(g)*(rho_d - rho_c)*d^2 / (18*mu_c)

# Fixed-point Schiller-Naumann terminal velocity (Manninen buoyancy, dilute limit).
function sn_terminal_manninen(rho_d, rho_c, d, mu_c, g; tol=1e-12, itmax=500)
    tau_d = rho_d * d^2 / (18*mu_c)
    buoy = (rho_d - rho_c) / rho_d          # rho_m = rho_c in dilute limit
    u = stokes_terminal(rho_d, rho_c, d, mu_c, g)
    for _ in 1:itmax
        Re = rho_c*abs(u)*d/mu_c
        u_new = (tau_d / sn_factor(Re)) * buoy * abs(g)
        abs(u_new - u) < tol && return u_new
        u = u_new
    end
    return u
end

@testset "Mixture multiphase unit tests" begin
    rho_d, rho_c, mu_c, d, gmag = 2500.0, 1000.0, 1.0e-3, 2.0e-4, 9.81

    grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
    mesh = UNV2D_mesh(joinpath(grids_dir, "unit_test_stage2.unv"), scale=1.0)
    backend = CPU(); workgroup = AutoTune()
    activate_multithread(backend)
    mesh_dev = adapt(backend, mesh)
    config = (hardware=Hardware(backend=backend, workgroup=workgroup),)

    @testset "blend_properties! linearity" begin
        alpha = ScalarField(mesh_dev)
        prop  = ScalarField(mesh_dev)

        initialise!(alpha, 0.0)
        XCALibre.Solvers.blend_properties!(prop, alpha, 1000.0, 1.225)
        @test all(prop.values .≈ 1.225)

        initialise!(alpha, 1.0)
        XCALibre.Solvers.blend_properties!(prop, alpha, 1000.0, 1.225)
        @test all(prop.values .≈ 1000.0)

        initialise!(alpha, 0.5)
        XCALibre.Solvers.blend_properties!(prop, alpha, 1000.0, 1.225)
        @test all(prop.values .≈ 0.5*(1000.0 + 1.225))
    end

    @testset "compute_DUmDt! with a dUdt jump" begin
        U      = VectorField(mesh_dev)
        U_prev = VectorField(mesh_dev)
        gradU  = Grad{Gauss}(U)
        DUmDt  = VectorField(mesh_dev)

        initialise!(U,      [0.0, 0.2, 0.0])
        initialise!(U_prev, [0.0, 0.1, 0.0])
        dt = 0.01
        XCALibre.Solvers.compute_DUmDt!(DUmDt, U, U_prev, gradU, dt, config)
        @test all(isapprox.(DUmDt.y.values, (0.2 - 0.1) / dt; rtol=1e-10))
    end

    @testset "compute_Ur!" begin
        alpha = ScalarField(mesh_dev)
        rho   = ScalarField(mesh_dev)
        Ur    = VectorField(mesh_dev)
        DUmDt = VectorField(mesh_dev)

        initialise!(alpha, 1.0 - 1e-3)
        XCALibre.Solvers.blend_properties!(rho, alpha, rho_c, rho_d)
        initialise!(Ur,    [0.0, 0.0, 0.0])
        initialise!(DUmDt, [0.0, 0.0, 0.0])

        g_vec = SVector{3,Float64}(0.0, -gmag, 0.0)
        tau_d = rho_d*d^2 / (18*mu_c)
        
        rho_c_f = ConstantScalar(rho_c)
        rho_d_f = ConstantScalar(rho_d)
        mu_c_f  = ConstantScalar(mu_c)
        tau_d_f = ConstantScalar(tau_d)

        for _ in 1:200  # picard iterate
            XCALibre.Solvers.compute_Ur!(Ur, alpha, rho, g_vec, DUmDt,
                                         rho_c_f, rho_d_f, mu_c_f, d, tau_d_f, config)
        end

        u_expected = sn_terminal_manninen(rho_d, rho_c, d, mu_c, gmag)
        @test maximum(abs, Ur.x.values) < 1e-10
        @test maximum(abs, Ur.z.values) < 1e-10
        @test isapprox(-u_expected, mean_nonzero(Ur.y.values); rtol=5e-2)
    end

    @testset "turbulent_dispersion!" begin
        alpha  = ScalarField(mesh_dev)
        Ur     = VectorField(mesh_dev)
        ∇alpha = Grad{Gauss}(alpha)
        initialise!(alpha, 0.5)
        initialise!(Ur, [1.0, 2.0, 3.0])

        XCALibre.Solvers.turbulent_dispersion!(Ur, alpha, ∇alpha, Laminar(), 0.7, config)
        @test all(Ur.x.values .≈ 1.0)
        @test all(Ur.y.values .≈ 2.0)
        @test all(Ur.z.values .≈ 3.0)
    end
end
