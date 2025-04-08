export KEquation

# Model type definition
"""
    KEquation <: AbstractTurbulenceModel

KEquation LES model containing all Smagorinksy field parameters.

### Fields
- `nut` -- Eddy viscosity ScalarField.
- `nutf` -- Eddy viscosity FaceScalarField.
- `coeffs` -- Model coefficients.

"""
struct KEquation{S1,S2,S3,S4,C} <: AbstractLESModel
    nut::S1
    nutf::S2
    k::S3
    kf::S4
    coeffs::C #I know there is only one coefficient for LES but this makes the DES implementation easier
end
Adapt.@adapt_structure KEquation

struct KEquationModel{T,D,S1,S2, E1}
    turbulence::T
    Δ::D 
    magS::S1
    keqn::E1
    state::S2
end
Adapt.@adapt_structure KEquationModel

# Model API constructor (pass user input as keyword arguments and process as needed)
LES{KEquation}(; C=0.15) = begin 
    coeffs = (C=C,)
    ARG = typeof(coeffs)
    LES{KEquation,ARG}(coeffs)
end

# Functor as constructor (internally called by Physics API): Returns fields and user data
(les::LES{KEquation, ARG})(mesh) where ARG = begin
    nut = ScalarField(mesh)
    nutf = FaceScalarField(mesh)
    k = ScalarField(mesh)
    kf = FaceScalarField(mesh)
    coeffs = les.args
    KEquation(nut, nutf, k, kf, coeffs)
end

# Model initialisation
"""
    initialise(turbulence::KEquation, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

Initialisation of turbulent transport equations.

### Input
- `turbulence` -- turbulence model.
- `model`  -- Physics model defined by user.
- `mdtof`  -- Face mass flow.
- `peqn`   -- Pressure equation.
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
          hardware structures set.

### Output
- `KEquationModel(
        turbulence, 
        Δ, 
        magS, 
        ModelState((), false)
    )`  -- Turbulence model structure.

"""
function initialise(
    turbulence::KEquation, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn, config
    ) where {T,F,M,Tu,E,D,BI}

    (; solvers, schemes, runtime) = config
    mesh = model.domain
    (; k, nut) = turbulence
    (; rho) = model.fluid
    eqn = peqn.equation
    
    magS = ScalarField(mesh)
    Δ = ScalarField(mesh)
    Pk = ScalarField(mesh)
    mueffk = FaceScalarField(mesh)
    Dkf = FaceScalarField(mesh)

    delta!(Δ, mesh, config)
    @. Δ.values = Δ.values^2 # store delta squared since it will be needed

    k_eqn = (
            Time{schemes.k.time}(rho, k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(mueffk, k) 
            + Si(Dkf,k) # Dkf = Ce*rho*sqrt(k)/Δ*k
            # + Si(divU,k) # Needs adding
            ==
            Source(Pk)
        ) → eqn
    
    return KEquationModel(
        turbulence, 
        Δ, 
        magS, 
        k_eqn,
        ModelState((), false)
    )
end

# Model solver call (implementation)
"""
    turbulence!(les::KEquationModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `les::KEquationModel` -- KEquation LES turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 
- `config` -- Configuration structure defined by user with solvers, schemes, runtime and 
              hardware structures set.

"""
function turbulence!(
    les::KEquationModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time, config
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    (; rho, rhof, nu, nuf) = model.fluid
    (; k, kf, nut, nutf, coeffs) = les.turbulence
    (; keqn, state) = les
    (; U, Uf, gradU) = S
    (; Δ, magS) = les

    Pk = get_source(keqn, 1)
    mueffk = get_flux(keqn, 3)
    Dkf = get_flux(keqn, 4)

    grad!(gradU, Uf, U, U.BCs, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    magnitude2!(Pk, S, config, scale_factor=2.0) # mag2 written to Pk

    # update fluxes 
    @. Pk.values = rho.values*nut.values*Pk.values # corrects Pk to become actual production
    @. mueffk.values = rhof.values * (nuf.values + nutf.values)

    Umag2 = ScalarField(model.domain)
    Umag2f = FaceScalarField(model.domain)
    Umag2hat = ScalarField(model.domain)
    magnitude2!(Umag2, U, config)
    magnitude2!(Umag2f, Uf, config)
    basic_filter!(Umag2hat, Umag2, Umag2f, time, config)

    Uhat = VectorField(model.domain)
    Uhat2 = ScalarField(model.domain)
    basic_filter!(Uhat, U, Uf, time, config)
    magnitude2!(Uhat2, Uhat, config)

    KK = ScalarField(model.domain)
    @. KK.values = 0.5*(Umag2hat.values - Uhat2.values)
    @. k.values = KK.values

    devUgrad = TensorField(model.domain)

    # from Smagorinsky to get solution during prototyping
    magnitude!(magS, S, config)
    @. magS.values *= sqrt(2) # should fuse into definition of magnitude function!

    # # Solve k equation
    # # prev .= k.values
    # discretise!(k_eqn, k, config)
    # apply_boundary_conditions!(k_eqn, k.BCs, nothing, time, config)
    # # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    # implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    # update_preconditioner!(k_eqn.preconditioner, mesh, config)
    # k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    # bound!(k, config)
    # # explicit_relaxation!(k, prev, solvers.k.relax, config)

    # update eddy viscosity 
    @. nut.values = coeffs.C*Δ.values*magS.values # careful: here Δ = Δ²

    interpolate!(nutf, nut, config)
    correct_boundaries!(nutf, nut, nut.BCs, time, config)
    correct_eddy_viscosity!(nutf, nut.BCs, model, config)

    # update solver state
    # state.residuals = ((:k , k_res),)
    # state.converged = k_res < solvers.k.convergence
    nothing
end

# Specialise VTK writer
function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration
    ) where {T,F,M,Tu<:KEquation,E,D,BI}
    if typeof(model.fluid)<:AbstractCompressible
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("nut", model.turbulence.nut),
            ("T", model.energy.T)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("p", model.momentum.p),
            ("k", model.turbulence.k),
            ("nut", model.turbulence.nut)
        )
    end
    write_results(iteration, model.domain, outputWriter, args...)
end

# KEquation - internal functions

function basic_filter!(phiFiltered, phi, phif, time, config)
    interpolate!(phif, phi, config)   
    correct_boundaries!(phif, phi, phi.BCs, time, config)
    integrate_surface!(phiFiltered, phif, config)
end

function integrate_surface!(phiFiltered, phif, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # (; x, y, z) = grad.result
    
    # # Launch result calculation kernel
    kernel! = _integrate_surface!(backend, workgroup)
    kernel!(phiFiltered, phif, ndrange=length(phiFiltered))

    # # number of boundary faces
    # nbfaces = length(phif.mesh.boundary_cellsID)
    
    # kernel! = boundary_faces_contribution!(backend, workgroup)
    # kernel!(x, y, z, phif, ndrange=nbfaces)
end

@kernel function _integrate_surface!(phiFiltered, phif::FaceScalarField)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phif
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        areaSum = 0.0
        # surfaceSum = SVector{3}(0.0,0.0,0.0)
        surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area) = faces[fID]
            # surfaceSum += values[fID]*area
            surfaceSum += phif[fID]*area
            areaSum += area
        end
        res = surfaceSum/areaSum

        phiFiltered[i] = res
    end
end

@kernel function _integrate_surface!(phiFiltered, phif::FaceVectorField)
    i = @index(Global)

    @uniform begin
        # (; mesh, values) = phif
        (; mesh) = phif
        (; faces, cells, cell_faces, cell_nsign) = mesh
    end
     
    @inbounds begin
        (; volume, faces_range) = cells[i]

        areaSum = 0.0
        surfaceSum = SVector{3}(0.0,0.0,0.0)
        # surfaceSum = 0.0
        for fi ∈ faces_range
            fID = cell_faces[fi]
            (; area) = faces[fID]
            # surfaceSum += values[fID]*area
            surfaceSum += phif[fID]*area
            areaSum += area
        end
        res = surfaceSum/areaSum

        phiFiltered[i] = res
    end
end

function Ck(D, KK)
    nothing
end

function Ce(D, KK)
    nothing
end

function correct_nut!(nut, D, KK)
    nothing
end