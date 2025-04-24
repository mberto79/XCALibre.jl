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
struct KEquation{S1,S2,S3,S4,S5,V1,C} <: AbstractLESModel
    nut::S1
    nutf::S2
    k::S3
    kf::S4
    outScalar::S5
    outVector::V1
    coeffs::C #I know there is only one coefficient for LES but this makes the DES implementation easier
end
Adapt.@adapt_structure KEquation

struct KEquationModel{T,D,S1,S2, E1}
    turbulence::T
    Δ::D 
    magS::S1
    k_eqn::E1
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
    outScalar = ScalarField(mesh)
    outVector = VectorField(mesh)
    coeffs = les.args
    KEquation(nut, nutf, k, kf, outScalar, outVector, coeffs)
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
    Dkf = ScalarField(mesh)
    divU = ScalarField(mesh)
    kSource = ScalarField(mesh)

    delta!(Δ, mesh, config)
    @. Δ.values = (((Δ.values^3)/0.001)^0.5) # get 2D delta
    # @. Δ.values = Δ.values^2 # store delta squared since it will be needed

    k_eqn = (
            Time{schemes.k.time}(rho, k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(mueffk, k) 
            + Si(Dkf,k) # Dkf = (Ce*rho*sqrt(k)/Δ)*k
            + Si(divU,k) # Needs adding
            ==
            Source(Pk) + Source(kSource)
        ) → eqn

    @reset k_eqn.preconditioner = set_preconditioner(
        solvers.k.preconditioner, k_eqn, k.BCs, config)

    @reset k_eqn.solver = solvers.k.solver(_A(k_eqn), _b(k_eqn))
    
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

    mesh = model.domain
    (; solvers, runtime, hardware) = config
    (; workgroup) = hardware
    (; rho, rhof, nu, nuf) = model.fluid
    (; k, kf, nut, nutf, outScalar, outVector, coeffs) = les.turbulence
    (; k_eqn, state) = les
    (; U, Uf, gradU) = S
    (; Δ, magS) = les

    Pk = get_source(k_eqn, 1)
    kSource = get_source(k_eqn, 2)
    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    divU = get_flux(k_eqn, 5)

    grad!(gradU, Uf, U, U.BCs, time, config)
    limit_gradient!(config.schemes.U.limiter, gradU, U, config)
    
    # update fluxes 
    # magnitude2!(Pk, S, config, scale_factor=2) # mag2 written to Pk
    # @. Pk.values = rho.values*nut.values*Pk.values # corrects Pk to become actual production
    production!(Pk, nut, gradU, S, config)
    @. mueffk.values = rhof.values * (nuf.values + nutf.values)
    div!(divU, Uf, config)
    @. divU.values = abs(2/3*rho.values*divU.values)
    @. kSource.values = k.values #/getproperty.(mesh.cells, :volume)

    # Umag2 = ScalarField(model.domain)
    # magnitude2!(Umag2, U, config)
    surfaceArea = cell_surface_area(U, config)
    filter = TopHatFilter(U, config)

    Umag2hat = ScalarField(model.domain)
    # basic_filter!(Umag2hat, Umag2, config)
    basic_filter_new!(Umag2hat, MagSqr(Uf), surfaceArea, config)
    # @. outScalar.values = Umag2hat.values
    

    Uhat = VectorField(model.domain)
    Uhat2 = ScalarField(model.domain)
    # basic_filter!(Uhat, U, config)
    basic_filter_new!(Uhat, Uf, surfaceArea, config)
    magnitude2!(Uhat2, Uhat, config)
    @. outScalar.values = Umag2hat.values - Uhat2.values

    AK.foreachindex(U, min_elems=workgroup, block_size=workgroup) do i 
        Umag2hat0 = filter(U, i, pre = (U, i)-> U[i]⋅U[i])
        Uhat20 = filter(U, i, post = (x) -> x⋅x)
        # outScalar[i] = sqrt(max(0.5*(Umag2hat0 - Uhat20), 0.0))
    end

    KK = ScalarField(model.domain)
    @. KK.values = max(0.5*(Umag2hat.values - Uhat2.values), 1e-30)
    # @. outScalar.values = sqrt(KK.values) # sqrt(KK.values)



    DevF = TensorField(model.domain)
   

    temp = ScalarField(model.domain)
    AK.foreachindex(DevF, min_elems=workgroup, block_size=workgroup) do i 
        DevFi = filter(Dev(S), i);  DevF[i] =  DevFi
        mag2DFi = filter(Dev(S), i, pre = (x, i) -> 2*x[i]⋅x[i])
        temp[i] = max((nu[i] + nut[i])*(mag2DFi - 2*DevFi⋅DevFi), 1e-30)
    end

    numerator = ScalarField(model.domain)
    # denominator = ScalarField(model.domain)
    basic_filter!(numerator, temp, surfaceArea, config)

    AK.foreachindex(temp, min_elems=workgroup, block_size=workgroup) do i 
        n = filter(temp, i)
        d = filter(KK, i, pre=(KK,i)-> KK[i]^(1.5)/(2.0*Δ[i]))
        a = max(n/d, 0)
        temp[i] = 0.5*(norm(a) + a) # Ce
    end

    # The fun part of dealing with tensors LL and MM 

    T_temp = TensorField(model.domain)
    AK.foreachindex(T_temp, min_elems=workgroup, block_size=workgroup) do i 
        U2Fi = filter(U, i, pre=(U, i)-> U[i]*U[i]')
        @inbounds T_temp[i] = U2Fi - Uhat[i]*Uhat[i]'
    end

    LL = TensorField(model.domain)
    basic_filter!(LL, Dev(T_temp), surfaceArea, config)
    
    MM = TensorField(model.domain)
    MMi!(T_temp, KK,Δ, DevF, config)
    basic_filter!(MM, T_temp, surfaceArea, config)
    
    MM2F = ScalarField(model.domain)

    AK.foreachindex(MM2F, min_elems=workgroup, block_size=workgroup) do i 
        MM2F[i] = filter(MM, i, pre = (x, i) -> 2*x[i]⋅x[i])
    end

    Ck = ScalarField(model.domain)
    AK.foreachindex(Ck, min_elems=workgroup, block_size=workgroup) do i 
        Ck_i = filter(LL, i, pre = (LL, i)-> 0.5*LL[i]⋅MM[i])/MM2F[i]
        Ck[i] = 0.5*(norm(Ck_i) + Ck_i)
        # println(Ck_i)
    end
    
    # @. Dkf.values = temp.values*rho.values*sqrt(k.values)/(Δ.values*k.values)
    @. Dkf.values = temp.values*rho.values*sqrt(k.values)/(Δ.values)

    # Solve k equation
    # prev .= k.values
    discretise!(k_eqn, k, config)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, time, config)
    # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing, config)
    implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing, config)
    update_preconditioner!(k_eqn.preconditioner, mesh, config)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing, config)
    bound!(k, config)
    # explicit_relaxation!(k, prev, solvers.k.relax, config)

    # from Smagorinsky to get solution during prototyping
    # magnitude!(magS, S, config)
    # @. magS.values *= sqrt(2) # should fuse into definition of magnitude function!
    # @. nut.values = coeffs.C*Δ.values*magS.values # careful: here Δ = Δ²

    @. nut.values = 0.0 #Ck.values*Δ.values*sqrt(k.values)
    # this->nut_ = Ck(D, KK)*sqrt(k_)*this->delta()

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
            ("nut", model.turbulence.nut),
            ("outScalar", model.turbulence.outScalar),
            ("outVector", model.turbulence.outVector)
        )
    end
    write_results(iteration, model.domain, outputWriter, args...)
end

# KEquation - internal functions

function MMi!(MM, KK,Δ, DevF, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # # Launch result calculation kernel
    kernel! = _MMi!(backend, workgroup)
    kernel!(MM, KK,Δ, DevF, ndrange=length(MM))
end

@kernel function _MMi!(MM, KK,Δ, DevF)
    i = @index(Global)

    @inbounds begin
        MM[i] = -2*Δ[i]*sqrt(max(KK[i],1e-30))*DevF[i]
    end
end

function production!(Pk, nut, gradU, S, config)
    (; hardware) = config
    (; backend, workgroup) = hardware

    # # Launch result calculation kernel
    kernel! = _production!(backend, workgroup)
    kernel!(Pk, nut, gradU, Dev(S), ndrange=length(Pk))
end

@kernel function _production!(Pk, nut, gradU, DevgradU)
    i = @index(Global)

    @inbounds begin
        # Pk[i] = 2*nut[i]*(gradU[i]⋅DevgradU[i])
        Pk[i] = 2*nut[i]*(gradU[i]⋅DevgradU[i])
    end 
end

function Ce(D, KK)
    nothing
end

function correct_nut!(nut, D, KK)
    nothing
end