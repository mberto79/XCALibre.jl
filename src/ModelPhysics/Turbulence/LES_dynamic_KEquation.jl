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
    turbulence::KEquation, model::Physics{T,F,M,Tu,E,D,BI}, mdotf, peqn
    ) where {T,F,M,Tu,E,D,BI}

    (; solvers, schemes, runtime) = get_configuration(CONFIG)
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

    delta!(Δ, mesh)
    @. Δ.values = (((Δ.values^3)/0.001)^0.5) # get 2D delta
    # @. Δ.values = (((Δ.values^3))^0.5) # get 2D delta
    # @. Δ.values = Δ.values^2 # store delta squared since it will be needed

    k_eqn = (
            Time{schemes.k.time}(rho, k)
            + Divergence{schemes.k.divergence}(mdotf, k) 
            - Laplacian{schemes.k.laplacian}(mueffk, k) 
            + Si(Dkf,k) # Dkf = (Ce*rho*sqrt(k)/Δ)*k
            + Si(divU,k) # Needs adding
            ==
            Source(Pk) + Source(kSource)
            # Source(Pk) + Source(k)
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
    turbulence!(les::KEquationModel, model::Physics{T,F,M,Tu,E,D,BI}, S, S2, prev, time
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

Run turbulence model transport equations.

### Input
- `les::KEquationModel` -- KEquation LES turbulence model.
- `model`  -- Physics model defined by user.
- `S`   -- Strain rate tensor.
- `S2`  -- Square of the strain rate magnitude.
- `prev`  -- Previous field.
- `time`   -- 

"""
function turbulence!(
    les::KEquationModel, model::Physics{T,F,M,Tu,E,D,BI}, S, prev, time
    ) where {T,F,M,Tu<:AbstractTurbulenceModel,E,D,BI}

    mesh = model.domain
    config = get_configuration(CONFIG)
    (; solvers, runtime, hardware) = config
    (; workgroup) = hardware
    (; rho, rhof, nu, nuf) = model.fluid
    (; k, kf, nut, nutf, outScalar, outVector, coeffs) = les.turbulence
    (; k_eqn, state) = les
    (; U, Uf, gradU) = S
    (; Δ, magS) = les

    Pk = get_source(k_eqn, 1)
    kSource = get_source(k_eqn, 2)
    mdotf = get_flux(k_eqn, 2)
    mueffk = get_flux(k_eqn, 3)
    Dkf = get_flux(k_eqn, 4)
    divU = get_flux(k_eqn, 5)

    grad!(gradU, Uf, U, U.BCs, time)
    limit_gradient!(config.schemes.U.limiter, gradU, U)
    
    # update fluxes
    # divUf = FaceVectorField(mesh)
    # AK.foreachindex(divUf, min_elems=workgroup, block_size=workgroup) do i 
    #     divUf[i] = mdotf[i]*Uf[i]
    # end
    # div!(divU, divUf, config)

    # AK.foreachindex(Pk, min_elems=workgroup, block_size=workgroup) do i 
    #     Pk[i] = 2*nut[i]*(gradU[i]⋅Dev(S)[i])
    #     # Pk[i] = 2*nut[i]*(gradU[i]⋅Dev(gradU.result)[i])
    #     mueffk[i] = rhof[i] * (nuf[i] + nutf[i])
    #     # divU[i] = abs(2/3*rho[i]*divU[i])
    #     divU[i] = 2/3*rho[i]*tr(gradU[i]) #/mesh.cells[i].volume
    #     kSource[i] = k[i] #/mesh.cells[i].volume
    #     outScalar[i] = mesh.cells[i].volume
    # end

    #

    _filter = TopHatFilter(U)
    # Umag2hat = ScalarField(model.domain)
    # Uhat = VectorField(model.domain)
    # Uhat2 = ScalarField(model.domain)
    KK = ScalarField(model.domain)
    DevF = TensorField(model.domain)
    mag2DF = ScalarField(model.domain)
    Ce = ScalarField(model.domain)
    T_temp = TensorField(model.domain)
    LL = TensorField(model.domain)
    MM = TensorField(model.domain)
    # MM2F = ScalarField(model.domain)
    Ck = ScalarField(model.domain)

    
    AK.foreachindex(U, min_elems=workgroup, block_size=workgroup) do i 
        Umag2hati = _filter(MagSqr(U), i) 
        Uhat2i = _filter(U, i, post=(U)-> U⋅U)
        KK[i] = max(0.5*(Umag2hati - Uhat2i), 0.0)
    end

    tensorForm = gradU.result # Dev(S) # gradU.result
    Ck!(Ck, tensorForm, KK, U, T_temp, DevF, Δ, LL, MM, _filter, workgroup) 
    @. nut.values = Ck.values*Δ.values*sqrt(k.values)
    Ce!(Ce, tensorForm, KK, mag2DF, DevF, nu, nut, Δ, _filter, workgroup)

    # @. nut.values = Ck.values*Δ.values*sqrt(k.values)

    # interpolate!(nutf, nut, config)
    # correct_boundaries!(nutf, nut, nut.BCs, time, config)
    # correct_eddy_viscosity!(nutf, nut.BCs, model, config)


    # goodish iwth gradU.result

    #
    
    @. Dkf.values = Ce.values*rho.values*sqrt(k.values)/(Δ.values)
    # @. Dkf.values = 1.048*rho.values*sqrt(k.values)/(Δ.values)
    
    AK.foreachindex(Pk, min_elems=workgroup, block_size=workgroup) do i 
        Pk[i] = 2*nut[i]*(gradU[i]⋅Dev(S)[i])
        # Pk[i] = 2*nut[i]*(Dev(S)[i]⋅Dev(S)[i])
        # Pk[i] = 2*nut[i]*(gradU[i]⋅Dev(gradU.result)[i])
        mueffk[i] = rhof[i] * (nuf[i] + nutf[i])
        # divU[i] = abs(2/3*rho[i]*divU[i])
        divU[i] = 2/3*rho[i]*tr(gradU[i]) #/mesh.cells[i].volume
        kSource[i] = 0.0*k[i] #/mesh.cells[i].volume
        outScalar[i] = mesh.cells[i].volume
    end
    # Solve k equation
    # prev .= k.values
    discretise!(k_eqn, k)
    apply_boundary_conditions!(k_eqn, k.BCs, nothing, time)
    # implicit_relaxation!(k_eqn, k.values, solvers.k.relax, nothing)
    implicit_relaxation_diagdom!(k_eqn, k.values, solvers.k.relax, nothing)
    update_preconditioner!(k_eqn.preconditioner, mesh)
    k_res = solve_system!(k_eqn, solvers.k, k, nothing)
    bound!(k)
    # explicit_relaxation!(k, prev, solvers.k.relax, config)

    AK.foreachindex(U, min_elems=workgroup, block_size=workgroup) do i 
        Umag2hati = _filter(MagSqr(U), i) 
        Uhat2i = _filter(U, i, post=(U)-> U⋅U)
        KK[i] = max(0.5*(Umag2hati - Uhat2i), 0.0)
    end

    tensorForm2 = gradU.result # gradU.result Dev(S)
    Ck!(Ck, tensorForm2, KK, U, T_temp, DevF, Δ, LL, MM, _filter, workgroup) 
    # goodish iwth gradU.result
    @. nut.values = Ck.values*Δ.values*sqrt(k.values)
    # @. nut.values = 0.094*Δ.values*sqrt(k.values)

    interpolate!(nutf, nut)
    correct_boundaries!(nutf, nut, nut.BCs, time)
    correct_eddy_viscosity!(nutf, nut.BCs, model)

    # update solver state
    # state.residuals = ((:k , k_res),)
    # state.converged = k_res < solvers.k.convergence
    nothing
end

# Specialise VTK writer
function save_output(model::Physics{T,F,M,Tu,E,D,BI}, outputWriter, iteration, time
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
    config = get_configuration(CONFIG)
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

# KEquation - internal functions

function Ce!(Ce, D, KK, mag2DF, DevF, nu, nut, Δ, filter, workgroup)
    AK.foreachindex(DevF, min_elems=workgroup, block_size=workgroup) do i 
        # mag2DF[i] = filter(MagSqr(2,D), i) # added 2
        mag2DF[i] = filter(MagSqr(D), i)
        DevF[i] = filter(D, i)
    end

    AK.foreachindex(Ce, min_elems=workgroup, block_size=workgroup) do i 
        Ce_n = filter(mag2DF, i, pre=(mag2DF,i)-> Ce1(mag2DF, DevF, nu, nut, i))
        Ce_d = filter(KK, i, pre=(KK,i)-> Ce2(KK, Δ, i) )
        a = Ce_n/Ce_d
        Ce[i] = 0.5*(norm(a) + a)
    end
end

# Ce1(mag2DF, DevF, nu, nut, i) = 2*(nu[i] + nut[i])*(mag2DF[i] - DevF[i]⋅DevF[i]) # added 2
Ce1(mag2DF, DevF, nu, nut, i) = (nu[i] + nut[i])*(mag2DF[i] - DevF[i]⋅DevF[i])
Ce2(KK, Δ, i) = KK[i]^(1.5)/(2.0*Δ[i])

function Ck!(Ck, D, KK, U, T_temp, DevF, Δ, LL, MM, filter, workgroup)
    AK.foreachindex(T_temp, min_elems=workgroup, block_size=workgroup) do i 
        DevF[i] = filter(D, i)
        U2Fi = filter(Sqr(U), i)
        Uhati = filter(U, i)
        @inbounds T_temp[i] = ((U2Fi) - (Uhati*Uhati'))
    end

    AK.foreachindex(T_temp, min_elems=workgroup, block_size=workgroup) do i 
        LL[i] = filter(Dev(T_temp), i)
        MM[i] = filter(KK, i, pre =(KK,i) -> CkMM(KK, DevF, Δ, i))
    end
    
    AK.foreachindex(Ck, min_elems=workgroup, block_size=workgroup) do i 
        Ck_n = filter(LL, i, pre=(LL,i)->Ck1(LL, MM, i))
        # Ck_d =filter(MagSqr(2,MM), i) # added 2
        Ck_d =filter(MagSqr(MM), i)
        Ck_i = Ck_n/Ck_d
        Ck[i] = 0.5*(norm(Ck_i) + Ck_i)
    end
end

CkMM(KK, DevF, Δ, i) = -2*Δ[i]*sqrt(KK[i])*DevF[i]
Ck1(LL, MM, i) = 0.5*(LL[i]⋅MM[i])

function correct_nut!(nut, D, KK)
    nothing
end