export kOmega, kk

struct kOmegaCoefficients{T}
    β⁺::T
    α1::T
    β1::T
    σκ::T
    σω::T
end



Dk(β⁺, k, ω) = β⁺.*k.values.*ω.values

function kOmega(U, Uf, gradU, gradUT, mdotf)
    p = ScalarField(mesh)
    U = VectorField(mesh)
    κ = ScalarField(mesh)
    ω = ScalarField(mesh)
    
    mdotf = FaceScalarField(mesh)
    nueffk = FaceScalarField(mesh)
    nueffω = FaceScalarField(mesh)
    Dkf = ScalarField(mesh)
    Dωf = ScalarField(mesh)
    Pk = ScalarField(mesh)
    Pω = ScalarField(mesh)
    
    typeof(Divergence{Linear}(mdotf, κ)) <: AbstractOperator  # Dkf = β⁺*ω
    
    k_model = (
            Divergence{Linear}(mdotf, κ) 
            - Laplacian{Linear}(nueffk, κ) 
            + Si(Dkf,κ)  # Dkf = β⁺*ω
            ==
            Source(Pk)
        )
    
    ω_model = (
        Divergence{Linear}(mdotf, ω) 
        - Laplacian{Linear}(nueffω, ω) 
        + Si(Dωf,ω)  # Dkf = β⁺*ω
        ==
        Source(Pω)
    )
end

function kk()
    println("KOmega model implementation")
end