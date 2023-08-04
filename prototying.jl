using Plots

using FVM_1D

using Krylov

# backwardFacingStep_2mm, backwardFacingStep_10mm
mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = build_mesh(mesh_file, scale=0.001)

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

gradU = Grad{Linear}(U)
Uf = FaceVectorField(mesh)


kOmega()
kk()