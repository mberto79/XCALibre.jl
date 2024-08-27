using FVM_1D
using Accessors

mesh_file = "unv_sample_meshes/backwardFacingStep_10mm.unv"
mesh = UNV2D_mesh(mesh_file, scale=0.001)

# mesh_dev = adapt(CUDABackend(), mesh)
mesh_dev = mesh

velocity = [1.5, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

model = Physics(
    time = Steady(),
    fluid = FLUID{Incompressible}(nu = nu),
    turbulence = RANS{Laminar}(),
    energy = ENERGY{Isothermal}(),
    domain = mesh_dev
    )


modelTransient = change(model, :time, Transient())
modelTransient = change(model, (
    (:time, Transient()), 
    (:turbulence, RANS{KOmega}()(model.domain)) 
    )
)


function change(model::Physics, property, value)
    @assert property ∈ fieldnames(Physics) throw(ArgumentError(
    """$value is not in Physics. 
    Use "fieldnames(Physics)" to find available properties""")
    )

    lens = opcompose(PropertyLens(property))
    updatedModel = set(model, lens, value)
    return updatedModel
end

function change(model::Physics, args...)
    updatedModel = nothing
    for arg in args
        updatedModel = change(model, arg...)
    end
    return updatedModel
end

test(a, x...) = println(a," ", x[2])

test(1, (2,2), (3,3))

