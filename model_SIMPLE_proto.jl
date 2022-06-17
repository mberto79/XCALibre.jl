using Plots
using LinearOperators
using LinearAlgebra

using FVM_1D.Mesh2D
using FVM_1D.Plotting
using FVM_1D.Discretise
using FVM_1D.Calculate
using FVM_1D.Models
using FVM_1D.Solvers
using FVM_1D.VTK

using Krylov
using ILUZero
using IncompleteLU

function generate_mesh()
    # n_vertical      = 20 
    # n_horizontal    = 100 

    n_vertical      = 40 
    n_horizontal    = 200 

    p1 = Point(0.0,0.0,0.0)
    p2 = Point(0.5,0.0,0.0)
    p3 = Point(0.0,0.1,0.0)
    p4 = Point(0.5,0.1,0.0)
    points = [p1, p2, p3, p4]

    # Edges in x-direction
    e1 = line!(points,1,2,n_horizontal)
    e2 = line!(points,3,4,n_horizontal)
    
    # Edges in y-direction
    e3 = line!(points,1,3,n_vertical)
    e4 = line!(points,2,4,n_vertical)
    edges = [e1, e2, e3, e4]

    b1 = quad(edges, [1,2,3,4])
    blocks = [b1]

    patch1 = Patch(:inlet,  [3])
    patch2 = Patch(:outlet, [4])
    patch3 = Patch(:bottom, [1])
    patch4 = Patch(:top,    [2])
    patches = [patch1, patch2, patch3, patch4]

    builder = MeshBuilder2D(points, edges, patches, blocks)
    mesh = generate!(builder)
    return mesh
end

velocity = [1.0, 0.0, 0.0]
nu = 1e-3
Re = velocity[1]*0.1/nu

UBCs = ( 
    Dirichlet(:inlet, velocity),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, [0.0, 0.0, 0.0]),
    Dirichlet(:top, [0.0, 0.0, 0.0])
    # Neumann(:top, 0.0)
)

uxBCs = (
    Dirichlet(:inlet, velocity[1]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

uyBCs = (
    Dirichlet(:inlet, velocity[2]),
    Neumann(:outlet, 0.0),
    Dirichlet(:bottom, 0.0),
    Dirichlet(:top, 0.0)
    # Neumann(:top, 0.0)
)

pBCs = (
    Neumann(:inlet, 0.0),
    Dirichlet(:outlet, 0.0),
    Neumann(:bottom, 0.0),
    Neumann(:top, 0.0)
)

setup = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver,
    tolerance   = 1e-3,
    # tolerance   = 1e-01,
    relax       = 1.0,
    itmax       = 100,
    rtol        = 1e-6
)

setup_p = SolverSetup(
    iterations  = 100,
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    tolerance   = 1e-3,
    # tolerance   = 1e-01,
    relax       = 1.0,
    itmax       = 100,
    rtol        = 1e-6
)

#SymmlqSolver, MinresSolver - did not work!

mesh = generate_mesh()

ux = ScalarField(mesh)
uy = ScalarField(mesh)
p = ScalarField(mesh)

# function isimple!(
#     mesh, velocity, nu, ux, uy, p, 
#     uxBCs, uyBCs, pBCs, UBCs,
#     setup, setup_p, iterations
#     ; resume=true)
    # Pre-allocate fields

    U = VectorField(mesh)
    Uf = FaceVectorField(mesh)
    mdot = ScalarField(mesh)
    mdotf = FaceScalarField(mesh)
    pf = FaceScalarField(mesh)
    ∇p = Grad{Linear}(p)
    
    Hv = VectorField(mesh)
    divHv = Div(Hv)
    rD = ScalarField(mesh)
    rDf = FaceScalarField(mesh)

    # Pre-allocated auxiliary variables
    ux0 = zeros(length(ux.values))
    ux0 .= velocity[1]
    uy0 = zeros(length(ux.values))
    uy0 .= velocity[2]
    p0 = zeros(length(p.values))

    # Define models and equations
    x_momentum_eqn = Equation(mesh)
    x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)
    generate_boundary_conditions!(:ux_boundary_update!, mesh, x_momentum_model, uxBCs)

    y_momentum_eqn = Equation(mesh)
    y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)
    generate_boundary_conditions!(:uy_boundary_update!, mesh, y_momentum_model, uyBCs)

    pressure_eqn = Equation(mesh)
    pressure_correction = create_model(Diffusion, rDf, p, divHv.values) #.*D)
    generate_boundary_conditions!(:p_boundary_update!, mesh, pressure_correction, pBCs)

    # Preallocate preconditioners
    Fm = ilu0(x_momentum_eqn.A)
    Fp = ilu0(pressure_eqn.A)

    # Update initial (guessed) fields
    # if resume
    #     @. U.x = ux.values
    #     @. U.y = uy.values
    #     @. p0 = p.values
    # else
        U.x .= velocity[1]; U.y .= velocity[2]
        ux.values .= velocity[1]; uy.values .= velocity[2]
    # end

    # Perform SIMPLE loops 
    # @time for iteration ∈ 1:iterations
    @time for iteration ∈ 1:50

        println("Iteration ", iteration)
        
        interpolate!(Uf, U, UBCs)
        # interpolate!(Uf, U)
        # correct_boundaries!(Uf, U, UBCs)
        # mass_flux!(mdotf, Uf)
        # div!(mdot, mdotf)
        
        source!(∇p, pf, p, pBCs)
        negative_vector_source!(∇p)
        
        discretise!(x_momentum_eqn, x_momentum_model)
        Discretise.ux_boundary_update!(x_momentum_eqn, x_momentum_model, uxBCs)
        # println("Solving x-momentum")
        alpha_U = 0.9
        implicit_relaxation!(x_momentum_eqn, ux0, alpha_U)
        # Fm = ilu(x_momentum_eqn.A)
        ilu0!(Fm, x_momentum_eqn.A)
        run!(x_momentum_eqn, x_momentum_model, uxBCs, setup, F=Fm)
        
        discretise!(y_momentum_eqn, y_momentum_model)
        Discretise.uy_boundary_update!(y_momentum_eqn, y_momentum_model, uyBCs)
        # println("Solving y-momentum")
        implicit_relaxation!(y_momentum_eqn, uy0, alpha_U)
        run!(y_momentum_eqn, y_momentum_model, uyBCs, setup, F=Fm)
        
        # alpha = 0.9
        # @. U.x = alpha*ux.values + (1.0 - alpha)*U.x
        # @. U.y = alpha*uy.values + (1.0 - alpha)*U.y
        
        # alpha = 1.0
        # @. U.x = alpha*ux.values + (1.0 - alpha)*ux0
        # @. U.y = alpha*uy.values + (1.0 - alpha)*uy0
        
        @. ux0 = U.x
        @. uy0 = U.y
        
        @. U.x = ux.values 
        @. U.y = uy.values
        
        inverse_diagonal!(rD, x_momentum_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
        # H!(Hv, U, x_momentum_eqn, y_momentum_eqn, B, V, H)
        H_new!(Hv, U, x_momentum_eqn, y_momentum_eqn)
        div!(divHv, UBCs) 
        # divHv.values ./= volumes(mesh)
        
        discretise!(pressure_eqn, pressure_correction)
        Discretise.p_boundary_update!(pressure_eqn, pressure_correction, pBCs)
        # println("Solving pressure correction")
        # Fp = ilu(pressure_eqn.A, τ = 0.1)
        ilu0!(Fp, pressure_eqn.A)
        run!(pressure_eqn, pressure_correction, pBCs, setup_p, F=Fp)
        
        if iteration == 1
            @. p0 = p.values
        end
        
        # source!(∇p, pf, p, pBCs) 
        grad!(∇p, pf, p, pBCs) 
        @. U.x = ux0
        @. U.y = uy0
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U, UBCs)
        
        explicit_relaxation!(p, p0, 0.4)
        grad!(∇p, pf, p, pBCs) 
        # source!(∇p, pf, p, pBCs) 
        correct_velocity!(ux, uy, Hv, ∇p, rD)
        
        source!(∇p, pf, p, pBCs) 
        

    end # end for loop 
        
# end # end function

# isimple!(
#     mesh, velocity, nu, ux, uy, p, 
#     uxBCs, uyBCs, pBCs, UBCs,
#     setup, setup_p, 1000)
ux.values = Hv.values
write_vtk(mesh, ux)[1]
write_vtk(mesh, uy)
write_vtk(mesh, p)


plotly(size=(400,400), markersize=1, markerstrokewidth=1)
scatter(x(mesh), y(mesh), ux.values, color=:red)
scatter(x(mesh), y(mesh), uy.values, color=:red)

scatter(x(mesh), y(mesh), Hv.x, color=:green)
scatter(x(mesh), y(mesh), U.x + ∇p.x.*rD.values, color=:blue)

scatter(x(mesh), y(mesh), Hv.y, color=:green)
scatter(x(mesh), y(mesh), divHv.values, color=:red)
scatter(x(mesh), y(mesh), divHv.vector.x, color=:red)
scatter(x(mesh), y(mesh), divHv.vector.y, color=:red)
scatter(xf(mesh), yf(mesh), divHv.face_vector.x, color=:blue)
scatter(xf(mesh), yf(mesh), divHv.face_vector.y, color=:blue)

scatter(x(mesh), y(mesh), p.values, color=:blue)
scatter!(xf(mesh), yf(mesh), pf.values, color=:red)

scatter(x(mesh), y(mesh), ∇p.x, color=:green)
scatter(x(mesh), y(mesh), ∇p.y, color=:green)

scatter(x(mesh), y(mesh), U.x, color=:green)
scatter(x(mesh), y(mesh), U.y, color=:green)
scatter(xf(mesh), yf(mesh), Uf.x, color=:red)
scatter(xf(mesh), yf(mesh), Uf.y, color=:red)

scatter(x(mesh), y(mesh), rD.values, color=:red)
scatter(xf(mesh), yf(mesh), rDf.values, color=:red)

scatter(x(mesh), y(mesh), mdot.values, color=:red)
scatter(xf(mesh), yf(mesh), mdotf.values, color=:red)








function mass_flux!(mdotf::FaceScalarField{I,F}, Uf::FaceVectorField{I,F}) where {I,F}
    (; mesh, values) = mdotf
    (; cells, faces) = mesh 
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        values[fID] = Uf(fID)*area⋅normal
    end
end

function flux!()
    nothing
end

function create_model(::Type{ConvectionDiffusion}, U, J, phi, S)
    model = ConvectionDiffusion(
        Divergence{Linear}(U, phi),
        Laplacian{Linear}(J, phi),
        S
        )
    model.terms.term2.sign[1] = -1
    return model
end

function create_model(::Type{Diffusion}, J, phi, S)
    model = Diffusion(
        Laplacian{Linear}(J, phi),
        S
        )
    return model
end

function implicit_relaxation!(eqn::Equation{I,F}, field, alpha) where {I,F}
    (; A, b) = eqn
    for i ∈ eachindex(b)
        A[i,i] /= alpha
        b[i] += (1.0 - alpha)*A[i,i]*field[i]
    end
end

function correct_face_velocity!(Uf, p, )
    mesh = Uf.mesh
    (; cells, faces) = mesh
    nbfaces = total_boundary_faces(mesh)
    for fID ∈ (nbfaces + 1):length(faces)
        face = faces[fID]
        gradp = 0.0
        Uf.x = nothing
        ################
        # CONTINUE 
        ################
    end
end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

function correct_boundary_Hvf!(Hvf, ux, uy, ∇pf, UBCs)
    mesh = ux.mesh
    for BC ∈ UBCs
        if typeof(BC) <: Neumann
            bi = boundary_index(mesh, BC.name)
            boundary = mesh.boundaries[bi]
            correct_flux_boundary!(BC, phif, phi, boundary, faces)
        end
    end
end

function correct_flux_boundary!(
    BC::Neumann, phif::FaceScalarField{I,F}, phi, boundary, faces) where {I,F}
    (; facesID, cellsID) = boundary
    for fID ∈ facesID
        phif.values[fID] = BC.value 
    end
end

function inverse_diagonal!(rD::ScalarField, eqn)
    D = @view eqn.A[diagind(eqn.A)]
    rD.values .= 1.0./D
    nothing
end

function explicit_relaxation!(phi, phi0, alpha)
    # @. phi.values = phi.values + alpha*(phi.values - phi0)
    @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    @. phi0 = phi.values
    nothing
end

function correct_velocity!(U, Hv, ∇p, rD)
    @. U.x = Hv.x - ∇p.x*rD.values
    @. U.y = Hv.y - ∇p.y*rD.values
    nothing
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    @. ux.values = Hv.x - ∇p.x*rD.values
    @. uy.values = Hv.y - ∇p.y*rD.values
    # @. ux.values = (Hv.x - ∇p.x)*rD.values
    # @. uy.values = (Hv.y - ∇p.y)*rD.values
    nothing
end

function negative_vector_source!(∇p)
    ∇p.x .*= -1.0
    ∇p.y .*= -1.0
    nothing
end

function remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
    @. x_momentum_eqn.b -= ∇p.x
    @. y_momentum_eqn.b -= ∇p.y
    # @. x_momentum_eqn.b -= ∇p.x/rD.values
    # @. y_momentum_eqn.b -= ∇p.y/rD.values
    nothing
end
