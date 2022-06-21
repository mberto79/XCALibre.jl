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
using LoopVectorization

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
    iterations  = 1,
    solver      = GmresSolver,
    tolerance   = 1e-1,
    # tolerance   = 1e-01,
    relax       = 1.0,
    itmax       = 100,
    rtol        = 1e-1
)

setup_p = SolverSetup(
    iterations  = 1,
    solver      = GmresSolver, #CgSolver, #GmresSolver, #BicgstabSolver,
    tolerance   = 1e-1,
    # tolerance   = 1e-01,
    relax       = 1.0,
    itmax       = 100,
    rtol        = 1e-2
)

#SymmlqSolver, MinresSolver - did not work!

mesh = generate_mesh()

ux = ScalarField(mesh)
uy = ScalarField(mesh)
p = ScalarField(mesh)

function isimple!(
    mesh, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup, setup_p, iterations
    ; resume=true)

    # Pre-allocate fields
    U = VectorField(mesh)
    Uf = FaceVectorField(mesh)
    mdot = ScalarField(mesh)
    mdotf = FaceScalarField(mesh)
    pf = FaceScalarField(mesh)
    ∇p = Grad{Linear}(p)
    
    Hv = VectorField(mesh)
    # divHv = Div(Hv)
    divHv = Div(Hv, FaceVectorField(mesh), zeros(length(mesh.cells)), mesh)
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
    m, n = x_momentum_eqn.A.m, x_momentum_eqn.A.n
    opAx = LinearOperator(x_momentum_eqn.A)
    Px = ilu0(x_momentum_eqn.A)
    opPUx = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Px, v))
    
    
    x_momentum_model = create_model(ConvectionDiffusion, Uf, nu, ux, ∇p.x)
    
    y_momentum_eqn = Equation(mesh)
    opAy = LinearOperator(y_momentum_eqn.A)
    Py = ilu0(y_momentum_eqn.A)
    opPUy = LinearOperator(Float64, m, n, false, false, (y, v) -> ldiv!(y, Py, v))

    y_momentum_model = create_model(ConvectionDiffusion, Uf, nu, uy, ∇p.y)
    
    pressure_eqn = Equation(mesh)
    pressure_correction = create_model(Diffusion, rDf, p, divHv.values)
    rDf.values .= 1.0
    discretise!(pressure_eqn, pressure_correction)
    apply_boundary_conditions!(pressure_eqn, pressure_correction, pBCs)
    
    opAp = LinearOperator(pressure_eqn.A)
    opPP = opLDL(pressure_eqn.A)

    #### NEED TO IMPLEMENT A SENSIBLE INITIALISATION TO INCLUDE WARM START!!!!
    # Update initial (guessed) fields
    # if resume
    #     @. U.x = ux.values
    #     @. U.y = uy.values
    #     @. p0 = p.values
    # else
        @turbo ux0 .= ux.values
        @turbo uy0 .= uy.values 
        @turbo U.x .= velocity[1]
        @turbo U.y .= velocity[2]
        @turbo ux.values .= velocity[1]
        @turbo uy.values .= velocity[2]
        @turbo p0 .= p.values
    # end
    Rx = Float64[]
    volume  = volumes(mesh)
    
    interpolate!(Uf, U, UBCs)   
    source!(∇p, pf, p, pBCs)
    
    solver_p = setup_p.solver(pressure_eqn.A, pressure_eqn.b)
    solver_U = setup.solver(x_momentum_eqn.A, x_momentum_eqn.b)
    
    # Perform SIMPLE loops 
    @time for iteration ∈ 1:iterations

        print("\nIteration ", iteration, "\n") # 91 allocations
        
        # interpolate!(Uf, U, UBCs)
        # interpolate!(Uf, U)
        # correct_boundaries!(Uf, U, UBCs)
        # mass_flux!(mdotf, Uf)
        # div!(mdot, mdotf)
        
        # source!(∇p, pf, p, pBCs)
        # grad!(∇p, pf, p, pBCs)
        negative_vector_source!(∇p)
        
        discretise!(x_momentum_eqn, x_momentum_model)
        @turbo @. y_momentum_eqn.A.nzval = x_momentum_eqn.A.nzval

        apply_boundary_conditions!(x_momentum_eqn, x_momentum_model, uxBCs)
        print("Solving x-momentum. ") # 2 allocations
        alpha_U = 0.8
        implicit_relaxation!(x_momentum_eqn, ux0, alpha_U)
        
        ilu0!(Px, x_momentum_eqn.A)
        run!(
            x_momentum_eqn, x_momentum_model, uxBCs, 
            setup, opA=opAx, opP=opPUx, solver_alloc=solver_U
        ) # 41 allocations

        @turbo @. y_momentum_eqn.b = 0.0

        apply_boundary_conditions!(y_momentum_eqn, y_momentum_model, uyBCs)
        print("Solving y-momentum. ")
        implicit_relaxation!(y_momentum_eqn, uy0, alpha_U)
        ilu0!(Py, y_momentum_eqn.A)
        run!(
            y_momentum_eqn, y_momentum_model, uyBCs, 
            setup, opA=opAy, opP=opPUy, solver_alloc=solver_U
        )

        @turbo for i ∈ eachindex(ux0)
            ux0[i] = U.x[i]
            uy0[i] = U.y[i]
            U.x[i] = ux.values[i]
            U.y[i] = uy.values[i]
        end
        
        inverse_diagonal!(rD, x_momentum_eqn)
        interpolate!(rDf, rD)
        remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
        H_new!(Hv, U, x_momentum_eqn, y_momentum_eqn)
        
        @turbo for i ∈ eachindex(ux0)
            U.x[i] = ux0[i]
            U.y[i] = uy0[i]
        end
        div!(divHv, UBCs) # 7 allocations
        @turbo @. divHv.values *= 1.0./volume

        discretise!(pressure_eqn, pressure_correction)
        apply_boundary_conditions!(pressure_eqn, pressure_correction, pBCs)
        print("Solving pressure correction. ")
        run!(
            pressure_eqn, pressure_correction, pBCs, 
            setup_p, opA=opAp, opP=opPP, solver_alloc=solver_p
        )
        
        source!(∇p, pf, p, pBCs) # 7 allocations
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(U, Hv, ∇p, rD)
        interpolate!(Uf, U, UBCs) # 7 allocations
        
        explicit_relaxation!(p, p0, 0.3)
        source!(∇p, pf, p, pBCs)  # 7 allocations
        # grad!(∇p, pf, p, pBCs) 
        correct_velocity!(ux, uy, Hv, ∇p, rD)
    end # end for loop         
end # end function

GC.gc()
isimple!(
    mesh, velocity, nu, ux, uy, p, 
    uxBCs, uyBCs, pBCs, UBCs,
    setup, setup_p, 100)

write_vtk(mesh, ux)
write_vtk(mesh, uy)
write_vtk(mesh, p)

plotly(size=(400,400), markersize=1, markerstrokewidth=1)
plot(collect(1:100), Rx[1:100], yscale=:log10)
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
    @inbounds @simd for i ∈ eachindex(b)
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

function inverse_diagonal!(rD::ScalarField{I,F}, eqn) where {I,F}
    # D = @view eqn.A[diagind(eqn.A)]
    # @. rD.values = 1.0./D
    A = eqn.A
    rD_values = rD.values
    @inbounds for i ∈ eachindex(rD_values)
        # ap = @view A[i,i]
        # rD_values[i] = 1.0/ap
        rD_values[i] = 1.0/view(A, i, i)[1]
    end
end

function explicit_relaxation!(phi, phi0, alpha)
    # @. phi.values = alpha*phi.values + (1.0 - alpha)*phi0
    # @. phi0 = phi.values
    values = phi.values
    @turbo for i ∈ eachindex(values)
        values[i] = alpha*values[i] + (1.0 - alpha)*phi0[i]
        phi0[i] = values[i]
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    # @. U.x = Hv.x - ∇p.x*rD.values
    # @. U.y = Hv.y - ∇p.y*rD.values
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @turbo for i ∈ eachindex(Ux)
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues[i]
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues[i]
    end
end

function correct_velocity!(ux, uy, Hv, ∇p, rD)
    # @. ux.values = Hv.x - ∇p.x*rD.values
    # @. uy.values = Hv.y - ∇p.y*rD.values
    ux = ux.values; uy = uy.values; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.x; dpdy = ∇p.y; rDvalues = rD.values
    @turbo for i ∈ eachindex(ux)
        ux[i] = Hvx[i] - dpdx[i]*rDvalues[i]
        uy[i] = Hvy[i] - dpdy[i]*rDvalues[i]
    end
end

function negative_vector_source!(∇p)
    # ∇p.x .*= -1.0
    # ∇p.y .*= -1.0
    dpdx = ∇p.x; dpdy = ∇p.y
    @turbo for i ∈ eachindex(dpdx)
        dpdx[i] *= -1.0
        dpdy[i] *= -1.0
    end
end

function remove_pressure_source!(x_momentum_eqn, y_momentum_eqn, ∇p, rD)
    # @. x_momentum_eqn.b -= ∇p.x/rD.values
    # @. y_momentum_eqn.b -= ∇p.y/rD.values
    dpdx, dpdy, rD = ∇p.x, ∇p.y, rD.values
    bx, by = x_momentum_eqn.b, y_momentum_eqn.b
    @turbo for i ∈ eachindex(bx)
        bx[i] -= dpdx[i]/rD[i]
        by[i] -= dpdy[i]/rD[i]
    end
end
