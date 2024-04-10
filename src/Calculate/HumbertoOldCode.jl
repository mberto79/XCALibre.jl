function wallDistance(
    mesh, BCs; skewnessCorrectors=1, tol = 1e-10, Rtol=1e-4, phiRelax=0.85, iterations=500
    )


    ncells = length(mesh.cells)
    
    phi = FVM.ScalarField(mesh, BCs, 0.0)
    phiPrev = zeros(ncells)
    phif = FVM.FaceScalarField(phi, 0.0)


    phiGrad = FVM.VectorField(phi, SVector(0.0,0.0,0.0))
    phiGradf = FVM.FaceVectorField(phiGrad, SVector(0.0,0.0,0.0))
    phiGrad_Tf = zeros(ncells)


    R = zeros(ncells)


    volumes = getproperty.(mesh.cells, :volume)
    phiEqn = FVM.∇²(1.0, phi) == FVM.S(-volumes)


    solver = BicgstabSolver(phiEqn.A, phiEqn.b)


    print("\nCalculating wall distance...")
    exe_time_start = time()


    for iteration ∈ 1:iterations
        for i ∈ 1:skewnessCorrectors
            phiEqn(FVM.∇²(1.0, phi))
            FVM.applyBoundaryConditions!(phiEqn)
            FVM.updateSources!(phiEqn, -volumes .- phiGrad_Tf.*volumes)
            FVM.solve!(phiEqn, solver, R, tol=Rtol)
            @. phi.values = phiPrev + phiRelax*(phi.values - phiPrev)
            @. phiPrev = phi.values
                    
            # UPDATE CELL AND FACE PROPERTIES
            FVM.interpolate2faces_shortest!(phif, phi)
            FVM.correctBoundary_pf!(phif) # ADDED
            FVM.grad_shortest!(phiGrad, phif)
    
            FVM.interpolate2faces!(phiGradf,phiGrad)
            FVM.correctGradientInterpolation!(phiGradf, phi)
            FVM.correctBoundary_pGradf!(phiGradf)
            FVM.nonOrthogonality_correction!(phiGrad_Tf, phiGradf)


            res = FVM.residual(phiEqn, phi)
            FVM.clear!(phiEqn)
            if res < tol
                print("\nFinal residual = ", res)
                print("\nDone! time: ", time() - exe_time_start )
                return normalDistance(phi, phiGrad)
            end
            end
    end
    return normalDistance(phi, phiGrad)
end


function normalDistance(phi, phiGrad)
    wallDist = deepcopy(phi)
    @inbounds for i ∈ 1:length(phi.values)
        gradMag = norm(phiGrad.values[i])
        wallDist.values[i] = -gradMag + sqrt(gradMag^2 + 2.0*phi.values[i])
    end
    return wallDist
end
