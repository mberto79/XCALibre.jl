update_nueff!(nueff, nu, turb_model) = begin
    if turb_model === nothing
        for i ∈ eachindex(nueff)
            nueff[i] = nu[i]
        end
    else
        for i ∈ eachindex(nueff)
            nueff[i] = nu[i] + turb_model.νtf[i]
        end
    end
end

update_alphaeff!(alphaeff, mu, turb_model) = begin
    if turb_model === nothing
        for i ∈ eachindex(alphaeff)
            alphaeff[i] = mu[i] / 0.7
        end
    else
        for i ∈ eachindex(alphaeff)
            alphaeff[i] = (mu[i] / 0.7 + turb_model.νtf[i])
        end
    end
end

function residual!(Residual, equation, phi, iteration)
    (; A, b, R, Fx) = equation
    values = phi.values
    # Option 1
    
    mul!(Fx, A, values)
    @inbounds @. R = abs(Fx - b)
    # res = sqrt(mean(R.^2))/abs(mean(values))
    res = sqrt(mean(R.^2))/norm(b)


    # res = max(norm(R), eps())/abs(mean(values))

    # sum_mean = zero(TF)
    # sum_norm = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum_mean += values[i]
    #     sum_norm += abs(Fx[i] - b[i])^2
    # end
    # N = length(values)
    # res = sqrt(sum_norm)/abs(sum_mean/N)

    # Option 2
    # mul!(Fx, opA, values)
    # @inbounds @. R = b - Fx
    # sum = zero(TF)
    # @inbounds for i ∈ eachindex(R)
    #     sum += (R[i])^2 
    # end
    # res = sqrt(sum/length(R))

    # Option 3 (OpenFOAM definition)
    # solMean = mean(values)
    # mul!(R, opA, values .- solMean)
    # term1 = abs.(R)
    # mul!(R, opA, values)
    # Fx .= b .- R.*solMean./values
    # term2 = abs.(Fx)
    # N = sum(term1 .+ term2)
    # res = (1/N)*sum(abs.(b .- R))

    # term1 = abs.(opA*(values .- solMean))
    # term2 = abs.(b .- R.*solMean./values)
    # N = sum(term1 .+ term2)
    # res = (1/N)*sum(abs.(b - opA*values))

    # print("Residual: ", res, " (", niterations(solver), " iterations)\n") 
    # @printf "\tResidual: %.4e (%i iterations)\n" res niterations(solver)
    # return res
    Residual[iteration] = res
    nothing
end

function flux!(phif::FS, psif::FV) where {FS<:FaceScalarField,FV<:FaceVectorField}
    (; mesh, values) = phif
    (; faces) = mesh 
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        Sf = area*normal
        values[fID] = psif[fID]⋅Sf
    end
end

function flux!(phif::FS, psif::FV, rhof::FS) where {FS<:FaceScalarField,FV<:FaceVectorField,}
    (; mesh, values) = phif
    (; faces) = mesh 
    rhofvalues = rhof.values
    @inbounds for fID ∈ eachindex(faces)
        (; area, normal) = faces[fID]
        Sf = area*normal
        values[fID] = (psif[fID]⋅Sf)*rhofvalues[fID]
    end
end

volumes(mesh) = [mesh.cells[i].volume for i ∈ eachindex(mesh.cells)]

function inverse_diagonal!(rD::S, eqn) where S<:ScalarField
    (; mesh, values) = rD
    cells = mesh.cells
    A = eqn.A
    @inbounds for i ∈ eachindex(values)
        # D = view(A, i, i)[1]
        D = A[i,i]
        volume = cells[i].volume
        values[i] = volume/D
    end
end

function correct_velocity!(U, Hv, ∇p, rD)
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDvalues = rD.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDvalues_i = rDvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDvalues_i
    end
end

function correct_velocity_vec!(U, Hv, ∇p, rDx, rDy)
    Ux = U.x; Uy = U.y; Hvx = Hv.x; Hvy = Hv.y
    dpdx = ∇p.result.x; dpdy = ∇p.result.y; rDxvalues = rDx.values; rDyvalues = rDy.values
    @inbounds @simd for i ∈ eachindex(Ux)
        rDxvalues_i = rDxvalues[i]
        rDyvalues_i = rDyvalues[i]
        Ux[i] = Hvx[i] - dpdx[i]*rDxvalues_i
        Uy[i] = Hvy[i] - dpdy[i]*rDyvalues_i
    end
end

remove_pressure_source!(ux_eqn::M1, uy_eqn::M2, ∇p) where {M1,M2} = begin # Extend to 3D
    cells = get_phi(ux_eqn).mesh.cells
    source_sign = get_source_sign(ux_eqn, 1)
    dpdx, dpdy = ∇p.result.x, ∇p.result.y
    bx, by = ux_eqn.equation.b, uy_eqn.equation.b
    @inbounds for i ∈ eachindex(bx)
        volume = cells[i].volume
        bx[i] -= source_sign*dpdx[i]*volume
        by[i] -= source_sign*dpdy[i]*volume
    end
end

H!(Hv, v::VF, ux_eqn, uy_eqn) where VF<:VectorField = 
begin # Extend to 3D!
    (; x, y, z, mesh) = Hv 
    (; cells, faces) = mesh
    Ax = ux_eqn.equation.A; Ay = uy_eqn.equation.A
    bx = ux_eqn.equation.b; by = uy_eqn.equation.b
    vx, vy = v.x, v.y
    F = eltype(v.x.values)
    @inbounds for cID ∈ eachindex(cells)
        cell = cells[cID]
        (; neighbours, volume) = cell
        sumx = zero(F)
        sumy = zero(F)
        @inbounds for nID ∈ neighbours
            sumx += Ax[cID,nID]*vx[nID]
            sumy += Ay[cID,nID]*vy[nID]
        end

        Dx = view(Ax, cID, cID)[1] # add check to use max of Ax or Ay)
        rDx = 1.0/Dx
        Dy = view(Ay, cID, cID)[1] # add check to use max of Ax or Ay)
        rDy = 1.0/Dy
        # rD = volume/D
        x[cID] = (bx[cID] - sumx)*rDx
        y[cID] = (by[cID] - sumy)*rDy
        z[cID] = zero(F)
    end
end

courant_number(U, mesh::Mesh2, runtime) = begin
    dt = runtime.dt 
    co = zero(_get_float(mesh))
    # courant_max = zero(_get_float(mesh))
    cells = mesh.cells
    for i ∈ eachindex(U)
        umag = norm(U[i])
        volume = cells[i].volume
        dx = sqrt(volume)
        co = max(co, umag*dt/dx)
    end
    return co
end

function correct_Ψ(Ψ, h)
    (; values, mesh) = Ψ
    (; cells, faces) = mesh
    hvalues = h.values
    R = 287 # For air perfect gas
    Cp = 1005 # For air perfect gas
    @inbounds for cID ∈ eachindex(cells)
        T = h.values / Cp
        values[cID] = 1/(R*T)
    end
end