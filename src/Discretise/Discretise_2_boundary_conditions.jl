export dirichlet, neumann

# LAPLACIAN TERM (CONSTANT)

@inline (bc::Dirichlet{V})(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where {V,T<:AbstractFloat} = begin
    J = term.J
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    # ap = term.sign[1]*(-term.J*face.area/face.delta)
    A[cellID,cellID] += ap
    b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann{V})(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where {V,T<:AbstractFloat} = begin
    phi = term.phi 
    values = phi.values
    A[cellID,cellID] += 0.0
    b[cellID] += 0.0
    nothing
end

# LAPLACIAN TERM (NON-UNIFORM)

@inline (bc::Dirichlet{V})(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where {V,T<:AbstractScalarField} = begin
    J = term.J(fID)
    (; area, delta) = face 
    flux = J*area/delta
    ap = term.sign[1]*(-flux)
    # @time ap = term.sign[1]*(-term.J(fID)*face.area/face.delta)
    A[cellID,cellID] += ap
    b[cellID] += ap*bc.value
    nothing
end

@inline (bc::Neumann{V})(
    term::Laplacian{Linear, T}, A, b, cellID, cell, face, fID
    ) where {V,T<:AbstractScalarField} = begin
    phi = term.phi 
    values = phi.values
    # A[cellID,cellID] += 0.0*term.sign[1]*(-term.J(fID)*face.area/face.delta)
    # @time b[cellID] += 0.0*term.sign[1]*(-term.J(fID)*face.area/face.delta*values[cellID])
    A[cellID,cellID] += 0.0
    b[cellID] += 0.0
    nothing
end

# DIVERGENCE TERM (CONSTANT)

@inline (bc::Dirichlet{V})(
    term::Divergence{Linear, SVector{3, Float64}}, A, b, cellID, cell, face, fID
    ) where {V}= begin
    A[cellID,cellID] += 0.0 #term.sign[1]*(term.J⋅face.normal*face.area)/2.0
    b[cellID] += term.sign[1]*(-term.J⋅face.normal*face.area*bc.value)
    nothing
end

@inline (bc::Neumann{V})(
    term::Divergence{Linear, SVector{3, Float64}}, A, b, cellID, cell, face, fID
    ) where {V} = begin
    ap = term.sign[1]*(term.J⋅face.normal*face.area)
    A[cellID,cellID] += ap # 2.0*ap
    # b[cellID] += 0.0 # need to extend for gradients other than zero
    nothing
end

# DIVERGENCE TERM (NON-UNIFORM)

@inline (bc::Dirichlet{V})(
    term::Divergence{Linear, FaceScalarField{I,F}}, A, b, cellID, cell, face, fID
    # term::Divergence{Linear, FaceVectorField{I,F}}, A, b, cellID, cell, face, fID
    ) where {V,I,F} = begin
    A[cellID,cellID] += 0.0 #term.sign[1]*(term.J⋅face.normal*face.area)/2.0
    # b[cellID] += term.sign[1]*(term.J(fID)⋅face.normal*face.area*bc.value)
    b[cellID] += term.sign[1]*(-term.J(fID)*bc.value)
    nothing
end

@inline (bc::Neumann{V})(
    term::Divergence{Linear, FaceScalarField{I,F}}, A, b, cellID, cell, face, fID
    # term::Divergence{Linear, FaceVectorField{I,F}}, A, b, cellID, cell, face, fID
    ) where {V,I,F} = begin
    phi = term.phi 
    # values = phi.values

    ap = term.sign[1]*(term.J(fID))
    # ap = term.sign[1]*(term.J(fID)⋅face.normal*face.area)
    A[cellID,cellID] += ap
    # b[cellID] += 0.0 # need to extend for gradients other than zero
    nothing
end