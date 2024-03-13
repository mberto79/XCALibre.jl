export scheme!, scheme_source!

#= NOTE:
In source scheme the following indices are used and should be used with care:
cID - Index of the cell outer loop. Use to index "b" 
cIndex - Index of the cell based on sparse matrix. Use to index "nzval_array"
=#

# TIME 
# Steady 1
@inline function scheme!(
    term::Operator{F,P,I,Time{Steady}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Steady}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    nothing
end

# # Steady 2
# @inline function scheme!(
#     ::Type{Operator{F,P,I,Time{Steady}}}, t)  where {F,P,I}
#     quote
#         nothing
#     end
# end
# @inline function scheme_source!(
#     ::Type{Operator{F,P,I,Time{Steady}}}, t)  where {F,P,I}
#     quote
#         nothing
#     end
# end

## Euler 1
@inline function scheme!(
    term::Operator{F,P,I,Time{Euler}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime)  where {F,P,I}
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Time{Euler}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
        volume = cell.volume
        rdt = 1/runtime.dt
        nzval_array[cIndex] += volume*rdt
        Atomix.@atomic b[cID] += prev[cID]*volume*rdt
    nothing
end

# ## Euler 2
# @inline function scheme!(
#     ::Type{Operator{F,P,I,Time{Euler}}}, t)  where {F,P,I}
#     quote
#         nothing
#     end
# end
# @inline function scheme_source!(
#     ::Type{Operator{F,P,I,Time{Euler}}}, t)  where {F,P,I}
#     quote
#         # volume = cell.volume
#         # rdt = 1/runtime.dt
#         # nzval_array[cIndex] += volume*rdt
#         # Atomix.@atomic b[cID] += prev[cID]*volume*rdt
#     end
# end

# LAPLACIAN 1

@inline function scheme!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    ap = term.sign*(-term.flux[fID] * face.area)/face.delta
    Atomix.@atomic nzval_array[cIndex] += ap
    Atomix.@atomic nzval_array[nIndex] += -ap
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Laplacian{Linear}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    nothing
end

# # LAPLACIAN 2

# @inline function scheme!(
#     ::Type{Operator{F,P,I,Laplacian{Linear}}}, t)  where {F,P,I}
#     quote
#         term = terms[$t]
#         ap = term.sign*(-term.flux[fID] * face.area)/face.delta
#         Atomix.@atomic nzval_array[cIndex] += ap
#         Atomix.@atomic nzval_array[nIndex] += -ap
#     end
# end
# @inline function scheme_source!(
#     ::Type{Operator{F,P,I,Laplacian{Linear}}}, t)  where {F,P,I} 
#     quote
#         # nothing
#     end
# end

# DIVERGENCE

# linear 1

@inline function scheme!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    xf = face.centre
    xC = cell.centre
    xN = cellN.centre
    weight = norm(xf - xC)/norm(xN - xC)
    one_minus_weight = one(eltype(weight)) - weight
    ap = term.sign*(term.flux[fID]*ns)
    Atomix.@atomic nzval_array[cIndex] += ap*one_minus_weight
    Atomix.@atomic nzval_array[nIndex] += ap*weight
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Linear}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    nothing
end

# # linear 2

# @inline function scheme!(
#     ::Type{Operator{F,P,I,Divergence{Linear}}}, t)  where {F,P,I}
#     quote
#         term = terms[$t]
#         xf = face.centre
#         xC = cell.centre
#         xN = cellN.centre
#         weight = norm(xf - xC)/norm(xN - xC)
#         one_minus_weight = one(eltype(weight)) - weight
#         ap = term.sign*(term.flux[fID]*ns)
#         Atomix.@atomic nzval_array[cIndex] += ap*one_minus_weight
#         Atomix.@atomic nzval_array[nIndex] += ap*weight
#     end
# end
# @inline function scheme_source!(
#     ::Type{Operator{F,P,I,Divergence{Linear}}}, t) where {F,P,I}
#     quote
#         nothing
#     end
# end

# Upwind 1
@inline function scheme!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    nzval_array, cell, face, cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    ap = term.sign*(term.flux[fID]*ns)
    Atomix.@atomic nzval_array[cIndex] += max(ap, 0.0)
    Atomix.@atomic nzval_array[nIndex] += -max(-ap, 0.0)
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Divergence{Upwind}}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime) where {F,P,I} = begin
    nothing
end

# # Upwind 2
# @inline function scheme!(
#     ::Type{Operator{F,P,I,Divergence{Upwind}}}, t)  where {F,P,I}
#     quote
#         term = terms[$t]
#         ap = term.sign*(term.flux[fID]*ns)
#         Atomix.@atomic nzval_array[cIndex] += max(ap, 0.0)
#         Atomix.@atomic nzval_array[nIndex] += -max(-ap, 0.0)
#     end
# end
# @inline function scheme_source!(
#     ::Type{Operator{F,P,I,Divergence{Upwind}}}) where {F,P,I}
#     quote
#         nothing
#     end
# end

# IMPLICIT SOURCE 1

@inline function scheme!(
    term::Operator{F,P,I,Si}, 
    nzval_array, cell, face,  cellN, ns, cIndex, nIndex, fID, prev, runtime
    )  where {F,P,I}
    # ap = term.sign*(-term.flux[cIndex] * cell.volume)
    # nzval_array[cIndex] += ap
    nothing
end
@inline scheme_source!(
    term::Operator{F,P,I,Si}, 
    b, nzval_array, cell, cID, cIndex, prev, runtime)  where {F,P,I} = begin
    phi = term.phi
    # ap = max(flux, 0.0)
    # ab = min(flux, 0.0)*phi[cID]
    flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
    Atomix.@atomic nzval_array[cIndex] += flux # indexed with cIndex
    # flux = term.sign*term.flux[cID]*cell.volume*phi[cID]
    # b[cID] -= flux
    nothing
end

# # IMPLICIT SOURCE 2

# @inline function scheme!(
#     ::Type{Operator{F,P,I,Si}}, t
#     )  where {F,P,I}
#     # ap = term.sign*(-term.flux[cIndex] * cell.volume)
#     # nzval_array[cIndex] += ap
#     quote
#         nothing
#     end
# end
# @inline function scheme_source!(
#     ::Type{Operator{F,P,I,Si}}, t)  where {F,P,I}
#     quote
#         # term = terms[$t]
#         # phi = term.phi
#         # flux = term.sign*term.flux[cID]*cell.volume # indexed with cID
#         # Atomix.@atomic nzval_array[cIndex] += flux # indexed with cIndex
#     end
# end

