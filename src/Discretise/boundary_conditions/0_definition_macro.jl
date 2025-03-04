
"""
    macro define_boundary(boundary, operator, definition)
        quote
            @inline (bc::\$boundary)(
                term::Operator{F,P,I,\$operator}, cellID, zcellID, cell, face, fID, i, component, time
                ) where {F,P,I} = \$definition
        end |> esc
    end

Macro to reduce boilerplate code when defining boundary conditions (implemented as functors) and provides access to key fields needed in the implementation of boundary conditions, such as the boundary cell and face objects (more details below)

# Input arguments

- `boundary` specifies the boundary type being defined
- `operator` specifies the operator to which the condition applies e.g. `Laplacian`
- `definition` provides the implementation details

# Available fields

- `term` reference to operator on which the boundary applies (gives access to the field and mesh) 
- `cellID` ID of the corresponding boundary cell
- `zcellID` sparse matrix linear index for the cell
- `cell` gives access to boundary cell object and corresponding information
- `face` gives access to boundary face object and corresponding information
- `fID` ID of the boundary face (to index `Mesh2.faces` vector)
- `i` local index of the boundary faces within a kernel or loop
- `component` for vectors this specifies the components being evaluated (access as `component.value`). For scalars `component = nothing`
- `time` provides the current simulation time. This only applies to time dependent boundary implementation defined as functions or neural networks.

# Example

Below the use of this macro is illustrated for the implementation of a  `Dirichlet` boundary condition acting on the `Laplacian` using the `Linear` scheme:

    @define_boundary Dirichlet Laplacian{Linear} begin
        J = term.flux[fID]      # extract operator flux
        (; area, delta) = face  # extract boundary face information
        flux = J*area/delta     # calculate the face flux
        ap = term.sign*(-flux)  # diagonal (cell) matrix coefficient
        ap, ap*bc.value         # return `ap` and `an`
    end

When called, this functor will return two values `ap` and `an`, where `ap` is the cell contribution for approximating the boundary face value, and `an` is the explicit part of the face value approximation i.e. `ap` contributes to the diagonal of the sparse matrix (left-hand side) and `an` is the explicit contribution assigned to the solution vector `b` on the right-hand of the linear system of equations ``Ax = b``

"""
macro define_boundary(boundary, operator, definition)
    quote
        @inline (bc::$boundary)(term::Operator{F,P,I,$operator}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I} = 
        @inbounds begin
            $definition
        end
    end |> esc
end

macro define_boundary(boundary, operator, FieldType, definition)
    quote
        @inline (bc::$boundary)(term::Operator{F,P,I,$operator}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P<:$FieldType,I} = 
        @inbounds begin
            $definition
        end
    end |> esc
end

# macro define_boundary(operator, definition)
#     quote
#         @inline (bc::AbstractBoundary)(term::Operator{F,P,I,Op}, colval, rowptr, nzval, cellID, zcellID, cell, face, fID, i, component, time) where {F,P,I,Op<:$operator} = $definition
#     end |> esc
# end