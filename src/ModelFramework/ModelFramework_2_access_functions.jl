export get_phi, get_flux
export get_source, get_source_sign
export _A, _A0, _b
export _nzval, _rowval, _colptr
export get_sparse_fields
export XDir, YDir, ZDir, get_values

# Components
struct XDir{T} 
    value::T
end
Adapt.@adapt_structure XDir
XDir() = XDir(1)

struct YDir{T}  
    value::T
end
Adapt.@adapt_structure YDir
YDir() = YDir(2)

struct ZDir{T}
    value::T
end
Adapt.@adapt_structure ZDir
ZDir() = ZDir(3)

## MODEL ACCESSORS

get_phi(eqn::ModelEquation)  = begin 
    eqn.model.terms[1].phi
end

get_flux(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.terms[ti].flux
end

get_source(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.sources[ti].field
end

get_source_sign(eqn::ModelEquation, ti::Integer) = begin 
    eqn.model.sources[ti].sign
end

## SPARSE MATRIX ACCESSORS

# Access Scalar Model Equation
_A(eqn::ModelEquation{T,M,E,S,P}) where {T<:ScalarModel,M,E,S,P} = eqn.equation.A
_b(eqn::ModelEquation{T,M,E,S,P}) where {T<:ScalarModel,M,E,S,P} = eqn.equation.b
_b(eqn::ModelEquation{T,M,E,S,P},c::Nothing) where {T<:ScalarModel,M,E,S,P} = eqn.equation.b

# Access Vector Model Equation
_A0(eqn::ModelEquation{T,M,E,S,P}) where {T<:VectorModel,M,E,S,P} = eqn.equation.A0
_A(eqn::ModelEquation{T,M,E,S,P}) where {T<:VectorModel,M,E,S,P} = eqn.equation.A
_b(eqn::ModelEquation{T,M,E,S,P}, c::XDir) where {T<:VectorModel,M,E,S,P} = eqn.equation.bx
_b(eqn::ModelEquation{T,M,E,S,P}, c::YDir) where {T<:VectorModel,M,E,S,P} = eqn.equation.by
_b(eqn::ModelEquation{T,M,E,S,P}, c::ZDir) where {T<:VectorModel,M,E,S,P} = eqn.equation.bz

_nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
_nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

_colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
_colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr

_rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
_rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval

get_sparse_fields(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = begin
    A.nzVal, A.rowVal, A.colPtr
end

get_sparse_fields(A::SparseArrays.SparseMatrixCSC) = begin
    A.nzval, A.rowval, A.colptr
end

get_values(phi::ScalarField, component::Nothing) = phi.values
get_values(psi::VectorField, component::XDir) = psi.x.values
get_values(psi::VectorField, component::YDir) = psi.y.values
get_values(psi::VectorField, component::ZDir) = psi.z.values