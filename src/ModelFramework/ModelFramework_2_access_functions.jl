export get_phi, get_flux
export get_source, get_source_sign
export _A, _b
export _nzval, _rowval, _colptr

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

_A(eqn::ModelEquation) = eqn.equation.A

_b(eqn::ModelEquation) = eqn.equation.b

_nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
_nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

_colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
_colptr(A::SparseArrays.SparseMatrixCSC) = A.colptr

_rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
_rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval 