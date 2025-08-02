# Summation 

Base.:+(t1::Laplacian, t2::Laplacian) = begin
    scheme(args...) = scheme!(t1, args...) .+ scheme!(t2, args...)
    scheme_source(args...) = scheme_source!(t1, args...) .+ scheme!(t2, args...)
    LHS(scheme, scheme_source)
end

Base.:+(t1::LHS, t2::Laplacian) = begin
    scheme(args...) = t1.scheme(args...) .+ scheme!(t2, args...)
    scheme_source(args...) = t1.scheme_source(args...) .+ scheme!(t2, args...)
    LHS(scheme, scheme_source)
end
Base.:+(t1::Laplacian, t2::LHS) = t2 + t1

Base.:-(t1::Laplacian, t2::Laplacian) = begin
    scheme(args...) = scheme!(t1, args...) .- scheme!(t2, args...)
    scheme_source(args...) = scheme_source!(t1, args...) .- scheme!(t2, args...)
    LHS(scheme, scheme_source)
end

Base.:-(t1::LHS, t2::Laplacian) = begin
    scheme(args...) = t1.scheme(args...) .- scheme!(t2, args...)
    scheme_source(args...) = t1.scheme_source(args...) .- scheme!(t2, args...)
    LHS(scheme, scheme_source)
end
Base.:-(t1::Laplacian, t2::LHS) = begin
    scheme(args...) = scheme(t1, args...) .- t2.scheme(args...)
    scheme_source(args...) = scheme_source(t1, args...) .- t2.scheme(args...)
    LHS(scheme, scheme_source)
end

Base.:-(t1::Divergence, t2::Laplacian) = begin
    scheme(args...) = scheme!(t1, args...) .- scheme!(t2, args...)
    scheme_source(args...) = scheme_source!(t1, args...) .- scheme!(t2, args...)
    LHS(scheme, scheme_source)
end

Base.:-(t1::Laplacian, t2::Divergence) = begin
    scheme(args...) = scheme!(t1, args...) .- scheme!(t2, args...)
    scheme_source(args...) = scheme_source!(t1, args...) .- scheme!(t2, args...)
    LHS(scheme, scheme_source)
end

Base.:(==)(t1::Union{Laplacian,Divergence,Si}, n::Number) = begin
    scheme(args...) = scheme!(t1, args...)
    scheme_source(args...) = scheme_source!(t1, args...)
    source(args...) = n 
    return Discretisation(scheme, scheme_source, source)
end


# export →

# Base.:+(a::Operator, b::Operator) = [a, b]
# Base.:+(a::Vector{<:Operator}, b::Operator) = [a..., b]

# Base.:-(a::Operator) = begin
#     @reset a.sign = -1
#     [a]
# end
# Base.:-(a::Operator, b::Operator) = begin
#     @reset b.sign = -1
#     [a, b]
# end
# Base.:-(a::Vector{<:Operator}, b::Operator) = begin
#     @reset b.sign = -1
#     [a..., b]
# end

# Source operations

# Base.:+(a::Src, b::Src) = [a, b]
# Base.:+(a::Vector{<:Src}, b::Src) = [a..., b]

# Base.:-(a::Src) = begin
#     @reset a.sign = -1
#     [a]
# end

# Base.:-(a::Src, b::Src) = begin
#     @reset b.sign = -1
#     [a, b]
# end
# Base.:-(a::Vector{<:Src}, b::Src) = begin
#     @reset b.sign = -1
#     [a..., b]
# end

# Equality operation for model wrapper

# Base.:(==)(a::Operator, b::Src) = begin
#     Model{1,1}((a,),(b,))
# end

# Base.:(==)(a::Vector{<:Operator}, b::Src) = begin
#     Model{length(a),1}((a...,),(b,))
# end

# Base.:(==)(a::Operator, b::Vector{<:Src}) = begin
#     Model{1,length(b)}((a...,),(b...,))
# end

# Base.:(==)(a::Vector{<:Operator}, b::Vector{<:Src}) = begin
#     Model{length(a), length(b)}((a...,),(b...,))
# end

# (→)(model::Model{TN,SN,T,S}, eqn::AbstractEquation) where {TN,SN,T,S}= begin
#     # To-do: Add runtime check to ensure both sides are consistent (for now document)
#     if S.parameters[1].parameters[1] <: AbstractScalarField
#         Equation(ScalarModel(), model, eqn, (), ())
#     elseif S.parameters[1].parameters[1] <: AbstractVectorField
#         Equation(VectorModel(), model, eqn, (), ())
#     end
# end
