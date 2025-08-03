const OPERATORS = Union{Time, Laplacian, Divergence, Si}

# Addition and subtraction

ops = [
    (Symbol(:+), Base.Broadcast.BroadcastFunction(+)),
    (Symbol(:-), Base.Broadcast.BroadcastFunction(-))
]

for (symbol, broadcast) ∈ ops 
    @eval begin
        Base.$symbol(t1::OPERATORS, t2::OPERATORS) = begin
            scheme(args...) = $(broadcast)(
                scheme!(t1, args...), scheme!(t2, args...)
            )
            scheme_source(args...) = $(broadcast)(
                scheme_source!(t1, args...), scheme_source!(t2, args...)
            )
            apply_BCs(args...) = $(broadcast)(
                apply_BCs!(t1, args...), apply_BCs!(t2, args...)
            )
            LHS(scheme, scheme_source, apply_BCs)
        end

        Base.$symbol(t1::LHS, t2::OPERATORS) = begin
            scheme(args...) = $(broadcast)(
                t1.scheme(args...), scheme!(t2, args...)
            )
            scheme_source(args...) = $(broadcast)(
                t1.scheme_source(args...),scheme_source!(t2, args...)
            )
            apply_BCs(args...) = $(broadcast)(
                t1.apply_BCs(args...), apply_BCs!(t2, args...)
            )
            LHS(scheme, scheme_source, apply_BCs)
        end
        Base.$symbol(t1::OPERATORS, t2::LHS) = begin
            scheme(args...) = $(broadcast)(
                scheme!(t1, args...), t2.scheme(args...)
            )
            scheme_source(args...) = $(broadcast)(
                scheme_source!(t1, args...), t2.scheme_source(args...)
            )
            apply_BCs(args...) = $(broadcast)(
                apply_BCs!(t1, args...), t2.apply_BCs(args...)
            )
            LHS(scheme, scheme_source, apply_BCs)
        end
    end
end

Base.:-(t1::OPERATORS) = begin
    scheme(args...) = -1 .* scheme!(t1, args...)
    scheme_source(args...) = -1 .* scheme_source!(t1, args...)
    LHS(scheme, scheme_source, apply_BCs)
end

# Sources 
Base.:-(t1::Source) = begin
    source(args...) = -1 .* source!(t1, args...)
    RHS(source)
end

# Handle the equality

Base.:(==)(t1::OPERATORS, t2::Source) = begin
    scheme(args...) = scheme!(t1, args...)
    scheme_source(args...) = scheme_source!(t1, args...)
    source(args...) = source!(t2, args...)
    return Discretisation(scheme, scheme_source, source)
end

Base.:(==)(t1::LHS, t2::Source) = begin
    scheme(args...) = t1.scheme(args...)
    scheme_source(args...) = t1.scheme_source(args...)
    source(args...) = source!(t2, args...)
    return Discretisation(scheme, scheme_source, source)
end

Base.:(==)(t1::LHS, t2::RHS) = begin
    scheme(args...) = t1.scheme(args...)
    scheme_source(args...) = t1.scheme_source(args...)
    source(args...) = t2.source(args...)
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
