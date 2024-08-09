export Periodic

struct Periodic{I,V} <: AbstractDirichlet
    ID::I
    values::V
end
Adapt.@adapt_structure Periodic

Periodic(patch1::Symbol, patch2::Symbol) = begin
    BC1 = Periodic(1,1)
    BC2 = Periodic(2,2)
    BC1, BC2
end