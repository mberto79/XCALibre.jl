struct Test0{T}
    S::ScalarField{I,F} where {I,F}
end

struct Test1{T}
    S::ScalarField{I,F} where {I,F}
end


t = Test1{Linear}(phi)