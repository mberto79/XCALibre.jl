

struct kOmegaCoefficients{T}
    β⁺::T
    α1::T
    β1::T
    σκ::T
    σω::T
end

Dk(β⁺, k, ω) = β⁺.*k.values.*ω.values