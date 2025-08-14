export AbstractMaterial, Aluminium, Steel, Copper
export get_coefficients, MaterialCoefficients, material_coefficients


abstract type AbstractMaterial end

struct Aluminium <: AbstractMaterial end
struct Steel <: AbstractMaterial end
struct Copper <: AbstractMaterial end


@kwdef struct MaterialCoefficients{T<:AbstractFloat}
  c1::T
  c2::T
  c3::T
  c4::T
  c5::T
  c6::T
  c7::T
  c8::T
  c9::T
end

(coeffs::MaterialCoefficients)(T) = begin
  return coeffs.c1 .+ coeffs.c2 .* T .+ coeffs.c3 .* T.^2 .+ coeffs.c4 .* T.^3 .+ coeffs.c5 .* T.^4 .+
          coeffs.c6 .* T.^5 .+ coeffs.c7 .* T.^6 .+ coeffs.c8 .* T.^7 .+ coeffs.c9 .* T.^8
end

function material_coefficients(material::Aluminium)
  k_coeffs = MaterialCoefficients(
      c1=0.07918, c2=1.09570, c3=-0.07277, c4=0.08084, c5=0.02803,
      c6=-0.09464, c7=0.04179, c8=-0.00571, c9=0.0
  )
  cp_coeffs = MaterialCoefficients(
      c1=46.6467, c2=-314.292, c3=866.662, c4=-1298.30, c5=1162.27,
      c6=-637.795, c7=210.351, c8=-38.3094, c9=2.96344
  )
  return k_coeffs, cp_coeffs
end

function material_coefficients(material::Steel)
  k_coeffs = MaterialCoefficients(
      c1=-1.4087, c2=1.3982, c3=0.2543, c4=-0.6260, c5=0.2334,
      c6=0.4256, c7=-0.4658, c8=0.1650, c9=-0.0199
  )
  cp_coeffs = MaterialCoefficients(
      c1=22.0061, c2=-127.5528, c3=303.6470, c4=-381.0098, c5=274.0328,
      c6=-112.9212, c7=24.7593, c8=-2.239153, c9=0.0
  )
  return k_coeffs, cp_coeffs
end

function material_coefficients(material::Copper)
  k_coeffs = MaterialCoefficients(
      c1=-0.50015, c2=1.93190, c3=-1.69540, c4=0.71218, c5=1.27880,
      c6=-1.61450, c7=0.68722, c8=-0.10501, c9=0.0
  )
  cp_coeffs = MaterialCoefficients(
      c1=-1.91844, c2=-0.15973, c3=8.61013, c4=-18.99640, c5=21.96610,
      c6=-12.73280, c7=3.54322, c8=-0.37970, c9=0.0
  )
  return k_coeffs, cp_coeffs
end

function get_coefficients(material::AbstractMaterial, T_field::ScalarField)
  k_coeffs, cp_coeffs = material_coefficients(material)

  logT = log10.(T_field.values)

  k_log10 = k_coeffs(logT)
  cp_log10 = cp_coeffs(logT)

  k = 10.0 .^ k_log10
  cp = 10.0 .^ cp_log10

  return k, cp
end