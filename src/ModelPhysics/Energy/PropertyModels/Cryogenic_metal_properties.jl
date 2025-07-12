export get_coefficients

# Only Aluminium, Steel, and Copper are supported (copper is dodgy)



const K_COEFFS = Dict{Symbol,NTuple{9,Float64}}(
  :Aluminium    => (  0.07918,   1.09570,  -0.07277,   0.08084,   0.02803, #Al6061_T6_Aluminum
                           -0.09464,   0.04179,  -0.00571,   0.0     ),

  :Steel        => ( -1.4087,    1.3982,    0.2543,   -0.6260,    0.2334, #SS304
                            0.4256,   -0.4658,    0.1650,   -0.0199  ),

#   :Nickel       => ( -8.28921,  39.4470,  -83.4353,   98.1690,  -67.2088, #Inconel718
#                            26.7082,   -5.72050,   0.51115,   0.0     ),

  :Copper       => ( -0.50015,   1.93190,  -1.69540,   0.71218,   1.27880, #Beryllium_Copper
                           -1.61450,   0.68722,  -0.10501,   0.0     ),

#   :Titanium     => (-5107.8774,19240.422,-30789.064, 27134.756,-14226.379, #Ti6Al4V
#                             4438.2154,-763.07767,  55.796592, 0.0     )
)

const CP_COEFFS = Dict{Symbol,NTuple{9,Float64}}(
  :Copper        => ( -1.91844,   -0.15973,    8.61013,  -18.99640,   21.96610, #OFHC_Copper
                           -12.73280,    3.54322,   -0.37970,    0.0     ),

  :Aluminium     => (  46.6467,  -314.292,    866.662,  -1298.30,   1162.27, #Al6061_T6_Aluminum
                           -637.795,   210.351,   -38.3094,    2.96344 ),

  :Steel         => (  22.0061,  -127.5528,   303.6470,  -381.0098,   274.0328, #SS304
                           -112.9212,   24.7593,   -2.239153,   0.0     ),

#   :Fiberglass    => (  -2.4083,     7.6006,    -8.2982,     7.3301,    -4.2386, #G10
#                               1.4294,   -0.24396,    0.015236,   0.0     ),

#   :Teflon        => (  31.8825,  -166.519,    352.019,   259.981,   -104.614, #Teflon
#                              24.9927,    -3.20792,   0.165032,   0.0     )
)


function get_coefficients(material::Symbol, T_field::ScalarField)
    @assert haskey(K_COEFFS, material) "Unknown material: $material"
    @assert haskey(CP_COEFFS, material) "Unknown material: $material"

    logT = log10.(T_field.values)
    k_coeffs  = K_COEFFS[material]
    cp_coeffs = CP_COEFFS[material]

    k_log10 = k_coeffs[1] .+ k_coeffs[2] .* logT .+ k_coeffs[3] .* logT.^2 .+
              k_coeffs[4] .* logT.^3 .+ k_coeffs[5] .* logT.^4 .+ k_coeffs[6] .* logT.^5 .+
              k_coeffs[7] .* logT.^6 .+ k_coeffs[8] .* logT.^7 .+ k_coeffs[9] .* logT.^8

    cp_log10 = cp_coeffs[1] .+ cp_coeffs[2] .* logT .+ cp_coeffs[3] .* logT.^2 .+
               cp_coeffs[4] .* logT.^3 .+ cp_coeffs[5] .* logT.^4 .+ cp_coeffs[6] .* logT.^5 .+
               cp_coeffs[7] .* logT.^6 .+ cp_coeffs[8] .* logT.^7 .+ cp_coeffs[9] .* logT.^8

    k = 10.0 .^ k_log10
    cp = 10.0 .^ cp_log10

    return k, cp
end