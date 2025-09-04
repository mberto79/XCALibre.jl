using XCALibre

using Test


### 4 pressure test points for H2 and N2:

# H2: [1, 3, 10, 12.858] bar
# N2: [1, 5, 15, 33.958] bar

H2_pressure_configs = [1.0, 3.0, 10.0, 12.858]
N2_pressure_configs = [1.0, 5.0, 15.0, 33.958]


### Temperature ranges:

# H2: [14 to 40 K with 0.25 K step]
# H2: [40 to 100 K with 1 K step]
# H2: [100 to 500 K with 10 K step]

# N2: [64 to 90 K with 0.25 K step]
# N2: [90 to 150 K with 1 K step]
# N2: [150 to 500 K with 10 K step]

# H2_instance = HelmholtzEnergy(name=H2())
H2_instance = HelmholtzEnergy(name=H2())
# N2_instance = XCALibre.ModelPhysics.HelmholtzEnergy(name=N2())

trial = H2_instance(100.0, 1.0e5)

println(trial)



# Compare data for 8 pressure test points
# For each property (density, viscosity, conductivity), store highest deviation
# Select highest deviating test point
# Make sure deviation is < 3 or 5 % for each of the three properties thus test is passed