export HelmholtzEnergyFluid, H2, N2

abstract type HelmholtzEnergyFluid end
struct N2 <: HelmholtzEnergyFluid end
struct H2 <: HelmholtzEnergyFluid end