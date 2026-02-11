using CSV
using Plots

file_viscous = CSV.File("Viscosity included.csv");
U_Viscous = file_viscous.var"U:0";
x_coords = file_viscous.var"Points:0";

file_inviscid = CSV.File("Viscosity not included.csv");
U_inviscid = file_inviscid.var"U:0";

file_viscous_0001 = CSV.File("Viscosity included h=0.001.csv");
U_Viscous_0001 = file_viscous_0001.var"U:0";

g = 9.81;
u0 = 0.004;
y(x)=sqrt(2*g*x+u0^2);

plot(x_coords, U_Viscous, label="h = 0.01m, viscous")
plot!(x_coords, U_inviscid, label="inviscid")
plot!(x_coords, U_Viscous_0001, label="h = 0.001m, viscous")
plot!(x_coords, y, label="mathematical model")