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
y_scaled(x)=0.7*sqrt(2*g*x+u0^2);

plot(x_coords, U_Viscous, label="h = 0.01m, viscous")
plot!(x_coords, U_inviscid, label="inviscid")
plot!(x_coords, U_Viscous_0001, label="h = 0.001m, viscous")
plot!(x_coords, y, label="inviscid mathematical model")
plot!(x_coords, y_scaled, label="scaled inviscid mathematical model")


file_h0_00015 = CSV.File("h-0.00015.csv");
U_h_0_00015 = file_h0_00015.var"U:0";
x_coords = file_h0_00015.var"Points:0";

file_h0_00025 = CSV.File("h-0.00025.csv");
U_h_0_00025 = file_h0_00025.var"U:0";

plot(x_coords, U_h_0_00015, label="h=0.00015m")
plot!(x_coords, U_h_0_00025, label="h=0.00025m")


file_inviscid_2 = CSV.File("Viscosity not-included 500.csv");
U_IV_500 = file_inviscid_2.var"U:0";
x_coords = file_inviscid_2.var"Points:0";

file_viscous_2 = CSV.File("Viscosity included 500.csv");
U_V_500 = file_viscous_2.var"U:0";

plot(x_coords, U_V_500, label="Inviscid")
plot!(x_coords, U_IV_500, label="Viscous")