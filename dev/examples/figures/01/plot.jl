using Plots
using DelimitedFiles

OF = readdlm("2D_laminar_BFS_OF.csv", ',', skipstart=1)
XL = readdlm("2D_laminar_BFS_XCALibre.csv", ',', skipstart=1)

vel = plot(
    OF[:,1], OF[:,8], label="Ux", color=:skyblue2,
    ylabel="Distance [m]", xlabel="Velocity [m/s]")
scatter!(vel, XL[:,1], XL[:,8], color=:tomato, label=false)

plot!(vel, OF[:,2], OF[:,8], label="Uy", color=:black, linestyle=:dash)
scatter!(vel, XL[:,2], XL[:,8], color=:tomato, label=false)

p = plot(OF[:,4], OF[:,8], label="OpenFOAM", xlabel="p [Pa]", color=:skyblue2)
scatter!(p, XL[:,4], XL[:,8], color=:tomato, label="XCALibre")

plot(vel, p, size=(750,420), layout = (1, 2), ylim=(-0.1,0.10),foreground_color_legend = nothing)

savefig("BFS_verification.svg")