using Plots
using CSV

iter_2 = CSV.File("mesh_study/c8 cd1_2.csv")
iter_2_x = iter_2.var"Points:1"
iter_2_z_w = iter_2_x/0.61
iter_2_h = iter_2.var"h"*1e6

iter_4 = CSV.File("mesh_study/c8 i4.csv")
iter_4_h = iter_4.var"h"*1e6

iter_6 = CSV.File("mesh_study/c8 i6.csv")
iter_6_h = iter_6.var"h"*1e6

iter_8 = CSV.File("mesh_study/c8 i8.csv")
iter_8_h = iter_8.var"h"*1e6

plot(iter_2_z_w, iter_2_h, label="2 inner loops", legend=:bottom)
plot!(iter_2_z_w, iter_4_h, label="4 inner loops")
plot!(iter_2_z_w, iter_6_h, label="6 inner loops")
plot!(iter_2_z_w, iter_8_h, label="8 inner loops")
xlabel!("z/w")
ylabel!("h (μm)")