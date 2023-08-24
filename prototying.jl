E = 5.2
kappa = 0.41
cmu=0.09

yL = 11
for i ∈ 1:10
    println(yL)
    yL = log(yL)/kappa + B
end

nut = sqrt(1/sqrt(cmu))