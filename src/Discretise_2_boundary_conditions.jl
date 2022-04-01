export dirichlet, neumann

dirichlet(term::Laplacian{Linear}, cell, face, value) = begin
    ap! = term.sign[1]*(-term.J*face.area/face.delta)
    b!  = term.sign[1]*(-term.J*face.area/face.delta*value)
    return ap!, b!
end

neumann() = begin
    nothing
end