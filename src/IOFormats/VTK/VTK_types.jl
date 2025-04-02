export VTKWriter3D
export VTKWriter2D

struct VTKWriter3D{H,F}
    header::H
    footer::F
end

struct VTKWriter2D{H,F}
    header::H
    footer::F
end