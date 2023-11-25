export SF_GPU!
export FSF_GPU!
export VF_GPU!
export FVF_GPU!
export TF_GPU!

function SF_GPU!(scalarField)
    (; values, mesh, BCs) = scalarField
    values = cu(values)
    BCs = cu(BCs)
    scalarField = ScalarField(values, mesh, BCs)
end

function FSF_GPU!(faceScalarField)
    (; values, mesh) = faceScalarField
    values = cu(values)
    faceScalarField = FaceScalarField(values, mesh)
end

function VF_GPU!(VF)
    (; x, y, z, mesh, BCs) = VF
    x = SF_GPU!(x)
    y = SF_GPU!(y)
    z = SF_GPU!(z)
    BCs = cu(BCs)
    VF = VectorField(x, y, z, mesh, BCs)
end

function FVF_GPU!(FVF)
    (; x, y, z, mesh) = FVF
    x = FSF_GPU!(x)
    y = FSF_GPU!(y)
    z = FSF_GPU!(z)
    VF = FaceVectorField(x, y, z, mesh)
end

function TF_GPU!(TF)
    (; xx, xy, xz, yx, yy, yz, zx, zy, zz, mesh) = TF
    xx = SF_GPU!(xx)
    xy = SF_GPU!(xy)
    xz = SF_GPU!(xz)
    yx = SF_GPU!(yx)
    yy = SF_GPU!(yy)
    yz = SF_GPU!(yz)
    zx = SF_GPU!(zx)
    zy = SF_GPU!(zy)
    zz = SF_GPU!(zz)
    VF = TensorField(xx, xy, xz, yx, yy, yz, zx, zy, zz, mesh)
end