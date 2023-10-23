export _get_float, _get_int

_get_float(mesh) = eltype(mesh.cells[1].centre)
_get_int(mesh) = eltype(mesh.cells[1].facesID)