# OF decomposed-case format (verified vs OpenFOAM-9 source + real v1712 case, 2026-07-03)

## boundary file (processor<N>/constant/polyMesh/boundary)
Physical patches first, processor patches LAST, ascending neighbProcNo:
```
procBoundary0to1
{
    type            processor;
    inGroups        1(processor);
    nFaces          44;
    startFace       7739;
    matchTolerance  0.0001;
    transform       unknown;
    myProcNo        0;
    neighbProcNo    1;
}
```
Name = procBoundary<myProcNo>to<neighbProcNo>. matchTolerance/transform literal.

## Ordering/ownership rules
- Lower rank owns the patch and drives face ordering; both sides must list shared faces
  index-aligned with opposite normals (XCAL ProcessorPatch: both sides sorted by orig
  face id -> aligned ✓).
- Face normal must point OUT of the local owned cell (flip node order when local
  ownerCells[1] is the ghost).
- Internal faces: OF upper-triangular (owner<neighbour, sorted by owner then neighbour).

## Addressing files (needed by reconstructPar only; class labelList, in constant/polyMesh)
- cellProcAddressing / pointProcAddressing: 0-based global index, plain.
- faceProcAddressing: 1-based + turning sign: globalFace = |x|-1; x<0 means face reversed
  vs global mesh (local owner is global neighbour).
- boundaryProcAddressing: per local patch, index into global patch list, -1 for procBoundary.

## Field files: processor patch entry
```
procBoundary0to1
{
    type            processor;
    value           uniform (0 0 0);  // value REQUIRED, matches field rank
}
```
Sources: processorPolyPatch.C (newName/owner), domainDecomposition.C (turning index),
avankit/openfoam damBreak processor0 real case.
