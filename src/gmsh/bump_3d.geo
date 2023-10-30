ls = 2;
Xi = 100; // um
Xo = 100; // um
L = 100.0; // um
x0 = Xi + L/2.0;
R = 50.0;   // um
f0 = 0.3;   // 0--1

Z = 5;

Point(1) = {0, 0, 0, ls};
Point(2) = {Xi, 0, 0, ls};
Point(3) = {Xi, R, 0, ls};
Point(4) = {0, R, 0, ls};

Point(5) = {Xi + L, 0, 0, ls};
Point(6) = {Xi + L + Xo, 0, 0, ls};
Point(7) = {Xi + L + Xo, R, 0, ls};
Point(8) = {Xi + L, R, 0, ls};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {2, 5};

pList[0] = 3; // First point label
nPoints = 21; // Number of discretization points (top-right point of the inlet region)
For i In {1 : nPoints}
  x = Xi + L*i/(nPoints + 1);
  pList[i] = newp;
  Point(pList[i]) = {x,
                ( R * (1 - f0/2 *(1 + Cos(2.0*Pi * (x-x0)/L) ) )),
                0,
   ls};
EndFor
pList[nPoints+1] = 8; // Last point label (top-left point of the outlet region)

Spline(newl) = pList[];


Transfinite Line {9, 10} = Ceil(L/ls) Using Progression 1;
Transfinite Line {4, -2, 8, -6} = Ceil(R/ls) Using Progression 1.1;
Transfinite Line {1, 3} = Ceil(Xi/ls) Using Progression 1;
Transfinite Line {5, 7} = Ceil(Xo/ls) Using Progression 1;

Line Loop(11) = {4, 1, 2, 3};
Plane Surface(12) = {11};
Line Loop(13) = {2, 10, 8, -9};
Plane Surface(14) = {13};
Line Loop(15) = {8, 5, 6, 7};
Plane Surface(16) = {15};
Transfinite Surface {14,12,16};
Recombine Surface {14,12,16};

Extrude {0,0,Z} {
  Surface{14,12,16}; Layers{1}; Recombine;
}
Coherence;

Physical Surface("symmetryLine") = {51, 37, 73};
Physical Surface("frontAndBack") = {60, 38, 82, 16, 14, 12};
Physical Surface("wall") = {59, 29, 81};
Physical Surface("inlet") = {47};
Physical Surface("outlet") = {77};
Physical Volume("volume") = {2, 1, 3};