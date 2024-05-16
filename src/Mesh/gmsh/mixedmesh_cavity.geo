//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {0, 1, 0, 1.0};
//+
Point(4) = {1, 1, 0, 1.0};
//+
Point(5) = {0, 2, 0, 1.0};
//+
Point(6) = {1, 2, 0, 1.0};
//+
Line(1) = {1, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 2};
//+
Line(4) = {2, 1};
//+
Line(5) = {3, 5};
//+
Line(6) = {6, 4};
//+
Line(7) = {5, 6};
//+
//+
Transfinite Curve {5, 2, 6, 7} = 40 Using Progression 1;
//+
Transfinite Curve {2, 1, 4, 3} = 40 Using Progression 1;
//+
Curve Loop(1) = {7, 6, -2, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Plane Surface(2) = {2};
//+


//+
Extrude {0, 0, 1} {
  Curve{7}; Point{5}; Curve{5}; Surface{1}; Curve{6}; Curve{2}; Point{4}; Point{3}; Curve{1}; Surface{2}; Curve{3}; Point{2}; Curve{4}; Point{1}; Layers{2}; Recombine;
}
//+
Transfinite Surface {37};
//+
Transfinite Surface {1};
//+
Transfinite Surface {37};
//+
Transfinite Surface {1};
//+
Transfinite Volume{1} = {7, 5, 6, 8, 9, 3, 4, 15};
//+
Recombine Surface {37, 1};
//+
Physical Surface("symmetry") = {37, 1, 2, 63};
//+
Physical Surface("wall") = {28, 58, 62, 41, 15};
//+
Physical Surface("moving_wall") = {11};
//+
Physical Volume("fluid") = {1, 2};

