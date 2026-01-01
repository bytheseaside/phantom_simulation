// ============================================================================
// CONFIG 2: COARSE WITH REFINEMENT (near r0 "electrode" and shells)
// ============================================================================
SetFactory("OpenCASCADE");
General.Terminal = 1;

Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Config 2: COARSE WITH REFINEMENT");
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);


//* Volumes
int_vol_id[]  = { 1 };  
ext_vol_id[]  = { 2 };  
//* Surfaces
r0_surfaces[] = { 2 }; // r0 = 2 mm
r1_surfaces[] = { 1 }; // r1 = 80 mm
r2_surfaces[] = { 3 }; // r2 = 83.20 mm

Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 1;
Mesh.SecondOrderLinear = 1;

Mesh.CharacteristicLengthMin = 0.0001;  // 0.1 mm
Mesh.CharacteristicLengthMax = 0.005;   // 5.0 mm - coarse bulk

Mesh.MeshSizeFromCurvature = 25;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;

// Refinement near r0 "electrode"
Field[1] = Distance;
Field[1].SurfacesList = { r0_surfaces[] };
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.0001;    // 0.1 mm near electrode
Field[2].SizeMax = 0.005;     // 5.0 mm far from electrode
Field[2].DistMin = 0.003;     // ≤ 3 mm → SizeMin
Field[2].DistMax = 0.008;     // 3-8 mm → transition

// Refinement near shell interfaces (r1 and r2)
Field[3] = Distance;
Field[3].SurfacesList = { r1_surfaces[], r2_surfaces[] };
Field[3].NumPointsPerCurve = 100;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = 0.0005;    // 0.5 mm near shells
Field[4].SizeMax = 0.005;     // 5.0 mm far from shells
Field[4].DistMin = 0.003;     // ≤ 3 mm → SizeMin
Field[4].DistMax = 0.008;     // 3-8 mm → transition

// Combine refinements
Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

Physical Volume("int", 1) = { int_vol_id[] };
Physical Volume("ext", 2) = { ext_vol_id[] };
Physical Surface("r0", 3) = { r0_surfaces[] };
Physical Surface("r1", 5) = { r1_surfaces[] };
Physical Surface("r2", 4) = { r2_surfaces[] };

Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written: COARSE WITH REFINEMENT");
