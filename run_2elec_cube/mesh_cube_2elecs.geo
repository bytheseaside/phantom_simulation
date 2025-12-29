SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1, from the XAO_PATH environment variable
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

//* Volumes
inner_cube_volume[]  = { 1 };
outer_cube_volume[]  = { 2 };
//* Surfaces
jack1_t_surfaces[] = { 32, 31, 30, 29, 28 };
jack1_r_surfaces[] = { 26, 25, 24 };
jack1_s_surfaces[] = { 2 };

jack2_t_surfaces[] = { 21, 20, 19, 18, 17 };
jack2_r_surfaces[] = { 15, 14, 13 };
jack2_s_surfaces[] = { 5 };


// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 1;        // Basic optimization

Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.0005;  
Mesh.CharacteristicLengthMax = 0.0015;  

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;


// 3) Physical groups used in solver
Physical Volume("inner", 1) = { inner_cube_volume[] };
Physical Volume("outer", 2) = { outer_cube_volume[] };

Physical Surface("T1", 3) = { jack1_t_surfaces[] };
Physical Surface("R1", 4) = { jack1_r_surfaces[] };
Physical Surface("S1", 5) = { jack1_s_surfaces[] };
Physical Surface("T2", 6) = { jack2_t_surfaces[] };
Physical Surface("R2", 7) = { jack2_r_surfaces[] };
Physical Surface("S2", 8) = { jack2_s_surfaces[] };

// 4) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");