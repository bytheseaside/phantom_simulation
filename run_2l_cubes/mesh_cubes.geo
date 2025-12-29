SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1, from the XAO_PATH environment variable
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs
//* Volumes
outer_volume_id[]  = { 1 };  
inner_volume_id[]  = { 2 };  
//* Surfaces
t_surfaces[] = { 30, 29, 28, 27, 26 };
r_surfaces[] = { 14, 13, 12 };
s_surfaces[] = { 17 };


// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 1;        // Basic optimization

Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.0001;  
Mesh.CharacteristicLengthMax = 0.0005;  

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;


// 3) Physical groups used in solver
Physical Volume("inner", 1) = { inner_volume_id[] };
Physical Volume("outer", 2) = { outer_volume_id[] };
Physical Surface("R", 3) = { r_surfaces[] };
Physical Surface("T", 4) = { t_surfaces[] };
Physical Surface("S", 5) = { s_surfaces[] };
// 6) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");