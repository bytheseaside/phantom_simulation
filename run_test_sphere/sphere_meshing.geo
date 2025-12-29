SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1, from the XAO_PATH environment variable
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs
//* Volumes
sphere_volume_ids[]  = { 1 };  
//* Surfaces
internal_surfaces[] = { 2 };
external_surfaces[] = { 1 };


// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
// Mesh.HighOrderOptimize = 2;     // Advanced optimization of high-order nodes -> RAM killer
Mesh.HighOrderOptimize = 1;        // Basic optimization

Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.0005;  // 0.5 mm — minimum mesh size
Mesh.CharacteristicLengthMax = 0.002;   // 2.0 mm — maximum mesh size in interior gel
Mesh.MeshSizeFromCurvature  = 25;       // Curvature-driven refinement sensitivity

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;


// 3) Physical groups used in solver
Physical Volume("sphere", 1) = { sphere_volume_ids[] };

Physical Surface("r_int", 2) = { internal_surfaces[] };
Physical Surface("r_ext", 3) = { external_surfaces[] };

// 6) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");