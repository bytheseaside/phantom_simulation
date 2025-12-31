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
Mesh.HighOrderOptimize = 1;        // Basic optimization

Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.0005;  // 0.5 mm — minimum mesh size
Mesh.CharacteristicLengthMax = 0.003;   // 3.0 mm — maximum mesh size
Mesh.MeshSizeFromCurvature  = 25;       // Curvature-driven refinement sensitivity

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;

// Refinement field near surfaces
Field[1] = Distance;
Field[1].SurfacesList = {
    internal_surfaces[],
    external_surfaces[]
};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.0005;    // 0.5 mm — highest resolution near electrode–gel contact
Field[2].SizeMax = 0.002;     // 2.0 mm — far from electrodes
Field[2].DistMin = 0.003;     // ≤ 3 mm → apply SizeMin
Field[2].DistMax = 0.008;     // 3–8 mm → transition to SizeMax



// 3) Physical groups used in solver
Physical Volume("sphere", 1) = { sphere_volume_ids[] };

Physical Surface("r_int", 2) = { internal_surfaces[] };
Physical Surface("r_ext", 3) = { external_surfaces[] };

// 6) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");