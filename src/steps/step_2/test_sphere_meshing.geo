// test_sphere_meshing.geo — Test Sphere Meshing Script
// Purpose: Simple script to test sphere meshing. Loads frozen geometry,
// defines physical groups, sets mesh parameters, and generates a .msh file.
// We test with similar mesh parameters as in the phantom head model, except for the refinement
// field, which is omitted here for simplicity (not needed given the geometry).

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1, from the XAO_PATH environment variable
Merge "/Users/brisarojas/Desktop/phantom_simulation/run_20251220_192154/run_steps/prep.geo_unrolled.xao";
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
sphere_volume_ids[]  = { 1 };  
// Volumes to remove BEFORE meshing (leave {} if none)
volumes_to_delete[] = {  }; 

//* Surfaces
internal_surfaces[] = { 2 };
external_surfaces[] = { 1 };

If (#volumes_to_delete[] > 0)
  Printf(">> Deleting %g unwanted volume(s).", #volumes_to_delete[]);
  Delete { Volume{ volumes_to_delete[] }; }
EndIf

vp[] = Volume{:}; sp[] = Surface{:};
Printf(">> Geometry processed: Volumes=%g, Surfaces=%g", #vp[], #sp[]);

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
Save "/Users/brisarojas/Desktop/phantom_simulation/run_20251220_192154/run_steps/test_sphere_mesh.msh";
Printf(">> Mesh written");