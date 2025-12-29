// Mesh Parameters for Phantom Head EEG Model
// Purpose: Test version - quick rough parameters for fast mesh generation and debugging of pipeline

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
conductive_shell_volume_ids[] = { 2 };
non_conductive_shell_volume_ids[] = { 3 }; 
fill_volume_ids[]  = { 1 };  


//* Surfaces - Jack TRS electrodes
// Electrode 1
e1_tip[] = { 2272, 2273, 2274, 2275, 2276 };
e1_ring[] = { 2266, 2267, 2268 };
e1_sleeve[] = { 50 };
// Electrode 2
e2_tip[] = { 43, 44, 45, 46, 47 };
e2_ring[] = { 2281, 2282, 2283 };
e2_sleeve[] = { 48  };
// Electrode 3
e3_tip[] = {  2290, 2291, 2292, 2293, 2294 };
e3_ring[] = { 2298, 2299, 2300 };
e3_sleeve[] = { 2302 };
// Electrode 4
e4_tip[] = { 2305, 2306, 2307, 2308, 2309 };
e4_ring[] = {  35, 36, 37 };
e4_sleeve[] = { 2304 };
// Electrode 5
e5_tip[] = { 2318, 2319, 2320, 2321, 2322 };
e5_ring[] = { 2314, 2315, 2316 };
e5_sleeve[] = {  2312 };
// Electrode 6
e6_tip[] = { 2331, 2232, 2233, 2334, 2335 };
e6_ring[] = { 2327, 2328, 2329 };
e6_sleeve[] = { 2325 };
// Electrode 7
e7_tip[] = { 2340, 2341, 2342, 2343, 2344 };
e7_ring[] = { 2346, 2347, 2348 };
e7_sleeve[] = { 2338 };
// Electrode 8
e8_tip[] = { 7, 8, 9, 10, 11 };
e8_ring[] = { 14, 15, 16 };
e8_sleeve[] = { 2351 };
// Electrode 9
e9_tip[] = { 2355, 2356, 2357, 2358, 2359 };
e9_ring[] = { 2361, 2362, 2363 };
e9_sleeve[] = { 2353 };

vp[] = Volume{:}; sp[] = Surface{:};
Printf(">> Geometry processed: Volumes=%g, Surfaces=%g", #vp[], #sp[]);

// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 1;        // Basic optimization
Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.001;  // 0.5 mm — minimum mesh size
Mesh.CharacteristicLengthMax = 0.002;   // 2.0 mm — maximum mesh size
Mesh.MeshSizeFromCurvature  = 25;       // Curvature-driven refinement sensitivity

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;


// 3) Physical groups used in solver
Physical Volume("conductive_shell") = { conductive_shell_volume_ids[] };
Physical Volume("non_conductive_shell") = { non_conductive_shell_volume_ids[] };
Physical Volume("fill")  = { fill_volume_ids[]  };

Physical Surface("e1_T") = { e1_tip[] };
Physical Surface("e1_R") = { e1_ring[] };
Physical Surface("e1_S") = { e1_sleeve[] };

Physical Surface("e2_T") = { e2_tip[] };
Physical Surface("e2_R") = { e2_ring[] };
Physical Surface("e2_S") = { e2_sleeve[] };

Physical Surface("e3_T") = { e3_tip[] };
Physical Surface("e3_R") = { e3_ring[] };
Physical Surface("e3_S") = { e3_sleeve[] };

Physical Surface("e4_T") = { e4_tip[] };
Physical Surface("e4_R") = { e4_ring[] };
Physical Surface("e4_S") = { e4_sleeve[] };


Physical Surface("e5_T") = { e5_tip[] };
Physical Surface("e5_R") = { e5_ring[] };
Physical Surface("e5_S") = { e5_sleeve[] };

Physical Surface("e6_T") = { e6_tip[] };
Physical Surface("e6_R") = { e6_ring[] };
Physical Surface("e6_S") = { e6_sleeve[] };

Physical Surface("e7_T") = { e7_tip[] };
Physical Surface("e7_R") = { e7_ring[] };
Physical Surface("e7_S") = { e7_sleeve[] };

Physical Surface("e8_T") = { e8_tip[] };
Physical Surface("e8_R") = { e8_ring[] };
Physical Surface("e8_S") = { e8_sleeve[] };

Physical Surface("e9_T") = { e9_tip[] };
Physical Surface("e9_R") = { e9_ring[] };
Physical Surface("e9_S") = { e9_sleeve[] };


// 6) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");
