// phantom_head_meshing — Mesh Parameters for Phantom Head EEG Model
// Fine general params, without local refinement

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
conductive_shell_volume_ids[] = { 1 };
non_conductive_shell_volume_ids[] = { 5 }; 
fill_volume_ids[]  = { 2 };   

cleaning_volume_ids[] = { 3, 4, 6, 7, 8, 9, 10, 11, 12 };


//* Surfaces - Jack TRS electrodes
// Electrode 1
e1_tip[] = { 6005, 5997, 5989, 5981, 5973 };
e1_ring[] = { 5903, 5855, 5816 };
e1_sleeve[] = { 5639 };
// Electrode 2
e2_tip[] = { 6004, 5996, 5988, 5980, 5972 };
e2_ring[] = { 5902, 5854, 5815 };
e2_sleeve[] = { 5638 };
// Electrode 3
e3_tip[] = { 6006, 5998, 5990, 5982, 5974 };
e3_ring[] = { 5904, 5856, 5817 };
e3_sleeve[] = { 5640 };
// Electrode 4
e4_tip[] = {   };
e4_ring[] = {   };
e4_sleeve[] = {   };
// Electrode 5
e5_tip[] = {   };
e5_ring[] = {   };
e5_sleeve[] = {    };
// Electrode 6
e6_tip[] = {   };
e6_ring[] = {   };
e6_sleeve[] = {   };
// Electrode 7
e7_tip[] = {   };
e7_ring[] = {   };
e7_sleeve[] = {   };
// Electrode 8
e8_tip[] = {   };
e8_ring[] = {   };
e8_sleeve[] = {   };
// Electrode 9
e9_tip[] = {   };
e9_ring[] = {   };
e9_sleeve[] = {   };


// Delete cleaning volumes - side effect of BooleanFragments
Delete { Volume(cleaning_volume_ids[]); }

vp[] = Volume{:}; sp[] = Surface{:};
Printf(">> Geometry processed: Volumes=%g, Surfaces=%g", #vp[], #sp[]);

// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 1;        // Basic optimization
Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.0005;  // 0.5 mm — minimum mesh size
Mesh.CharacteristicLengthMax = 0.001;   // 1.0 mm — maximum mesh size
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
