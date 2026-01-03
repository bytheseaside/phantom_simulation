//Mesh Parameters for Phantom Head EEG Model
SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
shell_volume_ids[] = { 1 };
fill_volume_ids[] = { 2 };   
volume_ids_to_clean[] = {3 };

//* Surfaces - Jack TRS electrodes
// Electrode 1
e1_tip[] = { 5969, 5961, 5953, 5945, 5937 };
e1_ring[] = { 5867, 5823, 5784 };
e1_sleeve[] = { 5607 };
// Electrode 2
e2_tip[] = { 5968, 5960, 5952, 5944, 5936 };
e2_ring[] = { 5866, 5822, 5783 };
e2_sleeve[] = { 5606 };
// Electrode 3
e3_tip[] = { 5970, 5962, 5954, 5946, 5938 };
e3_ring[] = { 5868, 5824, 5785 };
e3_sleeve[] = { 5608 };
// Electrode 4
e4_tip[] = { 5967, 5959, 5951, 5943, 5935 };
e4_ring[] = { 5865, 5821, 5782 };
e4_sleeve[] = { 5605 };
// Electrode 5
e5_tip[] = { 5971, 5963, 5955, 5947, 5939 };
e5_ring[] = { 5869, 5825, 5786 };
e5_sleeve[] = { 5609 };
// Electrode 6
e6_tip[] = { 5966, 5958, 5950, 5942, 5934 };
e6_ring[] = { 5864, 5820, 5781 };
e6_sleeve[] = { 5604 };
// Electrode 7
e7_tip[] = { 5972, 5964, 5956, 5948, 5940 };
e7_ring[] = { 5870, 5826, 5787 };
e7_sleeve[] = { 5610 };
// Electrode 8
e8_tip[] = { 5965, 5957, 5949, 5941, 5933 };
e8_ring[] = { 5863, 5819, 5780 };
e8_sleeve[] = { 5603 };
// Electrode 9
e9_tip[] = { 5611, 5530, 5451, 5374, 5295 };
e9_ring[] = { 5289, 5287, 5285 };
e9_sleeve[] = { 5286 };


// Delete volumes - side effect of BooleanFragments
Delete { Volume{ volume_ids_to_clean[] }; }

// Recompute geometry sets
vp[] = Volume{:};
sp[] = Surface{:};
Printf(">> Geometry processed: Volumes=%g, Surfaces=%g", #vp[], #sp[]);

// 2) Global mesh parameters
Mesh.ElementOrder = 2;                  // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 1;             // Basic optimization
Mesh.SecondOrderLinear = 0;             // Use curved elements
Mesh.CharacteristicLengthMin = 0.0004;  // 0.4 mm
Mesh.CharacteristicLengthMax = 0.0012;   // 1.2 mm

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;


// 3) Physical groups used in solver
Physical Volume("shell", 1) = { shell_volume_ids[] };
Physical Volume("fill", 2)  = { fill_volume_ids[]  };

Physical Surface("e1_T", 11) = { e1_tip[] };
Physical Surface("e1_R", 12) = { e1_ring[] };
Physical Surface("e1_S", 13) = { e1_sleeve[] };

Physical Surface("e2_T", 21) = { e2_tip[] };
Physical Surface("e2_R", 22) = { e2_ring[] };
Physical Surface("e2_S", 23) = { e2_sleeve[] };

Physical Surface("e3_T", 31) = { e3_tip[] };
Physical Surface("e3_R", 32) = { e3_ring[] };
Physical Surface("e3_S", 33) = { e3_sleeve[] };

Physical Surface("e4_T", 41) = { e4_tip[] };
Physical Surface("e4_R", 42) = { e4_ring[] };
Physical Surface("e4_S", 43) = { e4_sleeve[] };

Physical Surface("e5_T", 51) = { e5_tip[] };
Physical Surface("e5_R", 52) = { e5_ring[] };
Physical Surface("e5_S", 53) = { e5_sleeve[] };

Physical Surface("e6_T", 61) = { e6_tip[] };
Physical Surface("e6_R", 62) = { e6_ring[] };
Physical Surface("e6_S", 63) = { e6_sleeve[] };

Physical Surface("e7_T", 71) = { e7_tip[] };
Physical Surface("e7_R", 72) = { e7_ring[] };
Physical Surface("e7_S", 73) = { e7_sleeve[] };

Physical Surface("e8_T", 81) = { e8_tip[] };
Physical Surface("e8_R", 82) = { e8_ring[] };
Physical Surface("e8_S", 83) = { e8_sleeve[] };

Physical Surface("e9_T", 91) = { e9_tip[] };
Physical Surface("e9_R", 92) = { e9_ring[] };
Physical Surface("e9_S", 93) = { e9_sleeve[] };

// 4) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");
