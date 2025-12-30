// Mesh Parameters for Phantom Head EEG Model
// Coarse general params, with local refinement near relevant surfaces

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
shell_volume_ids[] = {   };
fill_volume_ids[]  = {  };  
volume_ids_to_clean[] = {   };

//* Surfaces - Jack TRS electrodes
// Electrode 1
e1_tip[] = {  };
e1_ring[] = {  };
e1_sleeve[] = {  };
// Electrode 2
e2_tip[] = {  };
e2_ring[] = {  };
e2_sleeve[] = {  };
// Electrode 3
e3_tip[] = {  };
e3_ring[] = {  };
e3_sleeve[] = {  };
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

// Delete volumes - side effect of BooleanFragments
Delete { Volume(volume_ids_to_clean[]); }

// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 1;        // Basic optimization
Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.0005;  // 0.5 mm — minimum mesh size
Mesh.CharacteristicLengthMax = 0.005;   // 5.0 mm — maximum mesh size
Mesh.MeshSizeFromCurvature  = 25;       // Curvature-driven refinement sensitivity

Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;
Mesh.Smoothing = 10;

// 3) Refinement field near electrode contact surfaces
Field[1] = Distance;
Field[1].SurfacesList = {
  e1_tip[], e1_ring[], e1_sleeve[],
  e2_tip[], e2_ring[], e2_sleeve[],
  e3_tip[], e3_ring[], e3_sleeve[],
  e4_tip[], e4_ring[], e4_sleeve[],
  e5_tip[], e5_ring[], e5_sleeve[],
  e6_tip[], e6_ring[], e6_sleeve[],
  e7_tip[], e7_ring[], e7_sleeve[],
  e8_tip[], e8_ring[], e8_sleeve[],
  e9_tip[], e9_ring[], e9_sleeve[]
};
Field[1].NumPointsPerCurve = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.0005;    // 0.5 mm — highest resolution near electrode–gel contact
Field[2].SizeMax = 0.002;     // 2.0 mm — far from electrodes
Field[2].DistMin = 0.003;     // ≤ 3 mm → apply SizeMin
Field[2].DistMax = 0.008;     // 3–8 mm → transition to SizeMax

// 4) Refinement near shell–gel interface in the conductive shell
shell_volume_boundaries[] = Boundary{ Volume{ conductive_shell_volume_ids[] }; };

Field[3] = Distance;
Field[3].SurfacesList = { shell_volume_boundaries[] };
Field[3].NumPointsPerCurve = 200;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = 0.0005;    // 0.5 mm — near shell boundaries
Field[4].SizeMax = 0.002;     // 2.0 mm — far from shell
Field[4].DistMin = 0.003;     // ≤ 3 mm → SizeMin
Field[4].DistMax = 0.008;     // 3–8 mm → transition zone

// Combine both refinements
Field[5] = Min;
Field[5].FieldsList = {2, 4};

Background Field = 5; // Use the minimum size from electrode and shell fields

// 5) Physical groups used in solver
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

// 6) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");
