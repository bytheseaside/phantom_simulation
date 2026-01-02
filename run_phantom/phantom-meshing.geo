//Mesh Parameters for Phantom Head EEG Model
SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
shell_volume_ids[] = { 2 };
fill_volume_ids[] = { 1 };   
volume_ids_to_clean[] = { 3, 4, 5 };

//* Surfaces - Jack TRS electrodes
// Electrode 1
e1_tip[] = { 2001, 1893, 1739, 1617, 1493 };
e1_ring[] = { 1067, 891, 777 };
e1_sleeve[] = { 326 };
// Electrode 2
e2_tip[] = { 2000, 1892, 1738, 1616, 1492 };
e2_ring[] = { 1066, 890, 776 };
e2_sleeve[] = { 325 };
// Electrode 3
e3_tip[] = { 2002, 1894, 1740, 1618, 1494 };
e3_ring[] = { 1068, 892, 778 };
e3_sleeve[] = { 327 };
// Electrode 4
e4_tip[] = { 1999, 1891, 1737, 1615, 1491 };
e4_ring[] = { 1065, 889, 775 };
e4_sleeve[] = { 324 };
// Electrode 5
e5_tip[] = { 2003, 1895, 1741, 1619, 1495 };
e5_ring[] = { 1069, 893, 779 };
e5_sleeve[] = { 328};
// Electrode 6
e6_tip[] = { 1998, 1890, 1736, 1614, 1490 };
e6_ring[] = { 1064, 888, 774 };
e6_sleeve[] = { 323 };
// Electrode 7
e7_tip[] = { 2004, 1896, 1742, 1620, 1496 };
e7_ring[] = { 1070, 894, 780 };
e7_sleeve[] = { 329  };
// Electrode 8
e8_tip[] = { 1997, 1889, 1735, 1613, 1489 };
e8_ring[] = { 1063, 887, 773 };
e8_sleeve[] = { 322 };
// Electrode 9
e9_tip[] = { 330, 249, 170, 93, 14 };
e9_ring[] = { 8, 6, 4 };
e9_sleeve[] = { 5 };


// Delete volumes - side effect of BooleanFragments
Delete { Volume(volume_ids_to_clean[]); }

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
