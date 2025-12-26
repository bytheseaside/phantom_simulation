// phantom_head_meshing — Mesh Parameters for Phantom Head EEG Model
// Purpose: Load frozen geometry, define physical groups, set mesh parameters
// and generate a .msh file with locally refined mesh for phantom head EEG.

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
shell_volume_ids[] = { 1 }; 
fill_volume_ids[]  = { 2 };  
// Volumes to remove BEFORE meshing
volumes_to_delete[] = { 3, 4, 5 }; 

//* Surfaces - Jack TRS electrodes
// Electrode 1
e1_tip[] = { 6830, 6822, 6814, 6806, 6798 };
e1_ring[] = { 6774, 6766, 6758 };
e1_sleeve[] = { 6725 };
// Electrode 2
e2_tip[] = { 6829, 6821, 6813, 6805, 6797 };
e2_ring[] = { 6773, 6765, 6757 };
e2_sleeve[] = { 6724 };
// Electrode 3
e3_tip[] = { 6831, 6823, 6815, 6807, 6799 };
e3_ring[] = { 6775, 6767, 6759 };
e3_sleeve[] = { 6726 };
// Electrode 4
e4_tip[] = { 6828, 6820, 6812, 6804, 6796 };
e4_ring[] = { 6772, 6764, 6756 };
e4_sleeve[] = { 6723 };
// Electrode 5
e5_tip[] = { 6832, 6824, 6816, 6808, 6800 };
e5_ring[] = { 6776, 6768, 6760 };
e5_sleeve[] = { 6727};
// Electrode 6
e6_tip[] = { 6827, 6819, 6811, 6803, 6795 };
e6_ring[] = { 6771, 6763, 6755 };
e6_sleeve[] = { 6722 };
// Electrode 7
e7_tip[] = { 6833, 6825, 6817, 6809, 6801 };
e7_ring[] = { 6777, 6769, 6761 };
e7_sleeve[] = { 6728 };
// Electrode 8
e8_tip[] = { 6826, 6818, 6810, 6802, 6794 };
e8_ring[] = { 6770, 6762, 6754};
e8_sleeve[] = { 6721 };
// Electrode 9
e9_tip[] = { 6729, 6720, 6711, 6702, 6693 };
e9_ring[] = { 6687, 6685, 6683 };
e9_sleeve[] = { 6684 };

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
Field[2].SizeMin = 0.0005;   // 0.5 mm — highest resolution near electrode–gel contact
Field[2].SizeMax = 0.002;    // 2.0 mm — far from electrodes
Field[2].DistMin = 0.002;   // ≤ 2 mm → apply SizeMin
Field[2].DistMax = 0.008;    // 2–8 mm → transition to SizeMax

// 4) Refinement near shell–gel interface
shell_volume_boundaries[] = Boundary{ Volume{ shell_volume_ids[] }; };

Field[3] = Distance;
Field[3].SurfacesList = shell_volume_boundaries[]
Field[3].NumPointsPerCurve = 200;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = 0.0005;   // 0.5 mm — near shell boundaries (≈6 P2 elements across 3.2mm)
Field[4].SizeMax = 0.002;    // 2.0 mm — far from shell
Field[4].DistMin = 0.003;    // ≤ 3 mm → SizeMin
Field[4].DistMax = 0.008;    // 3–8 mm → transition zone

// Combine both refinements
Field[5] = Min;
Field[5].FieldsList = {2, 4};

Background Field = 5; // Use the minimum size from electrode and shell fields

// 5) Physical groups used in solver
Physical Volume("shell") = { shell_volume_ids[] };
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
