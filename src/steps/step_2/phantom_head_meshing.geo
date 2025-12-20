// phantom_head_meshing — Mesh Parameters for Phantom Head EEG Model
// Purpose: Load frozen geometry, define physical groups, set mesh parameters
// and generate a .msh file with locally refined mesh for phantom head EEG.

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge // TODO: path here...;
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

// 1) IDs for head, fill, and electrode contact surfaces
//* Volumes
fill_volume_ids[]  = { 2 };  
shell_volume_ids[] = { 1 }; 
// Volumes to remove BEFORE meshing (leave {} if none)
volumes_to_delete[] = { 3 }; 

//* Surfaces
e1_surfaces[] = { 5704, 5712 };
e2_surfaces[] = { 5703, 5711 };
e3_surfaces[] = { 5702, 5710 };
e4_surfaces[] = { 5701, 5709 };
e5_surfaces[] = { 5700, 5708 };
e6_surfaces[] = { 5707, 5699 };
e7_surfaces[] = { 5698, 5706 };
e8_surfaces[] = { 5697, 5705 };
e9_surfaces[] = { 5679, 5688 };

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
  e1_surfaces[], e2_surfaces[], e3_surfaces[],
  e4_surfaces[], e5_surfaces[], e6_surfaces[],
  e7_surfaces[], e8_surfaces[], e9_surfaces[]
};
Field[1].NumPointsPerCurve = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.0005;   // 0.5 mm — highest resolution near electrode–gel contact
Field[2].SizeMax = 0.002;    // 2.0 mm — far from electrodes
Field[2].DistMin = 0.002;    // ≤ 2 mm → apply SizeMin
Field[2].DistMax = 0.008;    // 2–8 mm → transition to SizeMax

// 4) Refinement near shell–gel interface
// Extract boundary surfaces of head volume automatically
shell_surfaces[] = Surface In BoundingSurfaces{ Volume{ shell_volume_ids[] }; };

Field[3] = Distance;
Field[3].SurfacesList = { shell_surfaces[] };
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
Physical Volume("shell", 1) = { shell_volume_ids[] };
Physical Volume("fill", 2)  = { fill_volume_ids[]  };

Physical Surface("e1", 11) = { e1_surfaces[] };
Physical Surface("e2", 12) = { e2_surfaces[] };
Physical Surface("e3", 13) = { e3_surfaces[] };
Physical Surface("e4", 14) = { e4_surfaces[] };
Physical Surface("e5", 15) = { e5_surfaces[] };
Physical Surface("e6", 16) = { e6_surfaces[] };
Physical Surface("e7", 17) = { e7_surfaces[] };
Physical Surface("e8", 18) = { e8_surfaces[] };
Physical Surface("e9", 19) = { e9_surfaces[] };

// 6) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save // TODO: output path here...;
Printf(">> Mesh written");
