// 02_label_and_mesh.geo
// Purpose: Load frozen geometry, define physical groups, set mesh parameters
// and generate a .msh file with locally refined mesh for phantom head EEG.

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge "prep.geo_unrolled.xao";

// 1) Bring in IDs for head, gel, and electrode contact surfaces
Include "ids.inc.geo";
If (#volumes_to_delete[] > 0)
  Printf(">> Deleting %g unwanted volume(s): %g", #volumes_to_delete[], volumes_to_delete[]);
  Delete { Volume{ volumes_to_delete[] }; }
EndIf

vs[] = Volume{:}; ss[] = Surface{:};
Printf(">> Geometry loaded: Volumes=%g, Surfaces=%g", #vs[], #ss[]);

// 2) Global mesh parameters
Mesh.ElementOrder = 2;             // Use 2nd-order tetrahedra (quadratic elements)
Mesh.HighOrderOptimize = 2;        // Advanced optimization of high-order nodes
Mesh.SecondOrderLinear = 1;        // Straight internal edges (safer for PDE)

Mesh.CharacteristicLengthMin = 0.00075; // 0.75 mm — minimum mesh size near fine features
Mesh.CharacteristicLengthMax = 0.0045;  // 4.5 mm — maximum mesh size in interior gel
Mesh.MeshSizeFromCurvature  = 25;       // Curvature-driven refinement sensitivity (moderate-high)

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
Field[2].SizeMin = 0.00075;  // 0.75 mm — highest resolution near electrode–gel contact
Field[2].SizeMax = 0.0045;   // 4.5 mm — revert to default size in deep gel
Field[2].DistMin = 0.002;    // ≤ 2 mm → apply SizeMin
Field[2].DistMax = 0.010;    // 2–10 mm → transition to SizeMax

// 4) Refinement near inner shell–gel interface
// Extract boundary surfaces of head volume automatically
shell_surfaces[] = Surface In BoundingSurfaces{ Volume{ head_volume_ids[] }; };

Field[3] = Distance;
Field[3].SurfacesList = { shell_surfaces[] };
Field[3].NumPointsPerCurve = 200;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = 0.00075;    // < 1 mm — to ensure ≥4 elements through 3.2 mm shell
Field[4].SizeMax = 0.0045;   // revert to default
Field[4].DistMin = 0.002;   // ≤ 2 mm → SizeMin
Field[4].DistMax = 0.004;    // transition zone

// Combine both refinements
Field[5] = Min;
Field[5].FieldsList = {2, 4};

Background Field = 5; // Use the minimum size from electrode and shell fields

// 4) Physical groups used in solver
Physical Volume("head", 1) = { head_volume_ids[] };
Physical Volume("gel", 2)  = { gel_volume_ids[]  };

Physical Surface("v_1", 11) = { e1_surfaces[] };
Physical Surface("v_2", 12) = { e2_surfaces[] };
Physical Surface("v_3", 13) = { e3_surfaces[] };
Physical Surface("v_4", 14) = { e4_surfaces[] };
Physical Surface("v_5", 15) = { e5_surfaces[] };
Physical Surface("v_6", 16) = { e6_surfaces[] };
Physical Surface("v_7", 17) = { e7_surfaces[] };
Physical Surface("v_8", 18) = { e8_surfaces[] };
Physical Surface("v_9", 19) = { e9_surfaces[] };

// Head outer surface
// Physical Surface("outer_head") = { head_outer_surface[] };

// 5) Mesh + save
Mesh 3;
Mesh.MshFileVersion = 4.1;
Save "mesh.msh";
Printf(">> Mesh written: mesh.msh");
