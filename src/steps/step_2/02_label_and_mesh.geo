// 02_label_and_mesh.geo
// Step-2: load frozen geometry (.xao), include IDs, define Physicals, mesh, save .msh (v4.1).

SetFactory("OpenCASCADE");
General.Terminal = 1;

// 0) Load frozen geometry from Step-1
Merge "prep.geo_unrolled.xao";

// 1) Bring in IDs read in GUI for entities of interest
Include "ids.inc.geo";
// --- Optional cleanup: delete stray volumes listed in ids.inc.geo ---
If (#volumes_to_delete[] > 0)
  Printf(">> Deleting %g unwanted volume(s): %g", #volumes_to_delete[], volumes_to_delete[]);
  Delete { Volume{ volumes_to_delete[] }; }
EndIf


// --- Sanity print ---
vs[] = Volume{:}; ss[] = Surface{:};
Printf(">> Geometry loaded: Volumes=%g, Surfaces=%g", #vs[], #ss[]);

// 2) Mesh parameters
Mesh.ElementOrder = 2;
Mesh.HighOrderOptimize = 2;        // high-order node placement
// Mesh.Algorithm3D = 4;
Mesh.CharacteristicLengthMin = 0.0015;   // 1.5 mm
Mesh.CharacteristicLengthMax = 0.004;    // 4 mm
Mesh.MeshSizeFromCurvature  = 20;

// Quality/robustness knobs
Mesh.SecondOrderLinear = 1;
Mesh.Optimize = 1; Mesh.OptimizeNetgen = 1; Mesh.Smoothing = 10;

// 3) (Optional) refinement near electrodes
Field[1] = Distance;
Field[1].SurfacesList = {
  e1_surfaces[], e2_surfaces[], e3_surfaces[],
  e4_surfaces[], e5_surfaces[], e6_surfaces[],
  e7_surfaces[], e8_surfaces[], e9_surfaces[]
};
Field[1].NumPointsPerCurve = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.0015;  // fine near contacts
Field[2].SizeMax = 0.004;   // revert far away
Field[2].DistMin = 0.006;   // <= 6 mm -> SizeMin
Field[2].DistMax = 0.012;   // transition to SizeMax

Background Field = 2;

// 4) Physical groups

// Volumes
Physical Volume("head", 1) = { head_volume_ids[] };
Physical Volume("gel", 2)  = { gel_volume_ids[]  };

// Electrodes
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
