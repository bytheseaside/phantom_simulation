// 01_prep_freeze_ids.geo
// Goal: import STEP, scale mm->m, (optional) light healing,
// perform conformal split (BooleanFragments) and serialize a
// frozen geometry with stable IDs (.geo_unrolled + .xao).
// No meshing, no physical groups here.

SetFactory("OpenCASCADE");
General.Terminal = 1;

// Disable auto-coherence write-backs we don't control explicitly
Geometry.AutoCoherence = 0;

// ---------- toggles ----------
doHeal      = 0;  // 1 = OCC small edges/faces fix + sewing (before booleans)
doCoherence = 1;  // 1 = remove duplicates after import (can renumber)
doFragment  = 1;  // 1 = conformal split (BooleanFragments)

// ---------- import ----------
Printf(">> Importing STEP...");
Merge StrCat(GetEnv("STEP_FILE"));

// ---------- scale mm -> m ----------
Printf(">> Scaling mm -> m...");
Dilate {{0,0,0}, 0.001} { Volume{:}; }

// ---------- optional healing ----------
If (doHeal)
  Printf(">> OCC healing ON (small edges/faces + sew + make solids)...");
  Geometry.OCCFixDegenerated     = 1;
  Geometry.OCCFixSmallEdges      = 1;
  Geometry.OCCFixSmallFaces      = 1;
  Geometry.OCCSewFaces           = 1;
  Geometry.OCCMakeSolids         = 1;
EndIf

// ---------- optional coherence ----------
If (doCoherence)
  Printf(">> Coherence (deduplicate)...");
  Coherence;
EndIf

// ---------- conformal split ----------
If (doFragment)
  Printf(">> BooleanFragments (conformal split, delete inputs)...");
  // Fragment everything with everything, remove originals
  vol_new() = BooleanFragments{ Volume{:}; Delete; }{};
  Coherence;
EndIf

// ---------- report ----------
vs[] = Volume{:}; ss[] = Surface{:};
Printf(">> Post-prep counts: Volumes=%g, Surfaces=%g", #vs[], #ss[]);

If (#vs[] == 0)
  Printf("!! ERROR: No volumes detected. Aborting.");
  Exit;
EndIf

// ---------- save frozen geometry (XAO + stub GEO) ----------
Printf(">> Saving frozen geometry: prep.geo_unrolled (+ .xao) ...");
Save "prep.geo_unrolled";