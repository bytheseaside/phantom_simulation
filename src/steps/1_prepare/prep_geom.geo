// 01_prep_freeze_ids.geo
SetFactory("OpenCASCADE");
General.Terminal = 1;
// Geometry.AutoCoherence = 

// ---------- import ----------
Printf(">> Importing STEP...");
Merge StrCat(GetEnv("STEP_FILE"));

// ---------- scale mm -> m ----------
Printf(">> Scaling mm -> m...");
Dilate {{0,0,0}, 0.001} { Volume{:}; }

// ---------- report ----------
vs[] = Volume{:}; ss[] = Surface{:};
Printf(">> Post-prep counts: Volumes=%g, Surfaces=%g", #vs[], #ss[]);

If (#vs[] == 0)
  Printf("!! ERROR: No volumes detected. Aborting.");
  Exit;
EndIf

// ---------- save geometry ----------
Printf(">> Saving frozen geometry...");
Save StrCat(GetEnv("OUT_PATH"));