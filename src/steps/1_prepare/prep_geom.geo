SetFactory("OpenCASCADE");
General.Terminal = 1;
// Geometry.AutoCoherence = 0

// ---------- import ----------
Printf(">> Importing STEP...");
Merge StrCat(GetEnv("STEP_FILE"));

// ---------- scale mm -> m ----------
Printf(">> Scaling mm -> m...");
Dilate {{0,0,0}, 0.001} { Volume{:}; }


// ---------- topology changes ----------
Printf(">> Conformality and coherence...");
BooleanFragments{ Volume{:}; Delete; }{}
Coherence;


// ---------- save geometry ----------
Printf(">> Saving frozen geometry...");
Save StrCat(GetEnv("OUT_PATH"));