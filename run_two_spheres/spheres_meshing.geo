SetFactory("OpenCASCADE");
General.Terminal = 1;

Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

//* Volumes
int_vol_id[]  = { 1 };  
ext_vol_id[]  = { 2 };  
//* Surfaces
r0_surfaces[] = { 2 }; // r0 = 2 mm
r1_surfaces[] = { 1 }; // r1 = 80 mm
r2_surfaces[] = { 3 }; // r2 = 83.20 mm

Mesh.ElementOrder = 2;
Mesh.SecondOrderLinear = 0;
Mesh.Optimize = 1;
Mesh.Smoothing = 10;
Mesh.CharacteristicLengthMin = 0.0003;  // Safety min ~0.3mm
Mesh.CharacteristicLengthMax = 0.006;   // Safety max ~6mm

// ============================================================
// Size Field: Refine near source (r0) surface
// ============================================================

// Distance field from inner surface (source location)
Field[1] = Distance;
Field[1].SurfacesList = { r0_surfaces[] };

// Threshold: fine near source, coarse far away
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = 0.0004;   // 0.4mm at source surface
Field[2].SizeMax = 0.005;    // 5mm far from source
Field[2].DistMin = 0.003;    // Fine mesh within 3mm of source
Field[2].DistMax = 0.025;    // Transition complete by 25mm

// Use as background field
Background Field = 2;

Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

// Physical groups
Physical Volume("int", 1) = { int_vol_id[] };
Physical Volume("ext", 2) = { ext_vol_id[] };
Physical Surface("r0", 3) = { r0_surfaces[] };
Physical Surface("r1", 5) = { r1_surfaces[] };
Physical Surface("r2", 4) = { r2_surfaces[] };

Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");