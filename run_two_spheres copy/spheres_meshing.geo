SetFactory("OpenCASCADE");
General.Terminal = 1;

Merge StrCat(GetEnv("XAO_PATH"));
vi[] = Volume{:}; si[] = Surface{:};
Printf(">> Config 1: FINE UNIFORM");
Printf(">> Loaded geometry: Volumes=%g, Surfaces=%g", #vi[], #si[]);

//* Volumes
int_vol_id[]  = { 1 };  
ext_vol_id[]  = { 2 };  
//* Surfaces
r0_surfaces[] = { 2 }; // r0 = 2 mm
r1_surfaces[] = { 1 }; // r1 = 80 mm
r2_surfaces[] = { 3 }; // r2 = 83.20 mm

Mesh.ElementOrder = 2;
Mesh.SecondOrderLinear = 2;
Mesh.Optimize = 1;
Mesh.Smoothing = 10;
Mesh.CharacteristicLengthMin = 0.0007;
Mesh.CharacteristicLengthMax = 0.001;

Physical Volume("int", 1) = { int_vol_id[] };
Physical Volume("ext", 2) = { ext_vol_id[] };
Physical Surface("r0", 3) = { r0_surfaces[] };
Physical Surface("r1", 5) = { r1_surfaces[] };
Physical Surface("r2", 4) = { r2_surfaces[] };

Mesh 3;
Mesh.MshFileVersion = 4.1;
Save StrCat(GetEnv("OUT_PATH"));
Printf(">> Mesh written");