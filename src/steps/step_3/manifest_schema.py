from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from pathlib import Path
import numpy as np
from typing_extensions import Annotated

from pydantic import (
    BaseModel,
    ValidationError,
    model_validator,
    field_validator,
    Field,
    FiniteFloat,
    PrivateAttr,
)

# ---------------------------------------------------------------------
# Reusable constrained types
# ---------------------------------------------------------------------

PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonEmptyStr = Annotated[str, Field(min_length=1)]

# ---------------------------------------------------------------------
# Base entity types
# ---------------------------------------------------------------------

class StrictBase(BaseModel):
    model_config = {"extra": "forbid", "strict": True}

class EntityByName(StrictBase):
    name: NonEmptyStr

class EntityById(StrictBase):
    id: Annotated[int, Field(ge=1)]

# ---------------------------------------------------------------------
# Material properties and Dirichlet value
# ---------------------------------------------------------------------

class MaterialProperties(StrictBase):
    sigma: PositiveFiniteFloat

class DirichletValue(StrictBase):
    value: FiniteFloat

# ---------------------------------------------------------------------
# Volume union (materials)
# ---------------------------------------------------------------------

class VolumeByName(EntityByName, MaterialProperties):
    pass

class VolumeById(EntityById, MaterialProperties):
    pass

Volume = Union[VolumeByName, VolumeById]

# ---------------------------------------------------------------------
# Dirichlet union (targets are FACETS/SURFACES in the mesh)
# ---------------------------------------------------------------------

class DirichletByName(EntityByName, DirichletValue):
    pass

class DirichletById(EntityById, DirichletValue):
    pass

Dirichlet = Union[DirichletByName, DirichletById]

# ---------------------------------------------------------------------
# Case
# ---------------------------------------------------------------------

class Case(StrictBase):
    name: NonEmptyStr
    dirichlet: Annotated[List[Dirichlet], Field(min_length=1)]

    @field_validator("dirichlet")
    @classmethod
    def unique_dirichlet_targets(cls, v: List[Dirichlet]):
        """
        Enforce no duplicate BC targets inside a single case.
        Note: ('id', 11) and ('name', 'E1') are considered different keys
        because we don't know their equivalence until we consult mesh metadata.
        """
        seen: set[tuple[str, object]] = set()
        for bc in v:
            key = ("id", bc.id) if hasattr(bc, "id") else ("name", bc.name)
            if key in seen:
                raise ValueError(f"Duplicate Dirichlet target with {key[0]}: {key[1]}")
            seen.add(key)
        return v

# ---------------------------------------------------------------------
# Mesh metadata
# ---------------------------------------------------------------------

class MeshInfo(StrictBase):
    """
    Mesh physical group metadata, as extracted from meshio / Gmsh.

    field_data format:
        { name: (physical_id, dim) }
    where:
        dim == 2 → surface (facet)
        dim == 3 → volume (cell)
    """
    field_data: Dict[str, Tuple[int, int]]

    @model_validator(mode="after")
    def check_field_data(self):
        for name, (pid, dim) in self.field_data.items():
            if not isinstance(pid, int) or pid < 1:
                raise ValueError(f"field_data['{name}'] has invalid id: {pid!r}")
            if dim not in (2, 3):
                raise ValueError(
                    f"field_data['{name}'] has unsupported dim={dim} "
                    f"(expected 2 for surfaces or 3 for volumes)"
                )
        return self

    @property
    def surface_names(self) -> set[str]:
        return {n for n, (_, d) in self.field_data.items() if d == 2}

    @property
    def surface_ids(self) -> set[int]:
        return {pid for (_, (pid, d)) in self.field_data.items() if d == 2}

    @property
    def volume_names(self) -> set[str]:
        return {n for n, (_, d) in self.field_data.items() if d == 3}

    @property
    def volume_ids(self) -> set[int]:
        return {pid for (_, (pid, d)) in self.field_data.items() if d == 3}

    # Unified physical entity maps (all dims together), as you prefer:
    @property
    def physical_name_to_id(self) -> Dict[str, int]:
        return {name: pid for name, (pid, _dim) in self.field_data.items()}

    @property
    def physical_id_to_name(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for name, (pid, _dim) in self.field_data.items():
            out[pid] = name
        return out

    def resolve_surface_id(self, *, name: str | None = None, id: int | None = None) -> int:
        """
        Resolve a surface reference (name or id) to its physical id.
        Raises ValueError if not found or if entity is not dim=2.
        """
        if name is not None:
            if name not in self.field_data:
                raise ValueError(f"Surface name '{name}' not found in mesh.")
            pid, dim = self.field_data[name]
            if dim != 2:
                raise ValueError(f"'{name}' exists but is not a surface (dim={dim}).")
            return pid

        if id is not None:
            # Find any name with that id; then check dim
            for n, (pid, dim) in self.field_data.items():
                if pid == id:
                    if dim != 2:
                        raise ValueError(f"Physical id {id} exists but is not a surface (dim={dim}).")
                    return pid
            raise ValueError(f"Surface id {id} not found in mesh.")

        raise ValueError("Either name or id must be provided.")

    def resolve_volume_id(self, *, name: str | None = None, id: int | None = None) -> int:
        """
        Resolve a volume reference (name or id) to its physical id.
        Raises ValueError if not found or if entity is not dim=3.
        """
        if name is not None:
            if name not in self.field_data:
                raise ValueError(f"Volume name '{name}' not found in mesh.")
            pid, dim = self.field_data[name]
            if dim != 3:
                raise ValueError(f"'{name}' exists but is not a volume (dim={dim}).")
            return pid

        if id is not None:
            for n, (pid, dim) in self.field_data.items():
                if pid == id:
                    if dim != 3:
                        raise ValueError(f"Physical id {id} exists but is not a volume (dim={dim}).")
                    return pid
            raise ValueError(f"Volume id {id} not found in mesh.")

        raise ValueError("Either name or id must be provided.")

# ---------------------------------------------------------------------
# Validation report (errors + warnings)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class MeshValidationReport:
    errors: List[str]
    warnings: List[str]

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError("Manifest does not match mesh physical groups:\n- " + "\n- ".join(self.errors))

# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

class Manifest(StrictBase):
    volumes: Annotated[List[Volume], Field(min_length=1)]
    cases: Annotated[List[Case], Field(min_length=1)]
    mesh: Path
    output: Path

    # ---- runtime-only caches (not validated / not serialized) ----
    _name_to_id: Dict[str, int] = PrivateAttr(default_factory=dict)
    _id_to_name: Dict[int, str] = PrivateAttr(default_factory=dict)

    @property
    def name_to_id(self) -> Dict[str, int]:
        return self._name_to_id

    @property
    def id_to_name(self) -> Dict[int, str]:
        return self._id_to_name


    @field_validator("mesh", "output", mode="before")
    @classmethod
    def coerce_paths(cls, v):
        return Path(v) if isinstance(v, str) else v

    @field_validator("volumes")
    @classmethod
    def unique_volumes(cls, v: List[Volume]):
        names: set[str] = set()
        ids: set[int] = set()
        for vol in v:
            if hasattr(vol, "name"):
                if vol.name in names:
                    raise ValueError(f"Duplicate volume name: {vol.name}")
                names.add(vol.name)
            if hasattr(vol, "id"):
                if vol.id in ids:
                    raise ValueError(f"Duplicate volume id: {vol.id}")
                ids.add(vol.id)
        return v

    @field_validator("cases")
    @classmethod
    def unique_case_names(cls, v: List[Case]):
        seen: set[str] = set()
        for case in v:
            if case.name in seen:
                raise ValueError(f"Duplicate case name: {case.name}")
            seen.add(case.name)
        return v

    # -----------------------------------------------------------------
    # Runtime (mesh-aware) validation
    # -----------------------------------------------------------------
    def validate_against_mesh(
        self,
        field_data: Dict[str, np.ndarray],
    ) -> MeshValidationReport:
        """
        Rules implemented:
        - ERROR if volumes do not match EXACTLY between mesh and manifest:
            * every mesh physical volume must be in manifest.volumes
            * every manifest volume must exist in mesh physical volumes
        - ERROR if a referenced surface (Dirichlet) doesn't exist or isn't dim=2.
        - WARNING if there are mesh physical surfaces that are never used by any Dirichlet BC.
        """
        mesh_info = mesh_info_from_field_data(field_data)
        errors: list[str] = []
        warnings: list[str] = []

        # ---------------------------
        # 1) Dirichlet surfaces: referenced must exist (dim=2)
        # ---------------------------
        referenced_surface_ids: set[int] = set()

        for case in self.cases:
            resolved_ids: list[int] = []

            for bc in case.dirichlet:
                try:
                    if hasattr(bc, "name"):
                        pid = mesh_info.resolve_surface_id(name=bc.name)
                    else:
                        pid = mesh_info.resolve_surface_id(id=bc.id)
                except ValueError as e:
                    errors.append(f"Case '{case.name}': {e}")
                    continue

                resolved_ids.append(pid)
                referenced_surface_ids.add(pid)

            # duplicates after resolution (e.g. {"id":12} and {"name":"E2"} where E2->12)
            if len(set(resolved_ids)) < len(resolved_ids):
                errors.append(
                    f"Case '{case.name}': duplicate Dirichlet targets after resolving names/ids."
                )

        unused_surface_ids = sorted(mesh_info.surface_ids - referenced_surface_ids)
        if unused_surface_ids:
            # Show ids + names (best effort)
            id_to_name = mesh_info.physical_id_to_name
            pretty = [f"{sid}('{id_to_name.get(sid, '?')}')" for sid in unused_surface_ids]
            warnings.append(
                f"{len(unused_surface_ids)} physical surface(s) exist in the mesh but are not referenced by any "
                f"Dirichlet BC: {', '.join(pretty)}"
            )

        # ---------------------------
        # 2) Volumes: EXACT match between mesh and manifest (by resolved physical IDs)
        # ---------------------------

        mesh_volume_ids = mesh_info.volume_ids  # all physical volume ids in mesh (dim=3)

        # Resolve manifest volumes to physical ids (works for both name-based and id-based entries)
        resolved_manifest_volume_ids: set[int] = set()
        for v in self.volumes:
            try:
                if hasattr(v, "name"):
                    pid = mesh_info.resolve_volume_id(name=v.name)
                else:
                    pid = mesh_info.resolve_volume_id(id=v.id)
            except ValueError as e:
                # This covers "not found" and "exists but is not a volume (dim!=3)"
                errors.append(str(e))
                continue

            resolved_manifest_volume_ids.add(pid)

        missing_in_manifest = sorted(mesh_volume_ids - resolved_manifest_volume_ids)
        if missing_in_manifest:
            id_to_name = mesh_info.physical_id_to_name
            pretty = [f"{vid}('{id_to_name.get(vid, '?')}')" for vid in missing_in_manifest]
            errors.append(
                f"Mesh has {len(missing_in_manifest)} physical volume id(s) not declared in manifest.volumes: "
                f"{', '.join(pretty)}"
            )


        return MeshValidationReport(errors=errors, warnings=warnings)

    def resolve_with_mesh(self, field_data: Dict[str, np.ndarray]) -> None:
        """
        Resolve ALL manifest entities against mesh field_data so that:
        - every Volume has BOTH .id and .name
        - every Dirichlet has BOTH .id and .name
        """
        mesh_info = mesh_info_from_field_data(field_data)
        # Global maps (all physical entities)
        self._name_to_id = mesh_info.physical_name_to_id
        self._id_to_name = mesh_info.physical_id_to_name

        # -----------------------
        # Volumes
        # -----------------------
        new_volumes: list[Volume] = []

        for v in self.volumes:
            if hasattr(v, "id") and hasattr(v, "name"):
                new_volumes.append(v)
                continue

            if hasattr(v, "name"):
                vid = mesh_info.resolve_volume_id(name=v.name)
                new_volumes.append(
                    VolumeById(id=vid, sigma=v.sigma).model_copy(
                        update={"name": v.name}
                    )
                )
            else:  # id only
                name = mesh_info.physical_id_to_name[v.id]
                new_volumes.append(
                    VolumeByName(name=name, sigma=v.sigma).model_copy(
                        update={"id": v.id}
                    )
                )

        self.volumes = new_volumes

        # -----------------------
        # Dirichlet BCs
        # -----------------------
        for case in self.cases:
            resolved: list[Dirichlet] = []

            for bc in case.dirichlet:
                if hasattr(bc, "id") and hasattr(bc, "name"):
                    resolved.append(bc)
                    continue

                if hasattr(bc, "name"):
                    sid = mesh_info.resolve_surface_id(name=bc.name)
                    resolved.append(
                        DirichletById(id=sid, value=bc.value).model_copy(
                            update={"name": bc.name}
                        )
                    )
                else:  # id only
                    name = mesh_info.physical_id_to_name[bc.id]
                    resolved.append(
                        DirichletByName(name=name, value=bc.value).model_copy(
                            update={"id": bc.id}
                        )
                    )

            case.dirichlet = resolved

# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------

def load_manifest(path: str | Path) -> Manifest:
    import json

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        return Manifest(**data)
    except ValidationError as e:
        print("Manifest validation failed:")
        print(e)
        raise

# ---------------------------------------------------------------------
# Helper: build MeshInfo from meshio field_data
# ---------------------------------------------------------------------

def mesh_info_from_field_data(field_data: Dict[str, np.ndarray]) -> MeshInfo:
    normalized: Dict[str, tuple[int, int]] = {}
    for name, v in field_data.items():
        normalized[name] = (int(v[0]), int(v[1]))
    return MeshInfo(field_data=normalized)
