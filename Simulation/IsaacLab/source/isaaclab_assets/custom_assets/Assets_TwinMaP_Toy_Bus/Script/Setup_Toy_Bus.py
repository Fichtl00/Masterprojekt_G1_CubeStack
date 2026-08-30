import omni
import omni.usd
import omni.isaac.core.utils.stage as stage_utils

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

# ----------------------------------------
# Pfade für Files
# ----------------------------------------
BODY_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Body_2.usd"
SCHRAUBE1_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_3.6.usd"
SCHRAUBE2_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Scrhaube_2.usd"
SCHRAUBE3_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_3.usd"
SCHRAUBE4_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_4.usd"
SCHRAUBE5_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_5.usd"
SCHRAUBE6_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_6.usd"
SCHRAUBE7_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_7.usd"
SCHRAUBE8_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_8.usd"
SCHRAUBE9_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_9.usd"

BODY_PRIM_PATH = "/World/Body"
SCHRAUBE1_PRIM_PATH = "/World/Schraube1"
SCHRAUBE2_PRIM_PATH = "/World/Schraube2"
SCHRAUBE3_PRIM_PATH = "/World/Schraube3"
SCHRAUBE4_PRIM_PATH = "/World/Schraube4"
SCHRAUBE5_PRIM_PATH = "/World/Schraube5"
SCHRAUBE6_PRIM_PATH = "/World/Schraube6"
SCHRAUBE7_PRIM_PATH = "/World/Schraube7"
SCHRAUBE8_PRIM_PATH = "/World/Schraube8"
SCHRAUBE9_PRIM_PATH = "/World/Schraube9"
GROUND_PRIM_PATH = "/World/GroundPlane"

Z_OFFSET = 0.02
SDF_RESOLUTION = 1000

stage = omni.usd.get_context().get_stage()

# ----------------------------------------
# Hilfsfunktionen
# ----------------------------------------
def ensure_world():
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.DefinePrim("/World", "Xform")

def set_translation_and_rotation(prim, z_value, rotation_deg=Gf.Vec3f(-90.0, 0.0, 0.0)):
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()

    translate_op = None
    rotate_op = None

    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()

    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()

    current_translate = translate_op.Get()
    if current_translate is None:
        current_translate = Gf.Vec3d(0.0, 0.0, 0.0)

    translate_op.Set(Gf.Vec3d(current_translate[0], current_translate[1], z_value))
    rotate_op.Set(rotation_deg)

def apply_rigidbody(root_prim, mass=1.0):
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr().Set(mass)

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    physx_rb.CreateEnableCCDAttr().Set(True)
    physx_rb.CreateDisableGravityAttr().Set(False)
    

def is_mesh_prim(prim):
    return prim.IsA(UsdGeom.Mesh)

def apply_sdf_to_mesh_prim(mesh_prim, contact_offset=0.00003, rest_offset=0.00001):
    UsdPhysics.CollisionAPI.Apply(mesh_prim)

    approx_attr = mesh_prim.CreateAttribute(
        "physics:approximation",
        Sdf.ValueTypeNames.Token
    )
    approx_attr.Set("sdf")

    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
    physx_collision.CreateContactOffsetAttr().Set(contact_offset)
    physx_collision.CreateRestOffsetAttr().Set(rest_offset)

    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
    sdf_api.CreateSdfResolutionAttr().Set(SDF_RESOLUTION)

def remove_collision_from_root_if_present(root_prim):
    attr = root_prim.GetAttribute("physics:approximation")
    if attr.IsValid():
        attr.Clear()

def find_mesh_candidates_under_root(root_prim):
    candidates = []
    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim:
            continue
        if is_mesh_prim(prim):
            name_lower = prim.GetName().lower()
            path_lower = str(prim.GetPath()).lower()

            if (
                "collision" in name_lower
                or "collision" in path_lower
                or "collider" in name_lower
                or "collider" in path_lower
                or "visualcollision" in name_lower
                or "visualcollision" in path_lower
            ):
                candidates.append(prim)

    return candidates

def fallback_all_meshes_under_root(root_prim):
    meshes = []
    for prim in Usd.PrimRange(root_prim):
        if prim == root_prim:
            continue
        if is_mesh_prim(prim):
            meshes.append(prim)
    return meshes

def create_ground_plane():
    if stage.GetPrimAtPath(GROUND_PRIM_PATH).IsValid():
        return

    ground_xform = stage.DefinePrim(GROUND_PRIM_PATH, "Xform")
    xformable = UsdGeom.Xformable(ground_xform)

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, -0.009))

    orient_op = xformable.AddOrientOp()
    orient_op.Set(Gf.Quatf(0.70710677, 0.70710677, 0.0, 0.0))

    scale_op = xformable.AddScaleOp()
    scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    mesh_prim = stage.DefinePrim(f"{GROUND_PRIM_PATH}/CollisionMesh", "Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-2500.0, 0.0, -2500.0),
        Gf.Vec3f(2500.0, 0.0, -2500.0),
        Gf.Vec3f(2500.0, 0.0, 2500.0),
        Gf.Vec3f(-2500.0, 0.0, 2500.0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([3, 2, 1, 0])
    mesh.CreateNormalsAttr([
        Gf.Vec3f(0.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
    ])
    mesh.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])

    plane_prim = stage.DefinePrim(f"{GROUND_PRIM_PATH}/CollisionPlane", "Plane")
    UsdPhysics.CollisionAPI.Apply(plane_prim)
    plane_prim.CreateAttribute("axis", Sdf.ValueTypeNames.Token).Set("Y")
    plane_prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set("guide")

def setup_part(usd_path, prim_path, mass, rotation_deg=Gf.Vec3f(-90.0, 0.0, 0.0)):
    if not stage.GetPrimAtPath(prim_path).IsValid():
        stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Prim konnte nicht geladen werden: {prim_path} aus {usd_path}")

    UsdGeom.Imageable(root_prim).MakeVisible()
    set_translation_and_rotation(root_prim, Z_OFFSET, rotation_deg=rotation_deg)
    remove_collision_from_root_if_present(root_prim)
    apply_rigidbody(root_prim, mass=mass)

    collision_meshes = find_mesh_candidates_under_root(root_prim)

    if not collision_meshes:
        collision_meshes = fallback_all_meshes_under_root(root_prim)

    for mesh_prim in collision_meshes:
        UsdGeom.Imageable(mesh_prim).MakeVisible()
        apply_sdf_to_mesh_prim(mesh_prim)

    print(f"{prim_path} -> SDF applied to {len(collision_meshes)} mesh prim(s)")

# ----------------------------------------
# Setup
# ----------------------------------------
ensure_world()
create_ground_plane()

#setup_part(BODY_USD_PATH, BODY_PRIM_PATH, mass=1.0, rotation_deg=Gf.Vec3f(0.0, 0.0, 0.0))
setup_part(BODY_USD_PATH, BODY_PRIM_PATH, mass=1.0)
setup_part(SCHRAUBE1_USD_PATH, SCHRAUBE1_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE2_USD_PATH, SCHRAUBE2_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE3_USD_PATH, SCHRAUBE3_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE4_USD_PATH, SCHRAUBE4_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE5_USD_PATH, SCHRAUBE5_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE6_USD_PATH, SCHRAUBE6_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE7_USD_PATH, SCHRAUBE7_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE8_USD_PATH, SCHRAUBE8_PRIM_PATH, mass=0.1)
#setup_part(SCHRAUBE9_USD_PATH, SCHRAUBE9_PRIM_PATH, mass=0.1)
