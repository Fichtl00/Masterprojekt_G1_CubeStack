import omni
import omni.usd
import omni.isaac.core.utils.stage as stage_utils
import carb

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from omni.physx.scripts import utils
from omni.physx.scripts import physicsUtils


# ----------------------------------------
# Pfade für Files
# ----------------------------------------
BODY_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Body_2.usd"
SCHRAUBE1_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/M1030_Schraube.usd"
SCHRAUBE2_USD_PATH = "/home/omniverse-2/Documents/Toy_Bus/Schraube_3.6.usd"
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
PHYSICS_SCENE_PATH = "/World/PhysicsScene"

BODY_MATERIAL_PATH = "/World/PhysicsMaterials/BodyMaterial"
SCHRAUBE_MATERIAL_PATH = "/World/PhysicsMaterials/SchraubeMaterial"

Z_OFFSET = 0.02
SCHRAUBE_Z_EXTRA_OFFSET = -0.010
SCHRAUBE_SCALE = 0.001

SDF_RESOLUTION_BODY = 1400
SDF_RESOLUTION_SCHRAUBE = 1400

POSITION_ITERATIONS = 64
VELOCITY_ITERATIONS = 8
MAX_ANGULAR_VELOCITY = 5.0e5

stage = omni.usd.get_context().get_stage()


# ----------------------------------------
# Hilfsfunktionen
# ----------------------------------------
def ensure_world():
    if not stage.GetPrimAtPath("/World").IsValid():
        stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))


def ensure_physics_scene():
    if not stage.GetPrimAtPath(PHYSICS_SCENE_PATH).IsValid():
        scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    scene_prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

    physx_scene_api.CreateEnableCCDAttr(True)
    physx_scene_api.CreateEnableStabilizationAttr(True)
    physx_scene_api.CreateEnableGPUDynamicsAttr(False)
    physx_scene_api.CreateBroadphaseTypeAttr("MBP")
    physx_scene_api.CreateSolverTypeAttr("TGS")
    physx_scene_api.CreateMaxPositionIterationCountAttr().Set(POSITION_ITERATIONS)
    physx_scene_api.CreateMinPositionIterationCountAttr().Set(POSITION_ITERATIONS)
    physx_scene_api.CreateMaxVelocityIterationCountAttr().Set(VELOCITY_ITERATIONS)
    physx_scene_api.CreateMinVelocityIterationCountAttr().Set(VELOCITY_ITERATIONS)

    utils.set_physics_scene_asyncsimrender(scene_prim)

    settings = carb.settings.acquire_settings_interface()
    try:
        settings.set("/persistent/physics/numThreads", 0)
    except:
        pass


def ensure_physics_materials():
    UsdGeom.Scope.Define(stage, "/World/PhysicsMaterials")

    density = 7.85e3
    static_friction = 0.015
    dynamic_friction = 0.015

    if not stage.GetPrimAtPath(BODY_MATERIAL_PATH).IsValid():
        utils.addRigidBodyMaterial(
            stage,
            BODY_MATERIAL_PATH,
            density=density,
            staticFriction=static_friction,
            dynamicFriction=dynamic_friction,
        )

    if not stage.GetPrimAtPath(SCHRAUBE_MATERIAL_PATH).IsValid():
        utils.addRigidBodyMaterial(
            stage,
            SCHRAUBE_MATERIAL_PATH,
            density=density,
            staticFriction=static_friction,
            dynamicFriction=dynamic_friction,
        )


def set_transform(
    prim,
    z_value,
    rotation_deg=Gf.Vec3f(-90.0, 0.0, 0.0),
    scale_vec=Gf.Vec3f(1.0, 1.0, 1.0),
):
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()

    translate_op = None
    rotate_op = None
    scale_op = None

    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()

    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()

    if scale_op is None:
        scale_op = xformable.AddScaleOp()

    current_translate = translate_op.Get()
    if current_translate is None:
        current_translate = Gf.Vec3d(0.0, 0.0, 0.0)

    translate_op.Set(Gf.Vec3d(current_translate[0], current_translate[1], z_value))
    rotate_op.Set(rotation_deg)
    scale_op.Set(scale_vec)


def is_mesh_prim(prim):
    return prim.IsA(UsdGeom.Mesh)


def apply_sdf_to_mesh_prim(mesh_prim, sdf_resolution, contact_offset=0.0002, rest_offset=0.0):
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
    sdf_api.CreateSdfResolutionAttr().Set(sdf_resolution)


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


def apply_collision_and_material(root_prim, material_path, sdf_resolution):
    collision_meshes = find_mesh_candidates_under_root(root_prim)
    if not collision_meshes:
        collision_meshes = fallback_all_meshes_under_root(root_prim)

    for mesh_prim in collision_meshes:
        UsdGeom.Imageable(mesh_prim).MakeVisible()
        apply_sdf_to_mesh_prim(
            mesh_prim,
            sdf_resolution=sdf_resolution,
            contact_offset=0.0002,
            rest_offset=0.0,
        )

    physicsUtils.add_physics_material_to_prim(stage, root_prim, material_path)
    return collision_meshes


def make_body_like_bolt(root_prim):
    pass


def make_schraube_like_nut(root_prim, mass=0.1):
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    rigid_api.CreateKinematicEnabledAttr(False)

    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr().Set(mass)

    rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    rb_api.CreateSolverPositionIterationCountAttr(POSITION_ITERATIONS)
    rb_api.CreateSolverVelocityIterationCountAttr(VELOCITY_ITERATIONS)
    rb_api.CreateMaxAngularVelocityAttr().Set(MAX_ANGULAR_VELOCITY)
    rb_api.CreateSleepThresholdAttr().Set(0.0)
    rb_api.CreateEnableCCDAttr(True)
    rb_api.CreateDisableGravityAttr().Set(False)
    rb_api.CreateEnableGyroscopicForcesAttr().Set(True)
    rb_api.CreateAngularDampingAttr().Set(0.2)
    rb_api.CreateLinearDampingAttr().Set(0.05)


def create_ground_plane():
    if stage.GetPrimAtPath(GROUND_PRIM_PATH).IsValid():
        return
    utils.addPlaneCollider(stage, GROUND_PRIM_PATH, "Z")


def setup_body(usd_path, prim_path, rotation_deg=Gf.Vec3f(-90.0, 0.0, 0.0)):
    if not stage.GetPrimAtPath(prim_path).IsValid():
        stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Body konnte nicht geladen werden: {prim_path} aus {usd_path}")

    UsdGeom.Imageable(root_prim).MakeVisible()
    set_transform(
        root_prim,
        Z_OFFSET,
        rotation_deg=rotation_deg,
        scale_vec=Gf.Vec3f(1.0, 1.0, 1.0),
    )
    remove_collision_from_root_if_present(root_prim)

    meshes = apply_collision_and_material(
        root_prim,
        BODY_MATERIAL_PATH,
        sdf_resolution=SDF_RESOLUTION_BODY,
    )
    make_body_like_bolt(root_prim)

    print(f"{prim_path} -> BODY/static-thread setup applied to {len(meshes)} mesh prim(s)")


def setup_schraube(usd_path, prim_path, mass=0.1, rotation_deg=Gf.Vec3f(-90.0, 0.0, 0.0)):
    if not stage.GetPrimAtPath(prim_path).IsValid():
        stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Schraube konnte nicht geladen werden: {prim_path} aus {usd_path}")

    UsdGeom.Imageable(root_prim).MakeVisible()
    set_transform(
        root_prim,
        Z_OFFSET + SCHRAUBE_Z_EXTRA_OFFSET,
        rotation_deg=rotation_deg,
        scale_vec=Gf.Vec3f(SCHRAUBE_SCALE, SCHRAUBE_SCALE, SCHRAUBE_SCALE),
    )
    remove_collision_from_root_if_present(root_prim)

    meshes = apply_collision_and_material(
        root_prim,
        SCHRAUBE_MATERIAL_PATH,
        sdf_resolution=SDF_RESOLUTION_SCHRAUBE,
    )
    make_schraube_like_nut(root_prim, mass=mass)

    print(f"{prim_path} -> SCHRAUBE/NUT setup applied to {len(meshes)} mesh prim(s)")


# ----------------------------------------
# Setup
# ----------------------------------------
ensure_world()
ensure_physics_scene()
ensure_physics_materials()
create_ground_plane()

setup_body(BODY_USD_PATH, BODY_PRIM_PATH, rotation_deg=Gf.Vec3f(-90.0, 0.0, 0.0))

#setup_schraube(SCHRAUBE1_USD_PATH, SCHRAUBE1_PRIM_PATH, mass=0.1)
setup_schraube(SCHRAUBE2_USD_PATH, SCHRAUBE2_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE3_USD_PATH, SCHRAUBE3_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE4_USD_PATH, SCHRAUBE4_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE5_USD_PATH, SCHRAUBE5_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE6_USD_PATH, SCHRAUBE6_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE7_USD_PATH, SCHRAUBE7_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE8_USD_PATH, SCHRAUBE8_PRIM_PATH, mass=0.1)
#setup_schraube(SCHRAUBE9_USD_PATH, SCHRAUBE9_PRIM_PATH, mass=0.1)

omni.usd.get_context().get_selection().set_selected_prim_paths([], False)