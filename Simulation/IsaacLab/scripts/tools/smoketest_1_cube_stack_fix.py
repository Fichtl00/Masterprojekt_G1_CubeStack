import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--enable_pinocchio", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.enable_pinocchio:
    import pinocchio
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place
print(">>> STEP 0: import ok")
spec = gym.spec("Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack")
cfg = spec.kwargs["env_cfg_entry_point"]()
print(">>> STEP 1: cfg built, xr anchor_pos=", cfg.xr.anchor_pos)
print(">>> STEP 2: arm stiffness=", cfg.scene.robot.actuators["left_arm"].stiffness)
env = gym.make("Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-1_cube_stack", cfg=cfg)
print(">>> STEP 3: env made")
env.reset()
print(">>> STEP 4: env reset ok")
env.close()
print(">>> STEP 5: done")
simulation_app.close()
