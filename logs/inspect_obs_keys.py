import os, sys, json, numpy as np, torch
repo='/home/cv25/siwon/sam2act_memdebug/sam2act'
sys.path.insert(0, repo)
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from yarr.agents.agent import ActResult
from sam2act.eval import load_agent
from sam2act.libs.peract.helpers import utils
from sam2act.utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2
from sam2act.utils.peract_utils import CAMERAS, IMAGE_SIZE
from sam2act.utils.rlbench_planning import EndEffectorPoseViaPlanning2

def to_torch_obs(obs_history, device):
    return {k: torch.tensor(np.array([vals]), device=device) for k, vals in obs_history.items()}
agent=load_agent(model_path='/home/cv25/siwon/sam2act_memdebug/sam2act/runs/sam2act_mb_stage2_mem11_bs12_20260421/model_plus_last.pth', eval_log_dir='/tmp/inspect_obs', device=0)
agent.load_clip()
obs_config = utils.create_obs_config(CAMERAS, [IMAGE_SIZE, IMAGE_SIZE], method_name='', use_mask_from_replay=False)
env = CustomMultiTaskRLBenchEnv2(task_classes=[task_file_to_task_class('put_block_back')], observation_config=obs_config, action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning2(), Discrete()), episode_length=25, dataset_root='./data_memory/test', headless=False, swap_task_every=1, include_lang_goal_in_obs=True)
env.eval=True
env.launch()
out={}
obs=env.reset_to_demo(0)
out['reset']={k:list(np.asarray(obs[k]).shape) for k in sorted(obs.keys()) if hasattr(obs[k],'__array__')}
out['reset_keys']=sorted(obs.keys())
obs_history={k:[np.array(v)] for k,v in obs.items()}
act=agent.act(-1,to_torch_obs(obs_history,'cuda:0'),deterministic=True)
trans=env.step(ActResult(action=np.asarray(act.action,dtype=np.float32)))
obs=trans.observation
out['step']={k:list(np.asarray(obs[k]).shape) for k in sorted(obs.keys()) if hasattr(obs[k],'__array__')}
out['step_keys']=sorted(obs.keys())
with open('/home/cv25/siwon/sam2act_memdebug/logs/inspect_obs_keys.json','w') as f:
    json.dump(out,f,indent=2)
env.shutdown()
