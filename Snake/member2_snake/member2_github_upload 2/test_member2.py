"""Focused checks for target semantics, teacher isolation and checkpoint use."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from environment import SnakeEnv
from main import TrainConfig, as_tensor, evaluate_model, load_agent, train
from model import ConvActorCritic

class Member2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_terminal_and_time_limit_targets_have_no_gradient(self):
        nxt = torch.tensor([10.,10.], requires_grad=True)
        target = ConvActorCritic.td_target([2.,2.], nxt, [True,False], .9)
        torch.testing.assert_close(target, torch.tensor([2.,11.]))
        self.assertFalse(target.requires_grad)

    def test_bc_learns_explicit_action_and_checkpoint_preserves_predictions(self):
        env=SnakeEnv();obs,_=env.reset(seed=12)
        agent=ConvActorCritic(7,3,8)
        state=as_tensor(obs,agent.device)
        action=env.teacher_action()
        before=float(torch.softmax(agent(state)[0],dim=-1)[0,action].detach())
        for _ in range(30):agent.imitation_update(state,action)
        after=float(torch.softmax(agent(state)[0],dim=-1)[0,action].detach())
        self.assertGreater(after,before)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'model.pt'
            agent.save_checkpoint(file,episode=2,config=TrainConfig().checkpoint_payload())
            restored,cfg=load_agent(file)
            torch.testing.assert_close(agent(state)[0],restored(state)[0])
        env.close()

    def test_evaluation_never_uses_teacher_or_changes_parameters(self):
        agent=ConvActorCritic(7,3,8)
        weights={k:v.clone() for k,v in agent.state_dict().items()}
        with patch.object(SnakeEnv,'teacher_action',side_effect=AssertionError('teacher leak')):
            result,rows=evaluate_model(agent,TrainConfig(),seed_base=900000,episodes=3,return_episodes=True)
        self.assertEqual([r['episode_seed'] for r in rows],[900000,900001,900002])
        self.assertTrue(np.isfinite(result['reward']))
        for k,v in agent.state_dict().items():torch.testing.assert_close(v,weights[k])

    def test_exact_budget_no_teacher_ac_and_same_paired_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for teacher in [0,1]:
                cfg=TrainConfig(teacher_episodes=teacher,rl_steps=20,eval_every_steps=10,
                                eval_episodes=1,seed=7,output_dir=Path(tmp)/str(teacher))
                if teacher==0:
                    with patch.object(SnakeEnv,'teacher_action',side_effect=AssertionError('teacher leak')):
                        summary=train(cfg)
                else:summary=train(cfg)
                self.assertEqual(summary['rl_steps'],20)
                self.assertEqual(summary['demo_steps']==0,teacher==0)
                paths.append(cfg.output_dir/'checkpoints/initial.pt')
            a=torch.load(paths[0],weights_only=True)['model_state_dict']
            b=torch.load(paths[1],weights_only=True)['model_state_dict']
            for k in a:torch.testing.assert_close(a[k],b[k])

if __name__=='__main__':unittest.main()
