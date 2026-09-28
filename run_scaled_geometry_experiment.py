"""Create and run a separate centered/scaled geometry experiment.

Run with --project pointing to the unmodified joint_geometry_project root.
Requires its existing dependencies. Never edits that source project.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import shutil
import subprocess
import sys


PREPROCESSING = '''import math
import torch
from torch import nn


class RelativePreprocessing(nn.Module):
    """Fixed facing arrays: BS normal +x, UE normal -x.

    Inputs in wavelength units. Geometry output: centered G0 (6),
    2*pi times [BS reference, UE reference, BS relative, UE relative]
    signed normal displacement (4), zero padding (8).
    Padding preserves the original 26-channel CNN and parameter count.
    These are phase-scale descriptors, not actual propagation phases.
    """
    def __init__(self,n_deformations,reference_index=0):
        super().__init__()
        if not 0 <= reference_index < n_deformations:
            raise ValueError('Invalid reference index')
        self.n_deformations=n_deformations
        self.reference_index=reference_index

    def forward(self,observations,geometry,nominal_geometry):
        if observations.ndim != 4:
            raise ValueError('Expected observations [B,4*M,NB,NU]')
        b,c,nb,nu=observations.shape
        m=self.n_deformations
        if c!=4*m or geometry.shape!=(b,m,nb,nu,6) or nominal_geometry.shape!=(nb,nu,6):
            raise ValueError('Incompatible observation or geometry dimensions')
        y=observations.reshape(b,m,4,nb,nu)
        yr=y[:,self.reference_index:self.reference_index+1]
        gr=geometry[:,self.reference_index:self.reference_index+1]
        g0=nominal_geometry[None,None].expand(b,m,-1,-1,-1)
        # Each descriptor replicates BS position across UE pairs, and vice versa.
        bs=nominal_geometry[...,:3]
        ue=nominal_geometry[...,3:]
        centered=torch.cat((bs-bs.mean((0,1),keepdim=True),
                            ue-ue.mean((0,1),keepdim=True)),dim=-1)
        # Fail instead of silently discarding tangential deformation.
        displacement=geometry-g0
        if torch.any(displacement[...,[1,2,4,5]].abs()>1e-5):
            raise ValueError('This experiment supports only the simulator normal-displacement model (+x/-x).')
        ref=gr-g0
        relative=geometry-gr
        signed=torch.stack((ref[...,0],-ref[...,3],relative[...,0],-relative[...,3]),dim=-1)
        descriptors=torch.cat((centered[None,None].expand_as(g0),
            2*math.pi*signed,torch.zeros_like(g0).repeat_interleave(2,dim=-1)[...,:8]),dim=-1)
        return torch.cat((yr.expand_as(y),y-yr),dim=2),descriptors
'''

TESTS = '''import math
import unittest
import torch
from model.estimator import ChannelEstimator


class RepresentationTests(unittest.TestCase):
    def test_signed_normals_centering_and_reference(self):
        model=ChannelEstimator(n_deformations=2,n_h_b=2,n_v_b=2,n_h_u=2,n_v_u=2,
            feature_channels=8,decoder_hidden_channels=8,dilations=(1,))
        y=torch.randn(1,8,4,4)
        g=model.nominal_geometry[None,None].repeat(1,2,1,1,1)
        g[:,0,...,0]+=.1;g[:,0,...,3]-=.2
        g[:,1,...,0]+=.3;g[:,1,...,3]-=.5
        x,d=model.preprocessing(y,g,model.nominal_geometry)
        torch.testing.assert_close(x[:,:,:4]+x[:,:,4:],y.reshape(1,2,4,4,4))
        torch.testing.assert_close(d[0,0,:,:,0:6].mean((0,1)),torch.zeros(6),atol=1e-6,rtol=0)
        expected=torch.tensor([[.1,.2,0,0],[.1,.2,.2,.3]])*2*math.pi
        torch.testing.assert_close(d[0,:,:, :,6:10],expected[:,None,None].expand(2,4,4,4),atol=3e-6,rtol=1e-5)
        self.assertEqual(torch.count_nonzero(d[...,10:]).item(),0)
        self.assertEqual(model.joint_inputs(y,g).shape,(1,2,26,4,4))

    def test_nominal_mask_checkpoint_and_gradients(self):
        torch.manual_seed(4)
        model=ChannelEstimator(n_deformations=2,n_h_b=2,n_v_b=2,n_h_u=2,n_v_u=2,
            feature_channels=8,decoder_hidden_channels=8,dilations=(1,))
        nominal=ChannelEstimator(**dict(model.config,geometry_mode='nominal'))
        nominal.load_state_dict(model.state_dict())
        y=torch.randn(1,8,4,4)
        g=model.nominal_geometry[None,None].repeat(1,2,1,1,1)
        changed=g.clone();changed[...,0]+=.1
        torch.testing.assert_close(nominal(y,g),nominal(y,changed),rtol=0,atol=0)
        torch.testing.assert_close(nominal(y,changed),model(y,changed,displacement_strength=0))
        changed.requires_grad_()
        model(y,changed).square().mean().backward()
        self.assertGreater(changed.grad[...,0].abs().sum().item(),0)
        restored=ChannelEstimator(**nominal.config)
        restored.load_state_dict(nominal.state_dict())
        torch.testing.assert_close(restored(y,g),nominal(y,g))
        self.assertEqual(model.config['architecture'],'joint_centered_normal_v1')

    def test_reject_tangential_motion(self):
        model=ChannelEstimator(n_deformations=1)
        g=model.nominal_geometry[None,None].clone();g[...,1]+=.1
        with self.assertRaises(ValueError):
            model(torch.zeros(1,4,25,25),g)


if __name__=='__main__':
    unittest.main()
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    source=args.project.resolve()
    estimator=(source/'model/estimator.py').read_text()
    if 'joint_antenna_v1' not in estimator or 'input_channels=26' not in estimator:
        raise ValueError('Expected the original 26-channel joint encoder project.')
    output=(args.output or source.parent/datetime.now().strftime('scaled_geometry_%Y%m%d_%H%M%S_%f')).resolve()
    if output==source or source in output.parents:
        raise ValueError('Choose an output folder outside the source project.')
    shutil.copytree(source,output,ignore=shutil.ignore_patterns('.git','artifacts','__pycache__','*.pyc'))
    (output/'model/relative_preprocessing.py').write_text(PREPROCESSING)
    (output/'model/estimator.py').write_text(estimator.replace('joint_antenna_v1','joint_centered_normal_v1'))
    # The old input-alignment test reconstructs XYZ descriptors and generates
    # unsupported tangential motion. Replace that test file with representation-specific tests.
    (output/'tests/test_joint.py').write_text(TESTS)
    (output/'REPRESENTATION_EXPERIMENT.md').write_text(
        'Centered nominal coordinates and signed normal displacements scaled by 2*pi.\n'
        'Four displacement values and eight zero padding slots preserve 26 input channels.\n'
        'Only the supplied facing-array, normal-displacement simulator is supported.\n'
        'Both full and nominal modes use centered nominal coordinates. Retrain from scratch.\n'
        'This tests the combined representation change; it does not isolate centering from scaling.\n'
        'Older README sections describing XYZ descriptors do not apply to this experiment.\n')
    (output/'representation_setup.json').write_text(json.dumps(dict(source=str(source),seed=args.seed,
        geometry='centered G0 + 2*pi signed normal displacement + zero padding',
        full_and_nominal_input_channels=26),indent=2))
    print(f'Prepared separate project: {output}',flush=True)
    if args.prepare_only:
        return
    def run(command,log):
        with (output/log).open('w') as stream:
            child=subprocess.Popen(command,cwd=output,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                   bufsize=1)
            for line in child.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            if child.wait():
                raise RuntimeError(f'Command failed; see {output/log}')
    run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'],'tests.log')
    paired=output/'artifacts/paired'
    command=[sys.executable,'-u','-m','main.compare','--device',args.device,'--seed',str(args.seed),
        '--epochs',str(args.epochs),'--batch-size',str(args.batch_size),'--output',str(paired)]
    if args.quick:
        command.extend(['--quick','--subcarriers','2'])
    run(command,'training.log')
    for mode in ('full','nominal'):
        run([sys.executable,'-u','-m','tests.diagnose_geometry','--run-dir',str(paired/mode),
             '--device',args.device,'--environments','2' if args.quick else '50','--seed','90401'],mode+'_diagnostic.log')
    archive=shutil.make_archive(str(output),'zip',root_dir=output)
    print(f'COMPLETE. Download report and project: {archive}',flush=True)


if __name__=='__main__':
    main()
