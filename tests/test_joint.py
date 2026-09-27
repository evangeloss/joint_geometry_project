import io
import unittest
import torch
from model.estimator import ChannelEstimator


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(2)
        self.model=ChannelEstimator(n_deformations=3,n_h_b=2,n_v_b=2,n_h_u=2,n_v_u=2,
            feature_channels=8,decoder_hidden_channels=8,dilations=(1,2))
        self.y=torch.randn(2,12,4,4)
        self.g=self.model.nominal_geometry[None,None].repeat(2,3,1,1,1)+torch.randn(2,3,4,4,6)*.1

    def test_input_alignment(self):
        joint=self.model.joint_inputs(self.y,self.g)
        self.assertEqual(joint.shape,(2,3,26,4,4))
        torch.testing.assert_close(joint[:,:,:4]+joint[:,:,4:8],self.y.reshape(2,3,4,4,4))
        geometry=joint[:,:,8:].permute(0,1,3,4,2)
        torch.testing.assert_close(geometry[...,:6]+geometry[...,6:12]+geometry[...,12:],self.g)

    def test_nominal_ignores_displacement_and_has_same_parameters(self):
        nominal=ChannelEstimator(**dict(self.model.config,geometry_mode='nominal'))
        nominal.load_state_dict(self.model.state_dict())
        torch.testing.assert_close(nominal(self.y,self.g),nominal(self.y,self.g+1),rtol=0,atol=0)
        torch.testing.assert_close(nominal(self.y,self.g),self.model(self.y,self.g,displacement_strength=0))
        self.assertEqual(sum(p.numel() for p in nominal.parameters()),sum(p.numel() for p in self.model.parameters()))

    def test_geometry_gradient_reaches_prediction(self):
        geometry=self.g.clone().requires_grad_()
        self.model(self.y,geometry).square().mean().backward()
        self.assertTrue(torch.isfinite(geometry.grad).all())
        self.assertGreater(geometry.grad.abs().sum().item(),0)
        stem=self.model.cnn.net[0].weight.grad
        self.assertGreater(stem[:,14:].abs().sum().item(),0)

    def test_zero_residual_returns_mean(self):
        torch.nn.init.zeros_(self.model.decoder.output_head.weight)
        torch.nn.init.zeros_(self.model.decoder.output_head.bias)
        torch.testing.assert_close(self.model(self.y,self.g),self.y.reshape(2,3,4,4,4).mean(1))

    def test_checkpoint_modes_persist(self):
        for mode in ('full','nominal'):
            model=ChannelEstimator(**dict(self.model.config,geometry_mode=mode))
            buffer=io.BytesIO()
            torch.save(dict(config=model.config,state=model.state_dict()),buffer)
            buffer.seek(0)
            ckpt=torch.load(buffer,weights_only=True)
            restored=ChannelEstimator(**ckpt['config'])
            restored.load_state_dict(ckpt['state'])
            self.assertEqual(restored.geometry_mode,mode)
            torch.testing.assert_close(restored(self.y,self.g),model(self.y,self.g))

    def test_nonreference_permutation(self):
        order=[0,2,1]
        yp=self.y.reshape(2,3,4,4,4)[:,order].reshape_as(self.y)
        torch.testing.assert_close(self.model(yp,self.g[:,order]),self.model(self.y,self.g))


if __name__=='__main__':
    unittest.main()
