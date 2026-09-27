import torch
from torch import nn
from .relative_preprocessing import RelativePreprocessing
from .dilated_cnn import SharedObservationCNN
from .decoder import AntennaResidualDecoder
from .residual_reconstruction import ResidualReconstruction
from geometry.surfaces import generate_reference_geometry
from geometry.features import build_fixed_origin_features
import numpy as np


class ChannelEstimator(nn.Module):
    def __init__(self, n_deformations=8, n_h_b=5,n_v_b=5,n_h_u=5,n_v_u=5,
                 feature_channels=64,decoder_hidden_channels=64,
                 dilations=(1,2,4),residual_reference='mean',reference_index=0,
                 spacing_b=(0.125,0.125),spacing_u=(0.125,0.125),
                 nominal_geometry=None, geometry_mode='full', architecture='joint_antenna_v1'):
        super().__init__()
        if residual_reference != 'mean' or architecture != 'joint_antenna_v1':
            raise ValueError('This architecture requires mean residual targets.')
        if geometry_mode not in ('full','nominal'):
            raise ValueError('geometry_mode must be full or nominal.')
        if nominal_geometry is None:
            pb,pu,_,_ = generate_reference_geometry(n_h_b,n_v_b,*spacing_b,
                n_h_u,n_v_u,*spacing_u,1.0)
            nominal_geometry = build_fixed_origin_features([pb],[np.zeros_like(pb)],
                [pu],[np.zeros_like(pu)],1.0,pb.mean(axis=1))[0]
        nominal = torch.as_tensor(nominal_geometry,dtype=torch.float32)
        if nominal.shape != (n_h_b*n_v_b,n_h_u*n_v_u,6) or not torch.isfinite(nominal).all():
            raise ValueError('Invalid nominal geometry.')
        self.register_buffer('nominal_geometry',nominal.clone())
        self.config = dict(n_deformations=n_deformations,n_h_b=n_h_b,n_v_b=n_v_b,
            n_h_u=n_h_u,n_v_u=n_v_u,feature_channels=feature_channels,
            decoder_hidden_channels=decoder_hidden_channels,
            dilations=tuple(dilations),residual_reference=residual_reference,
            reference_index=reference_index,spacing_b=tuple(spacing_b),spacing_u=tuple(spacing_u),
            nominal_geometry=nominal.tolist(),geometry_mode=geometry_mode,architecture=architecture)
        self.geometry_mode = geometry_mode
        self.preprocessing = RelativePreprocessing(n_deformations,reference_index)
        self.cnn = SharedObservationCNN(feature_channels,dilations,input_channels=26)
        self.decoder = AntennaResidualDecoder(feature_channels,decoder_hidden_channels)
        self.reconstruction = ResidualReconstruction()

    def joint_inputs(self, observations, geometry, displacement_strength=1.0):
        x,g = self.preprocessing(observations,geometry,self.nominal_geometry)
        strength = displacement_strength if self.geometry_mode == 'full' else 0.0
        g = torch.cat((g[...,:6], strength*g[...,6:]), dim=-1)
        return torch.cat((x, g.permute(0,1,4,2,3)), dim=2)

    def forward(self,observations,geometry,return_auxiliary=False,displacement_strength=1.0):
        joint = self.joint_inputs(observations,geometry,displacement_strength)
        features = self.cnn(joint)
        residual = self.decoder(features.mean(dim=1))
        prediction = self.reconstruction(residual,observations)
        if return_auxiliary:
            return prediction,dict(residual_antenna=residual)
        return prediction
