from copy import deepcopy
from pathlib import Path
import torch
from torch import nn


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([2*feature_dim, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        feature_dim = 128
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.maxpool = torch.nn.MaxPool1d(2,2)

    def forward(self, desc0, desc1):
        desc0 = self.maxpool(desc0.permute(0,2,1))
        desc1 = self.maxpool(desc1.permute(0,2,1))
        desc0 = desc0.permute(0,2,1)
        desc1 = desc1.permute(0,2,1)
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def log_optimal_transport_zvec(scores, zA, zB, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = zA.unsqueeze(0).unsqueeze(-1)
    bins1 = zB.unsqueeze(0).unsqueeze(0)
    alpha = torch.ones((b,1,1), device=scores.device)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z




def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1



class Sinkhorn_wGNN(nn.Module):
    """Sinkhorn Matcher
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        GNN_layers = ['self', 'cross'] * 1
        descriptor_dim = 256

        self.gnn = AttentionalGNN(
            descriptor_dim,
            GNN_layers)

        self.final_proj = nn.Conv1d(
            128,
            128,
            kernel_size=1,
            bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        if self.config['use_pretrain_weights']:
            assert self.config['weights'] in ['indoor', 'outdoor']
            path = Path(__file__).parent
            path = path / 'weights/sinkhorn_{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        output = {
            'matches0':[],
            'matches1':[],
            'matches_scores0':[],
            'matches_scores1':[],
            'scores':[],
        }
        B = len(data['descriptors0'])
        for b in range(B):
            desc0, desc1 = data['descriptors0'][b], data['descriptors1'][b]
            #kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            desc0 = desc0.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)

            if desc0.shape[1] == 128:
                desc0 = torch.cat((desc0,desc0), dim=1)
                desc1 = torch.cat((desc1,desc1), dim=1)
            else:
                assert desc0.shape[1] == 256

            desc0 = 2*torch.nn.functional.normalize(desc0, p=2, dim=1)
            desc1 = 2*torch.nn.functional.normalize(desc1, p=2, dim=1)

            # Multi-layer Transformer network.
            desc0, desc1 = self.gnn(desc0, desc1)

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            descriptor_dim = mdesc0.shape[1]

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / descriptor_dim**.5

            # Run the optimal transport.
            scores = log_optimal_transport(
                scores, self.bin_score,
                iters=self.config['sinkhorn_iterations'])

            scores = torch.nn.functional.softmax(scores, dim=2)

            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            output['matches0'].append(indices0)
            output['matches1'].append(indices1)
            output['matches_scores0'].append(mscores0)
            output['matches_scores1'].append(mscores1)
            output['scores'].append(scores[0])

        return output


class Sinkhorn_wZ(nn.Module):
    """Sinkhorn Matcher
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        descriptor_dim = 256

        self.final_proj = nn.Conv1d(
            256,
            256,
            kernel_size=1,
            bias=True)

        self.z = nn.Sequential(
            nn.Linear(2*descriptor_dim, descriptor_dim),
            nn.ReLU(),
            nn.Linear(descriptor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if self.config['use_pretrain_weights']:
            assert self.config['weights'] in ['indoor', 'outdoor']
            path = Path(__file__).parent
            path = path / 'weights/sinkhorn_wz_{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        output = {
            'matches0':[],
            'matches1':[],
            'matches_scores0':[],
            'matches_scores1':[],
            'scores':[],
        }
        B = len(data['descriptors0'])
        for b in range(B):
            desc0, desc1 = data['descriptors0'][b], data['descriptors1'][b]
            #kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            desc0 = desc0.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)

            if desc0.shape[1] == 128:
                desc0 = torch.cat((desc0,desc0), dim=1)
                desc1 = torch.cat((desc1,desc1), dim=1)
            else:
                assert desc0.shape[1] == 256

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # get z value estimate
            m0 = torch.mean(mdesc0,dim=2)
            m1 = torch.mean(mdesc0,dim=2)
            m = torch.cat((m0,m1), dim=1)
            z = self.z(m)
            z = z[0,0]

            self.bin_score = z

            descriptor_dim = mdesc0.shape[1]

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / descriptor_dim**.5

            # Run the optimal transport.
            scores = log_optimal_transport(
                scores, z,
                iters=self.config['sinkhorn_iterations'])

            scores = torch.nn.functional.softmax(scores, dim=2)

            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            output['matches0'].append(indices0)
            output['matches1'].append(indices1)
            output['matches_scores0'].append(mscores0)
            output['matches_scores1'].append(mscores1)
            output['scores'].append(scores[0])

        return output





class Sinkhorn_wZatt(nn.Module):
    """Sinkhorn Matcher
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        descriptor_dim = 256

        self.final_proj = nn.Conv1d(
            256,
            256,
            kernel_size=1,
            bias=True)

        self.attention = nn.Sequential(
                #nn.Tanh(),
                #nn.Dropout(0.5),
                nn.Linear(512,64),
                nn.ReLU(),
                nn.Linear(64,1),
        )

        self.z = nn.Sequential(
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.ReLU(),
            nn.Linear(descriptor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if self.config['use_pretrain_weights']:
            assert self.config['weights'] in ['indoor', 'outdoor']
            path = Path(__file__).parent
            path = path / 'weights/sinkhorn_wz_{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        output = {
            'matches0':[],
            'matches1':[],
            'matches_scores0':[],
            'matches_scores1':[],
            'scores':[],
        }
        B = len(data['descriptors0'])
        for b in range(B):
            desc0, desc1 = data['descriptors0'][b], data['descriptors1'][b]
            #kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            desc0 = desc0.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)

            if desc0.shape[1] == 128:
                desc0 = torch.cat((desc0,desc0), dim=1)
                desc1 = torch.cat((desc1,desc1), dim=1)
            else:
                assert desc0.shape[1] == 256

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # get z value estimate
            nA = mdesc0.shape[2]
            nB = mdesc1.shape[2]
            tmp_A = mdesc0.unsqueeze(-1).repeat((1,1,1,nB)) # (1,256, nA, nB)
            tmp_B = mdesc1.unsqueeze(2).repeat((1,1,nA,1)) # (1,256, nA, nB)
            tmp_A = tmp_A.view((1,256,-1))
            tmp_B = tmp_B.view((1,256,-1))

            input_attention = torch.cat((tmp_A, tmp_B), dim=1)
            input_attention = input_attention.permute(0,2,1)
            attention_weights = self.attention(input_attention)
            attention_weights = attention_weights.view(1,nA,nB,1)

            tmp_A = tmp_A.view((1,256,nA,nB))
            tmp_B = tmp_B.view((1,256,nA,nB))

            # get z - A
            prob_A = torch.nn.functional.softmax(attention_weights[0,:,:,0],
                                                 dim=1)
            prob_A = prob_A.unsqueeze(0).unsqueeze(0).repeat((1,256,1,1))
            zfeat_A = torch.mul(tmp_B,prob_A)
            zfeat_A = torch.sum(zfeat_A,dim=3)
            zfeat_A = zfeat_A.permute(0,2,1)

            zA = self.z(zfeat_A)
            zA = zA[0,:,0]

            # get z - B
            prob_B = torch.nn.functional.softmax(attention_weights[0,:,:,0],
                                                 dim=0)
            prob_B = prob_B.unsqueeze(0).unsqueeze(0).repeat((1,256,1,1))
            zfeat_B = torch.mul(tmp_A,prob_B)
            zfeat_B = torch.sum(zfeat_B,dim=2)
            zfeat_B = zfeat_B.permute(0,2,1)

            zB = self.z(zfeat_B)
            zB = zB[0,:,0]

            self.bin_score = zA.mean()

            descriptor_dim = mdesc0.shape[1]

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / descriptor_dim**.5

            # Run the optimal transport.
            scores = log_optimal_transport_zvec(
                scores, zA, zB,
                iters=self.config['sinkhorn_iterations'])

            scores = torch.nn.functional.softmax(scores, dim=2)

            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            output['matches0'].append(indices0)
            output['matches1'].append(indices1)
            output['matches_scores0'].append(mscores0)
            output['matches_scores1'].append(mscores1)
            output['scores'].append(scores[0])

        return output


class Sinkhorn_wZatt_Big(nn.Module):
    """Sinkhorn Matcher
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        descriptor_dim = 256

        self.final_proj = nn.Conv1d(
            256,
            256,
            kernel_size=1,
            bias=True)

        self.attention = nn.Sequential(
                #nn.Tanh(),
                #nn.Dropout(0.5),
                nn.Linear(512,256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Linear(64,1),
        )

        self.z = nn.Sequential(
            nn.Linear(2*descriptor_dim, descriptor_dim),
            nn.ReLU(),
            nn.Linear(descriptor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if self.config['use_pretrain_weights']:
            assert self.config['weights'] in ['indoor', 'outdoor']
            path = Path(__file__).parent
            path = path / 'weights/sinkhorn_wz_{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        output = {
            'matches0':[],
            'matches1':[],
            'matches_scores0':[],
            'matches_scores1':[],
            'scores':[],
        }
        B = len(data['descriptors0'])
        for b in range(B):
            desc0, desc1 = data['descriptors0'][b], data['descriptors1'][b]
            #kpts0, kpts1 = data['keypoints0'], data['keypoints1']
            desc0 = desc0.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)

            if desc0.shape[1] == 128:
                desc0 = torch.cat((desc0,desc0), dim=1)
                desc1 = torch.cat((desc1,desc1), dim=1)
            else:
                assert desc0.shape[1] == 256

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # get z value estimate
            nA = mdesc0.shape[2]
            nB = mdesc1.shape[2]
            tmp_A = mdesc0.unsqueeze(-1).repeat((1,1,1,nB)) # (1,256, nA, nB)
            tmp_B = mdesc1.unsqueeze(2).repeat((1,1,nA,1)) # (1,256, nA, nB)
            tmp_A = tmp_A.view((1,256,-1))
            tmp_B = tmp_B.view((1,256,-1))

            input_attention = torch.cat((tmp_A, tmp_B), dim=1)
            input_attention = input_attention.permute(0,2,1)
            attention_weights = self.attention(input_attention)
            attention_weights = attention_weights.view(1,nA,nB,1)

            tmp_A = tmp_A.view((1,256,nA,nB))
            tmp_B = tmp_B.view((1,256,nA,nB))

            # get z - A
            prob_A = torch.nn.functional.softmax(attention_weights[0,:,:,0],
                                                 dim=1)
            prob_A = prob_A.unsqueeze(0).unsqueeze(0).repeat((1,256,1,1))
            zfeat_A = torch.mul(tmp_B,prob_A)
            zfeat_A = torch.sum(zfeat_A,dim=3)
            zfeat_A = zfeat_A.permute(0,2,1)
            zfeat_A = torch.cat((zfeat_A, mdesc0.permute(0,2,1)), dim=2)

            zA = self.z(zfeat_A)
            zA = zA[0,:,0]

            # get z - B
            prob_B = torch.nn.functional.softmax(attention_weights[0,:,:,0],
                                                 dim=0)
            prob_B = prob_B.unsqueeze(0).unsqueeze(0).repeat((1,256,1,1))
            zfeat_B = torch.mul(tmp_A,prob_B)
            zfeat_B = torch.sum(zfeat_B,dim=2)
            zfeat_B = zfeat_B.permute(0,2,1)
            zfeat_B = torch.cat((zfeat_B, mdesc1.permute(0,2,1)), dim=2)

            zB = self.z(zfeat_B)
            zB = zB[0,:,0]

            self.bin_score = zA.mean()

            descriptor_dim = mdesc0.shape[1]

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / descriptor_dim**.5

            # Run the optimal transport.
            scores = log_optimal_transport_zvec(
                scores, zA, zB,
                iters=self.config['sinkhorn_iterations'])

            scores = torch.nn.functional.softmax(scores, dim=2)

            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            output['matches0'].append(indices0)
            output['matches1'].append(indices1)
            output['matches_scores0'].append(mscores0)
            output['matches_scores1'].append(mscores1)
            output['scores'].append(scores[0])

        return output

    def get_z_scores(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        output = {
            'zA':[],
            'zB':[],
        }
        B = len(data['descriptors0'])
        for b in range(B):
            desc0, desc1 = data['descriptors0'][b], data['descriptors1'][b]
            desc0 = desc0.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)

            if desc0.shape[1] == 128:
                desc0 = torch.cat((desc0,desc0), dim=1)
                desc1 = torch.cat((desc1,desc1), dim=1)
            else:
                assert desc0.shape[1] == 256

            # Final MLP projection.
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

            # get z value estimate
            nA = mdesc0.shape[2]
            nB = mdesc1.shape[2]
            tmp_A = mdesc0.unsqueeze(-1).repeat((1,1,1,nB)) # (1,256, nA, nB)
            tmp_B = mdesc1.unsqueeze(2).repeat((1,1,nA,1)) # (1,256, nA, nB)
            tmp_A = tmp_A.view((1,256,-1))
            tmp_B = tmp_B.view((1,256,-1))

            input_attention = torch.cat((tmp_A, tmp_B), dim=1)
            input_attention = input_attention.permute(0,2,1)
            attention_weights = self.attention(input_attention)
            attention_weights = attention_weights.view(1,nA,nB,1)

            tmp_A = tmp_A.view((1,256,nA,nB))
            tmp_B = tmp_B.view((1,256,nA,nB))

            # get z - A
            prob_A = torch.nn.functional.softmax(attention_weights[0,:,:,0],
                                                 dim=1)
            prob_A = prob_A.unsqueeze(0).unsqueeze(0).repeat((1,256,1,1))
            zfeat_A = torch.mul(tmp_B,prob_A)
            zfeat_A = torch.sum(zfeat_A,dim=3)
            zfeat_A = zfeat_A.permute(0,2,1)
            zfeat_A = torch.cat((zfeat_A, mdesc0.permute(0,2,1)), dim=2)

            zA = self.z(zfeat_A)
            zA = zA[0,:,0]

            # get z - B
            prob_B = torch.nn.functional.softmax(attention_weights[0,:,:,0],
                                                 dim=0)
            prob_B = prob_B.unsqueeze(0).unsqueeze(0).repeat((1,256,1,1))
            zfeat_B = torch.mul(tmp_A,prob_B)
            zfeat_B = torch.sum(zfeat_B,dim=2)
            zfeat_B = zfeat_B.permute(0,2,1)
            zfeat_B = torch.cat((zfeat_B, mdesc1.permute(0,2,1)), dim=2)

            zB = self.z(zfeat_B)
            zB = zB[0,:,0]

            output['zA'].append(zA)
            output['zB'].append(zB)

        return output
 


