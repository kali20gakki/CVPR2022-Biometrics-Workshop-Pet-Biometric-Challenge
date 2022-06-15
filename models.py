import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
from copy import deepcopy
from torch.nn import Parameter
from functools import partial
import math
from pooling import *
from resnext_ibn import resnext101_ibn_a
from timm.models.vision_transformer_hybrid import HybridEmbed    
from timm.models.layers import GroupNormAct

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class prediction_MLP1(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU() # resnetv2_101x1_mstage_multisim_b180_224_persudov3_bottleneck_nofreeze_gem_280e
            #nn.GELU(), # resnetv2_101x1_mstage_multisim_b176_224_persudov3_bottleneck_nofreeze_gem_240e_last_88021
            #nn.ReLU(inplace=True) 
        )

        self.layer2 = nn.Linear(hidden_dim, out_dim)

        self.layer1.apply(weights_init_kaiming)
        self.layer2.apply(weights_init_kaiming)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x 


class projection_MLP2(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 

class LinearNorm(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512):
        super(LinearNorm, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.fc.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class EmbeddingNet(nn.Module):
    def __init__(self, encoder_name='resnet50', proj_type = 'MLP', pretrained = True):
        super(EmbeddingNet, self).__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)

        if 'swin_base' in encoder_name or 'beit_large_patch16_224_in22k' in encoder_name:
            self.proj = prediction_MLP1(1024, 512, 1024) if proj_type == 'MLP' else projection_MLP2(1024, 512, 1024)

        elif 'swin_large' in encoder_name or 'convnext_large_in22ft1k' in encoder_name:
            self.proj = prediction_MLP1(1536, 768, 1536) if proj_type == 'MLP' else projection_MLP2(1536, 768, 1536)

        elif 'swin_small' in encoder_name or 'swin_tiny' in encoder_name or \
            'swin_v2_cr_small_224' in encoder_name or 'swin_s3_small_224' in encoder_name or \
            'beit_base_patch16_224_in22k' in encoder_name:
            self.proj = prediction_MLP1(768, 512, 256) if proj_type == 'MLP' else projection_MLP2(768, 512, 256)

        elif 'volo_d1_224' in encoder_name:
            self.proj = prediction_MLP1(384, 256, 256) if proj_type == 'MLP' else projection_MLP2(384, 256, 256)

        else:
            self.proj = prediction_MLP1(2048, 512, 2048) if proj_type == 'MLP' else projection_MLP2(2048, 512, 2048)
            print('use bottleneck')


    def forward(self, x):
        output = self.get_embedding(x)
        return output

    def get_embedding(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        return x

    def get_pair_embedding(self, x1, x2):
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        return out1, out2

    def get_pair_encoder_embedding(self, x1, x2):
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)

        return out1, out2


class EmbeddingNetWithBNNeck(nn.Module):
    def __init__(self, stride = 2, encoder_name='resnetv2_101x1_bitm', embedding_size = 2048, pretrained = True, act_layer=None):
        super(EmbeddingNetWithBNNeck, self).__init__()

        if encoder_name == 'resnext101_ibn_a':
            self.encoder = resnext101_ibn_a(pretrained=pretrained)
        else:
            if 'resnetv2_101x1' in encoder_name and act_layer == 'SiLU':
                self.encoder = timm.create_model(encoder_name, pretrained=pretrained, 
                                    num_classes=0, global_pool = '', drop_path_rate = 0.,
                                    norm_layer=partial(GroupNormAct, num_groups=32, act_layer=nn.SiLU))
                print('use SiLU ActLayer')
            else:
                self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, global_pool = '', drop_path_rate = 0.)

        print(f'EmbeddingNetWithBNNeck: {encoder_name}')

        self.pooling = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))

        if stride == 1:
            self.encoder.stages[3].blocks[0].downsample.conv.stride = (1,1)
            self.encoder.stages[3].blocks[0].conv2.stride = (1,1)
            print('set stage 3 stride = 1')
        elif stride == 2:
            print('keep stage 3 stride = 2')

        if encoder_name == 'resnext101_ibn_a':
            self.proj = nn.Sequential(
                nn.Linear(2048, 1024, bias=True),
                nn.BatchNorm1d(1024),
                nn.PReLU(),
                nn.Linear(1024, embedding_size, bias=True),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.encoder.num_features, self.encoder.num_features // 2, bias=True),
                nn.BatchNorm1d(self.encoder.num_features // 2),
                nn.PReLU(),
                nn.Linear(self.encoder.num_features // 2, self.encoder.num_features, bias=True),
            )

        # self.proj = nn.Sequential(
        #     nn.Linear(self.encoder.num_features, 1024, bias=True),
        #     nn.BatchNorm1d(1024),
        #     nn.PReLU(),
        #     nn.Linear(1024, embedding_size, bias=True),
        # )

        self.bnneck = nn.BatchNorm1d(embedding_size)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        emb, neck_out = self.get_embedding(x)
        return emb, neck_out

    def get_embedding(self, x):
        emb = self.encoder(x)
        emb = self.pooling(emb)
        emb = emb.view(emb.size(0), -1) # shape [N, C]
        emb = self.proj(emb)
        neck_out = self.bnneck(emb)

        return emb, neck_out

    def get_pair_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)
        return emb1, neck_out1, emb2, neck_out2

    def get_pair_encoder_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)

        #return neck_out1, neck_out2
        return emb1, emb2



class EmbeddingNetLocalkWithBNNeck(nn.Module):
    def __init__(self,stride, encoder_name='resnetv2_101x1_bitm', embedding_size = 2048, pretrained = True, act_layer=None):
        super(EmbeddingNetLocalkWithBNNeck, self).__init__()

        if 'res2net' in encoder_name or \
            'resnet' in encoder_name or \
            'resnest' in encoder_name or \
            'resnext' in encoder_name:
            out_indices=(3, 4)
        elif 'convnext' in encoder_name:
            out_indices=(2, 3)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, out_indices=out_indices, features_only=True)
        print(f'Feature channels: {self.encoder.feature_info.channels()}')

        if stride == 1:
            self.encoder.stages_3.blocks[0].downsample.conv.stride = (1,1)
            self.encoder.stages_3.blocks[0].conv2.stride = (1,1)
            print('set stages_3 stride = 1')

            # self.encoder.stages_2.blocks[0].downsample.conv.stride = (1,1)
            # self.encoder.stages_2.blocks[0].conv2.stride = (1,1)
            # print('set stages_2 stride = 1')

        elif stride == 2:
            print('keep stages_3 stride = 2')

        planes = self.encoder.feature_info.channels()[0]
        local_planes = planes // 2
        self.local_conv = nn.Conv2d(planes, local_planes, 1)
        self.local_bn = nn.BatchNorm2d(local_planes)
        self.local_bn.bias.requires_grad_(False)  # no shift

        self.pooling = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))

        self.proj = nn.Sequential(
            nn.Linear(self.encoder.feature_info.channels()[1], 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, embedding_size, bias=True),
        )

        self.bnneck = nn.BatchNorm1d(embedding_size)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        global_feat, local_feat, neck_out = self.get_embedding(x)
        return global_feat, local_feat, neck_out

    def get_embedding(self, x):
        local_feat, global_feat = self.encoder(x)

        # local branch
        local_feat = torch.mean(local_feat, -1, keepdim=True)
        local_feat = self.local_bn(self.local_conv(local_feat))
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)
        local_feat = l2_norm(local_feat, axis=-1)

        # global feat
        global_feat = self.pooling(global_feat)
        global_feat = global_feat.view(global_feat.size(0), -1) # shape [N, C]
        global_feat = self.proj(global_feat)

        neck_out = self.bnneck(global_feat)

        #print(f'global_feat = {global_feat.shape}, local_feat = {local_feat.shape}, neck_out = {neck_out.shape}')
        return l2_norm(global_feat), local_feat, neck_out


    def get_pair_embedding(self, x1, x2):
        global_feat1, local_feat1, neck_out1 = self.get_embedding(x1)
        global_feat2, local_feat2, neck_out2 = self.get_embedding(x2)

        return global_feat1, global_feat2

    def get_pair_encoder_embedding(self, x1, x2):
        global_feat1, local_feat1, neck_out1 = self.get_embedding(x1)
        global_feat2, local_feat2, neck_out2 = self.get_embedding(x2)

        #return neck_out1, neck_out2
        return global_feat1, global_feat2

class MultiStageEmbeddingNetWithBNNeckV2(nn.Module):
    def __init__(self, stride, encoder_name='resnetv2_101x1_bitm_in21k', embedding_size = 512, pretrained = True):
        super(MultiStageEmbeddingNetWithBNNeckV2, self).__init__()

        if 'res2net' in encoder_name or \
            'resnet' in encoder_name or \
            'resnest' in encoder_name or \
            'resnext' in encoder_name:
            out_indices=(3, 4)
        elif 'convnext' in encoder_name:
            out_indices=(2, 3)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, out_indices=out_indices, features_only=True)
        print(f'Feature channels: {self.encoder.feature_info.channels()}')

        print(f'MultiStageEmbeddingNetWithBNNeckV2: {encoder_name}')

        if stride == 1:
            self.encoder.stages_3.blocks[0].downsample.conv.stride = (1,1)
            self.encoder.stages_3.blocks[0].conv2.stride = (1,1)
            print('set stages_3 stride = 1')

        elif stride == 2:
            print('keep stages_3 stride = 2')

        self.pooling1 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))
        self.pooling2 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))

        self.proj = nn.Sequential(
            nn.Linear(sum(self.encoder.feature_info.channels()), embedding_size, bias=True),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
        )

        self.bnneck = nn.BatchNorm1d(embedding_size)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        emb, neck_out = self.get_embedding(x)
        return emb, neck_out

    def get_embedding(self, x):
        stage0_out, stage1_out = self.encoder(x)

        stage0_out = self.pooling1(stage0_out)
        stage1_out = self.pooling2(stage1_out)

        stage0_out = stage0_out.view(stage0_out.size(0), -1) # shape [N, C]
        stage1_out = stage1_out.view(stage1_out.size(0), -1) # shape [N, C]

        emb = self.proj(torch.cat((stage0_out, stage1_out), dim = -1))

        neck_out = self.bnneck(emb)

        return emb, neck_out

    def get_pair_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)
        return emb1, neck_out1, emb2, neck_out2

    def get_pair_encoder_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)

        #return neck_out1, neck_out2
        return emb1, emb2


class MultiStageEmbeddingNetWithBNNeck(nn.Module):
    def __init__(self, stride, encoder_name='resnetv2_101x1_bitm_in21k', embedding_size = 2048, pretrained = True):
        super(MultiStageEmbeddingNetWithBNNeck, self).__init__()

        if 'res2net' in encoder_name or \
            'resnet' in encoder_name or \
            'resnest' in encoder_name or \
            'resnext' in encoder_name:
            out_indices=(3, 4)
        elif 'convnext' in encoder_name:
            out_indices=(2, 3)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, out_indices=out_indices, features_only=True)
        print(f'Feature channels: {self.encoder.feature_info.channels()}')

        print(f'MultiStageEmbeddingNetWithBNNeck: {encoder_name}')

        #print(self.encoder)
        if stride == 1:
            self.encoder.stages_3.blocks[0].downsample.conv.stride = (1,1)
            self.encoder.stages_3.blocks[0].conv2.stride = (1,1)
            print('set stages_3 stride = 1')

        elif stride == 2:
            print('keep stages_3 stride = 2')

        self.pooling1 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))
        self.pooling2 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))

        self.proj = nn.Sequential(
            nn.Linear(sum(self.encoder.feature_info.channels()), 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, embedding_size, bias=True),
        )

        self.bnneck = nn.BatchNorm1d(embedding_size)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        emb, neck_out = self.get_embedding(x)
        return emb, neck_out

    def get_embedding(self, x):
        stage0_out, stage1_out = self.encoder(x)

        stage0_out = self.pooling1(stage0_out)
        stage1_out = self.pooling2(stage1_out)

        stage0_out = stage0_out.view(stage0_out.size(0), -1) # shape [N, C]
        stage1_out = stage1_out.view(stage1_out.size(0), -1) # shape [N, C]

        emb = self.proj(torch.cat((stage0_out, stage1_out), dim = -1))

        neck_out = self.bnneck(emb)

        return emb, neck_out

    def get_pair_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)
        return emb1, neck_out1, emb2, neck_out2

    def get_pair_encoder_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)

        #return neck_out1, neck_out2
        return emb1, emb2

class MultiStageEmbeddingNet(nn.Module):
    def __init__(self, stride, encoder_name='resnet50', pretrained = True):
        super(MultiStageEmbeddingNet, self).__init__()

        if 'res2net' in encoder_name or 'resnet' in encoder_name:
            out_indices=(3, 4)
        elif 'convnext' in encoder_name:
            out_indices=(2, 3)
        print('MultiStageEmbeddingNet :', encoder_name)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, out_indices=out_indices, features_only=True)
        
        if stride == 1:
            # self.encoder.stages_2.blocks[0].downsample.conv.stride = (1,1)
            # self.encoder.stages_2.blocks[0].conv2.stride = (1,1)
            # print('set stages_2 stride = 1')

            self.encoder.stages_3.blocks[0].downsample.conv.stride = (1,1)
            self.encoder.stages_3.blocks[0].conv2.stride = (1,1)
            print('set stages_3 stride = 1')

        elif stride == 2:
            print('keep stages_2 stride = 2')

        self.pooling1 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))
        self.pooling2 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))

        if 'convnext_base_in22k' in encoder_name:
            self.proj = prediction_MLP1(512 + 1024, 512, 1024)
            
        elif 'resnetv2_101x1' in encoder_name or \
            'resnext101_32x4d_gn_ws' in encoder_name or \
            'resnest101' in encoder_name:
            self.proj = prediction_MLP1(1024 + 2048, 1024, 2048)

        elif 'swin_large_patch4_window7_224_in22k' in encoder_name:
            self.proj = prediction_MLP1(1536 * 2, 1536, 2048)

        elif 'resnetv2_152x2_bitm_in21k' in encoder_name:
            self.proj = prediction_MLP1((1024 + 2048) * 2, 1024, 2048)


    def forward(self, x):
        out = self.get_embedding(x)
        return out

    def get_embedding(self, x):
        stage0_out, stage1_out = self.encoder(x)

        stage0_out = self.pooling1(stage0_out)
        stage1_out = self.pooling2(stage1_out)

        stage0_out = stage0_out.view(stage0_out.size(0), -1) # shape [N, C]
        stage1_out = stage1_out.view(stage1_out.size(0), -1) # shape [N, C]

        
        out = self.proj(torch.cat((stage0_out, stage1_out), dim = -1))

        return out


    def get_pair_embedding(self, x1, x2):
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        return out1, out2



class MultiStageEmbeddingNetV2(nn.Module):
    def __init__(self, stride, encoder_name='resnetv2_101x1_bitm', embedding_size = 2048, pretrained = True):
        super(MultiStageEmbeddingNetV2, self).__init__()

        if 'res2net' in encoder_name or \
            'resnet' in encoder_name or \
            'resnest' in encoder_name or \
            'resnext' in encoder_name:
            out_indices=(3, 4)
        elif 'convnext' in encoder_name:
            out_indices=(2, 3)

        print('MultiStageEmbeddingNetV2 :', encoder_name, ' embedding_size = ', embedding_size)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, out_indices=out_indices, features_only=True)
        print(f'Feature channels: {self.encoder.feature_info.channels()}')

        if stride == 1:
            self.encoder.stages_3.blocks[0].downsample.conv.stride = (1,1)
            self.encoder.stages_3.blocks[0].conv2.stride = (1,1)
            print('set stages_3 stride = 1')

        elif stride == 2:
            print('keep stages_3 stride = 2')

        self.pooling1 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))
        self.pooling2 = GeneralizedMeanPoolingP(norm=3, output_size=(1,1))

        self.proj = nn.Sequential(
            nn.Linear(sum(self.encoder.feature_info.channels()), embedding_size // 2, bias=True),
            nn.BatchNorm1d(embedding_size // 2),
            nn.PReLU(),
            nn.Linear(embedding_size // 2, embedding_size, bias=True),
        )

    def forward(self, x):
        out = self.get_embedding(x)
        return out

    def get_embedding(self, x):
        stage0_out, stage1_out = self.encoder(x)
        
        stage0_out = self.pooling1(stage0_out)
        stage1_out = self.pooling2(stage1_out)

        stage0_out = stage0_out.view(stage0_out.size(0), -1) # shape [N, C]
        stage1_out = stage1_out.view(stage1_out.size(0), -1) # shape [N, C]

        out = self.proj(torch.cat((stage0_out, stage1_out), dim = -1))
        #print(f'out = {out.shape}')
        return out


    def get_pair_embedding(self, x1, x2):
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        return out1, out2



class SingleStageEmbeddingNet(nn.Module):
    def __init__(self, encoder_name='vit_base_r50_s16_224_in21k', embedding_size = 1024, pretrained = True):
        super(SingleStageEmbeddingNet, self).__init__()

        print('SingleStageEmbeddingNet :', encoder_name)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
        print(self.encoder)
        print('encoder num_features = ', self.encoder.num_features, 'embedding_size = ', embedding_size)
        if encoder_name == 'vit_base_r50_s16_224_in21k':
            self.neck = nn.Sequential(
                    nn.Linear(self.encoder.num_features, embedding_size, bias=True),
                    nn.BatchNorm1d(embedding_size),
                    nn.PReLU()
                )
        elif encoder_name == 'vit_large_r50_s32_224_in21k':
            self.neck = nn.Sequential(
                    nn.Linear(self.encoder.num_features, embedding_size * 2, bias=True),
                    nn.BatchNorm1d(embedding_size * 2),
                    nn.PReLU()
                )

        elif encoder_name == 'twins_svt_large' or \
            encoder_name == 'swin_large_patch4_window7_224_in22k' or \
            encoder_name == 'beit_large_patch16_224_in22k' or \
            encoder_name == 'resnetv2_101x3_bitm_in21k':
            self.neck = nn.Sequential(
                    nn.Linear(self.encoder.num_features, embedding_size * 2, bias=True),
                    nn.BatchNorm1d(embedding_size * 2),
                    nn.PReLU()
                )
        else:
            self.neck = nn.Sequential(
                    nn.Linear(self.encoder.num_features, embedding_size, bias=True),
                    nn.BatchNorm1d(embedding_size),
                    nn.PReLU()
                )
        


    def forward(self, x):
        out = self.get_embedding(x)
        return out

    def get_embedding(self, x):
        out = self.encoder(x)
        out = self.neck(out)

        return out


    def get_pair_embedding(self, x1, x2):
        out1 = self.get_embedding(x1)
        out2 = self.get_embedding(x2)
        return out1, out2



class SingleStageEmbeddingNetWithBNNeck(nn.Module):
    def __init__(self, encoder_name='swin_large_patch4_window7_224_in22k', embedding_size = 2048, pretrained = True):
        super(SingleStageEmbeddingNetWithBNNeck, self).__init__()

        print('SingleStageEmbeddingNetWithBNNeck :', encoder_name)

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)

        print('encoder num_features = ', self.encoder.num_features, 'embedding_size = ', embedding_size)
        # if encoder_name == 'vit_base_r50_s16_224_in21k':
        #     self.proj = nn.Sequential(
        #             nn.Linear(self.encoder.num_features, 1024, bias=True),
        #             nn.BatchNorm1d(1024),
        #             nn.PReLU(),
        #             nn.Linear(1024, embedding_size, bias=True),
        #         )
        # elif encoder_name == 'vit_large_r50_s32_224_in21k':
        #     self.proj = nn.Sequential(
        #             nn.Linear(self.encoder.num_features, 1024, bias=True),
        #             nn.BatchNorm1d(1024),
        #             nn.PReLU(),
        #             nn.Linear(1024, embedding_size, bias=True),
        #         )

        # elif encoder_name == 'twins_svt_large' or \
        #     encoder_name == 'swin_large_patch4_window7_224_in22k' or \
        #     encoder_name == 'beit_large_patch16_224_in22k':
        #     self.proj = nn.Sequential(
        #             nn.Linear(self.encoder.num_features, 1024, bias=True),
        #             nn.BatchNorm1d(1024),
        #             nn.PReLU(),
        #             nn.Linear(1024, embedding_size, bias=True),
        #         )
        # else:
        #     self.proj = nn.Sequential(
        #             nn.Linear(self.encoder.num_features, 1024, bias=True),
        #             nn.BatchNorm1d(1024),
        #             nn.PReLU(),
        #             nn.Linear(1024, embedding_size, bias=True),
        #         )

        self.proj = nn.Sequential(
                nn.Linear(self.encoder.num_features, self.encoder.num_features // 2, bias=True),
                nn.BatchNorm1d(self.encoder.num_features // 2),
                nn.PReLU(),
                nn.Linear(self.encoder.num_features // 2, self.encoder.num_features, bias=True),
            )

        self.bnneck = nn.BatchNorm1d(self.encoder.num_features)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        emb, neck_out = self.get_embedding(x)
        return emb, neck_out

    def get_embedding(self, x):
        out = self.encoder(x)
        emb = self.proj(out)
        neck_out = self.bnneck(emb)

        return emb, neck_out

    def get_pair_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)
        return emb1, neck_out1, emb2, neck_out2

    def get_pair_encoder_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)

        #return neck_out1, neck_out2
        return emb1, emb2


class SingleStageEmbeddingNetWithBNNeckV2(nn.Module):
    def __init__(self, encoder_name='swinv2_large_window12_192_22k', pretrained = True):
        super(SingleStageEmbeddingNetWithBNNeckV2, self).__init__()
        self.encoder_name = encoder_name
        print('SingleStageEmbeddingNetWithBNNeckV2 :', encoder_name)
        

        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0)
        
        print('encoder num_features = ', self.encoder.num_features)

        self.proj = nn.Sequential(
                nn.Linear(self.encoder.num_features, self.encoder.num_features // 2, bias=True),
                nn.BatchNorm1d(self.encoder.num_features // 2),
                nn.PReLU(),
                nn.Linear(self.encoder.num_features // 2, self.encoder.num_features, bias=True),
            )

        self.bnneck = nn.BatchNorm1d(self.encoder.num_features)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        emb, neck_out = self.get_embedding(x)
        return emb, neck_out

    def get_embedding(self, x):
        out = self.encoder(x)
        emb = self.proj(out)
        neck_out = self.bnneck(emb)

        return emb, neck_out

    def get_pair_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)
        return emb1, neck_out1, emb2, neck_out2

    def get_pair_encoder_embedding(self, x1, x2):
        emb1, neck_out1 = self.get_embedding(x1)
        emb2, neck_out2 = self.get_embedding(x2)

        #return neck_out1, neck_out2
        return emb1, emb2

##################################################################################################

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)   
        return ret
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

    

class MultiAtrousModule(nn.Module):
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        
        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0], padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1], padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2], padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x

class SpatialAttention2d(nn.Module):
    def __init__(self, in_c):
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, dim=1)
         
        x = self.act1(x)
        x = self.conv2(x)
        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score   

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):

        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused  


class MultiScaleEmbeddingDOLG(nn.Module):
    def __init__(self, encoder_name='resnetv2_101x1_bitm_in21k', embedding_size = 512, stride = 1, pretrained = True):
        super(MultiScaleEmbeddingDOLG, self).__init__()
        self.embedding_size = embedding_size
        print(f'MultiScaleEmbeddingDOLG : {encoder_name}')
        if 'res2net' in encoder_name or 'resnet' in encoder_name:
            out_indices=(3, 4)
        elif 'convnext' in encoder_name:
            out_indices=(2, 3)
        
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, global_pool = '', out_indices=out_indices, features_only=True)
        self.planes_list = self.encoder.feature_info.channels()

        if 'resnetv2' in encoder_name and stride == 1:
            self.encoder.stem_conv.stride = (1,1)
            print('stride = 1')
        else:
            print(f'stride = {stride}')

        feature_dim_l_g = self.planes_list[-2]
        fusion_out = 2 * feature_dim_l_g

        self.global_pool = GeM(p_trainable=True)
        self.fusion_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                nn.PReLU()
            )

        dilations = [3,6,9]
        self.mam = MultiAtrousModule(self.planes_list[-2], feature_dim_l_g, dilations)
        self.conv_g = nn.Conv2d(self.planes_list[-1],feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()


    def forward(self, x):
        emb = self.get_embedding(x)
        return emb

    def get_pair_embedding(self, x1, x2):
        emb1 = self.get_embedding(x1)
        emb2 = self.get_embedding(x2)
        
        return emb1, emb2

    def get_embedding(self, x):
        feat_l, feat_g = self.encoder(x)
        #print(feat_l.shape, feat_g.shape)
        
        feat_l = self.mam(feat_l)
        feat_l, _ = self.attention2d(feat_l)

        feat_g = self.conv_g(feat_g)
        feat_g = self.bn_g(feat_g)
        feat_g = self.act_g(feat_g)

        feat_g = self.global_pool(feat_g)
        feat_g = feat_g[:,:,0,0]
        #print(feat_l.shape, feat_g.shape)

        feat_fused = self.fusion(feat_l, feat_g)
        feat_fused = self.fusion_pool(feat_fused)
        feat_fused = feat_fused[:,:,0,0]        
        
        emb = self.neck(feat_fused)
        #print(emb.shape)

        return emb

if __name__ == '__main__':
    net = EmbeddingNetLocalkWithBNNeck()
    x = torch.randn([2, 3, 224, 224])
    out1 = net(x)
    #print(out1.shape)
    # global_feat_list, local_feat_list = net(x)
    # print(global_feat_list[0].shape, global_feat_list[1].shape, local_feat_list[0].shape, local_feat_list[1].shape)