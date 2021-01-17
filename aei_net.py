import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

import os
import random
from PIL import Image
from skimage import io
from skimage.color import gray2rgb

from torch.utils.data import Dataset

import warnings

from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import resnet101
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

class MultilevelAttributesEncoder(nn.Module):
    def __init__(self):
        super(MultilevelAttributesEncoder, self).__init__()
        self.Encoder_channel = [3, 32, 128, 512, 512]
        self.stride_sizes_encoder = [2, 4, 4, 2]
        self.Encoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.Conv2d(self.Encoder_channel[i], 
                          self.Encoder_channel[i+1],
                          kernel_size=4, 
                          stride=self.stride_sizes_encoder[i],
                          padding=1),
                nn.BatchNorm2d(self.Encoder_channel[i+1]),
                nn.LeakyReLU(0.1)
            )for i in range(4)})

        self.Decoder_inchannel = [512, 1024, 256]
        self.Decoder_outchannel = [512, 128, 32]
        self.stride_sizes_decoder = [2, 4, 4]
        self.padding_sizes_decoder = [1, 0, 0]
        self.Decoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.ConvTranspose2d(self.Decoder_inchannel[i], 
                                   self.Decoder_outchannel[i], 
                                   kernel_size=4, 
                                   stride=self.stride_sizes_decoder[i],
                                   padding=self.padding_sizes_decoder[i]),
                nn.BatchNorm2d(self.Decoder_outchannel[i]),
                nn.LeakyReLU(0.1)
            )for i in range(3)})

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        arr_x = []
        #encode_x = []
        for i in range(4):
            x = self.Encoder[f'layer_{i}'](x)
            arr_x.append(x)
        
        arr_y = []
        arr_y.append(arr_x[3])
        y = arr_x[3]
        for i in range(3):
            y = self.Decoder[f'layer_{i}'](y)
            y = torch.cat((y, arr_x[2-i]), 1)
            arr_y.append(y)

        arr_y.append(self.Upsample(y))

        return arr_y

class ADD(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, z_id_size=256):
        super(ADD, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id):
        h_bar = self.BNorm(h_in)
        M = self.sigmoid(self.conv_f(h_bar))

        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)

        I = r_id*h_bar + beta_id

        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        A = r_att * h_bar + beta_att

        h_out = (1-M)*A + M*I

        return h_out

class ADDResBlock(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, h_outchannel):
        super(ADDResBlock, self).__init__()

        self.h_inchannel = h_inchannel
        self.z_inchannel = z_inchannel
        self.h_outchannel = h_outchannel

        self.add1 = ADD(h_inchannel, z_inchannel)
        self.add2 = ADD(h_inchannel, z_inchannel)

        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        if self.h_inchannel != self.h_outchannel:
            self.add3 = ADD(h_inchannel, z_inchannel)
            self.conv3 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, h_in, z_att, z_id):
        x1 = self.activation(self.add1(h_in, z_att, z_id))
        x1 = self.conv1(x1)
        x1 = self.activation(self.add2(x1, z_att, z_id))
        x1 = self.conv2(x1)

        x2 = h_in
        if self.h_inchannel != self.h_outchannel:
            x2 = self.activation(self.add3(h_in, z_att, z_id))
            x2 = self.conv3(x2)

        return x1 + x2


class ADDGenerator(nn.Module):
    def __init__(self, z_id_size):
        super(ADDGenerator, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 512, kernel_size=2, stride=1, padding=0)
        
        self.h_inchannel = [512, 512, 256, 128, 64]
        self.z_inchannel = [512, 1024, 256, 64, 64]
        self.h_outchannel = [512, 256, 128, 64, 3]
        self.upsample_scale = [2, 4, 4, 2]
        
        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(5)})


    def forward(self, z_id, z_att):
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))

        for i in range(4):
            x = self.model[f"layer_{i}"](x, z_att[i], z_id)
            x = nn.UpsamplingBilinear2d(scale_factor=self.upsample_scale[i])(x)
        x = self.model["layer_4"](x, z_att[4], z_id)

        return nn.Sigmoid()(x)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        kernel_window = 4
        padw = int(np.ceil((kernel_window - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kernel_window, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kernel_window, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kernel_window, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kernel_window, stride=1, padding=padw)]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor
        self.opt = opt

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, target_is_real, for_discriminator=True):
        if for_discriminator:
            if target_is_real:
                minval = torch.min(input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
            else:
                minval = torch.min(-input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
        else:
            loss = -torch.mean(input)

        return loss
        
        
    def __call__(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class AEI_Loss(nn.Module):
    def __init__(self):
        super(AEI_Loss, self).__init__()

        self.att_weight = 10
        self.id_weight = 5
        self.rec_weight = 10

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def att_loss(self, z_att_X, z_att_Y):
        loss = 0
        for i in range(5):
            loss += self.l2(z_att_X[i], z_att_Y[i])
        return 0.5*loss

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def rec_loss(self, X, Y, same):
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return 0.5*self.l2(X, Y)

    def forward(self, X, Y, z_att_X, z_att_Y, z_id_X, z_id_Y, same):

        att_loss = self.att_loss(z_att_X, z_att_Y)
        id_loss = self.id_loss(z_id_X, z_id_Y)
        rec_loss = self.rec_loss(X, Y, same)

        return self.att_weight*att_loss + self.id_weight*id_loss + self.rec_weight*rec_loss, att_loss, id_loss, rec_loss            

class AEINet(pl.LightningModule):
    def __init__(self):
        super(AEINet, self).__init__()
  
        self.G = ADDGenerator(256)
        self.E = MultilevelAttributesEncoder()
        self.D = MultiscaleDiscriminator(3)

        self.Z = resnet101(num_classes=256)
        self.Z.load_state_dict(torch.load('/aeinet_files/Arcface.pth', map_location='cpu'))

        self.Loss_GAN = GANLoss()
        self.Loss_E_G = AEI_Loss()


    def forward(self, target_img, source_img):
        z_id = self.Z(F.interpolate(source_img, size=112, mode='bilinear'))
        z_id = F.normalize(z_id)
        z_id = z_id.detach()

        feature_map = self.E(target_img)

        output = self.G(z_id, feature_map)

        output_z_id = self.Z(F.interpolate(output, size=112, mode='bilinear'))
        output_z_id = F.normalize(output_z_id)
        output_feature_map = self.E(output)
        return output, z_id, output_z_id, feature_map, output_feature_map


    def training_step(self, batch, batch_idx, optimizer_idx):
        target_img, source_img, same = batch

        if optimizer_idx == 0:
            output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

            self.generated_img = output

            output_multi_scale_val = self.D(output)
            loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
            loss_E_G, loss_att, loss_id, loss_rec = self.Loss_E_G(target_img, output, feature_map, output_feature_map, z_id,
                                                             output_z_id, same)

            loss_G = loss_E_G + loss_GAN

            sample_img = torch.rand((1, 3, 128, 128))
  
            self.logger.experiment.add_scalar("Loss G", loss_G.item(), self.global_step)
            self.logger.experiment.add_scalar("Attribute Loss", loss_att.item(), self.global_step)
            self.logger.experiment.add_scalar("ID Loss", loss_id.item(), self.global_step)
            self.logger.experiment.add_scalar("Reconstruction Loss", loss_rec.item(), self.global_step)

            self.logger.experiment.add_scalar("GAN Loss", loss_GAN.item(), self.global_step)

            return loss_G

        else:
            multi_scale_val = self.D(target_img)
            output_multi_scale_val = self.D(self.generated_img.detach())

            loss_D_fake = self.Loss_GAN(multi_scale_val, True)
            loss_D_real = self.Loss_GAN(output_multi_scale_val, False)

            loss_D = loss_D_fake + loss_D_real

            self.logger.experiment.add_scalar("Loss D", loss_D.item(), self.global_step)
            return loss_D

    def validation_step(self, batch, batch_idx):
        target_img, source_img, same = batch

        output, z_id, output_z_id, feature_map, output_feature_map = self(target_img, source_img)

        self.generated_img = output

        output_multi_scale_val = self.D(output)
        loss_GAN = self.Loss_GAN(output_multi_scale_val, True, for_discriminator=False)
        loss_E_G, loss_att, loss_id, loss_rec = self.Loss_E_G(target_img, output, feature_map, output_feature_map,
                                                              z_id, output_z_id, same)
        loss_G = loss_E_G + loss_GAN
        return {"loss": loss_G, 'target': target_img[0].cpu(), 'source': source_img[0].cpu(),  "output": output[0].cpu(), }

    def validation_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        validation_image = []
        for x in outputs:
            validation_image = validation_image + [x['target'], x['source'], x["output"]]
        validation_image = torchvision.utils.make_grid(validation_image, nrow=3)

        self.logger.experiment.add_scalar("Validation Loss", loss.item(), self.global_step)
        self.logger.experiment.add_image("Validation Image", validation_image, self.global_step)

        return {"loss": loss, "image": validation_image, }


    def configure_optimizers(self):
        lr_g = 4e-4
        lr_d = 4e-4
        b1 = 0
        b2 = 0.999

        opt_g = torch.optim.Adam(list(self.G.parameters()) + list(self.E.parameters()), lr=lr_g, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
            ])
        dataset = AEI_Dataset('/aeinet_files/img_ceba_100k', transform=transform)
        return DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor(),
        ])
        dataset = AEI_Val_Dataset('/aeinet_files/valset_dir', transform=transform)
        return DataLoader(dataset, batch_size=1, shuffle=False)        

class AEI_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Dataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transform = transform


    def __getitem__(self, index):
        l = len(self.files)
        s_idx = index%l
        if index >= int(0.99*l):
            f_idx = s_idx

        else:
            f_idx = random.randrange(l)


        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transform is not None:
            f_img = self.transform(f_img)
            s_img = self.transform(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files)


class AEI_Val_Dataset(Dataset):
    def __init__(self, root, transform=None):
        super(AEI_Val_Dataset, self).__init__()
        self.root = root
        self.files = [
            os.path.join(path, filename)
            for path, dirs, files in os.walk(root)
            for filename in files
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
        ]
        self.transfrom = transform

    def __getitem__(self, index):
        l = len(self.files)

        f_idx = index // l
        s_idx = index % l

        if f_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        f_img = Image.open(self.files[f_idx])
        s_img = Image.open(self.files[s_idx])

        f_img = f_img.convert('RGB')
        s_img = s_img.convert('RGB')

        if self.transfrom is not None:
            f_img = self.transfrom(f_img)
            s_img = self.transfrom(s_img)

        return f_img, s_img, same

    def __len__(self):
        return len(self.files) * len(self.files)        

        