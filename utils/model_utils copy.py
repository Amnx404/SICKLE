from models import utae, pastis_unet3d, convlstm, convgru, fpn, utae_mid
import torch.nn as nn
import torch


class Build_model(nn.Module):
    def __init__(self, CFG):
        super(Build_model, self).__init__()
        self.CFG = CFG
        self.sat = list(CFG.satellites.keys())[0]
        self.model = self.get_model(self.sat)

    def forward(self, data):
        (images, dates) = data[self.sat]
        y_pred = self.model(images, batch_positions=dates)
        return y_pred
    
    def get_model(self,sat):
        config = self.CFG
        input_dim = len(config.satellites[sat]["bands"])
        if config.model == "utae":
            model = utae.UTAE(
                input_dim=input_dim,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.model == "utae_mid":
            model = utae_mid.UTAE(
                input_dim=input_dim,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=False,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.model == "unet3d":
            model = pastis_unet3d.UNet3D(
                in_channel=input_dim, n_classes=config.out_conv[-1], pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=input_dim,
                num_classes=config.out_conv[-1],
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=config.img_size,
                mid_conv=True,
                pad_value=config.pad_value, ## yei bhi 0 hai
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.out_conv[-1],
                input_size=config.img_size,
                input_dim=input_dim,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.out_conv[-1],
                input_size=config.img_size,
                input_dim=input_dim,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        else:
            raise NotImplementedError
        return model
class Fusion_model(Build_model):
    def __init__(self, CFG):
        super(Fusion_model, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        self.conv_final = nn.Conv2d(len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], CFG.num_classes,kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        image_dict = {}
        date_dict = {}
        for satellite in self.CFG.satellites.keys():
            (image_dict[satellite], date_dict[satellite]) = data[satellite]
        model = self.models[satellite]
        y_preds[satellite] = model(image_dict, batch_positions=date_dict)
            
    
        # here we cat data from UTAE --- implement attention here?
        # print(y_preds[self.sat].shape)
        y_pred = torch.cat(list(y_preds.values()), dim=1)  # Concatenating here
        y_pred = self.conv_final(y_pred)
        return y_pred

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelWiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelWiseAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        attention = self.conv(x)
        attention = torch.sigmoid(attention)
        return attention
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, channels):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.channels = channels
        assert channels % num_heads == 0, "channels should be divisible by num_heads"
        self.depth = channels // num_heads
        
        self.Wq = nn.Linear(channels, channels)
        self.Wk = nn.Linear(channels, channels)
        self.Wv = nn.Linear(channels, channels)
        self.dense = nn.Linear(channels, channels)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v):
        batch_size = q.size(0)
        
        q = self.split_heads(self.Wq(q), batch_size)
        k = self.split_heads(self.Wk(k), batch_size)
        v = self.split_heads(self.Wv(v), batch_size)
        
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(self.depth, dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.channels)
        
        return self.dense(output)


# class Fusion_model_PXL_ATTN(Build_model):
#     def __init__(self, CFG):
#     # def __init__(self, CFG, num_heads = 8):
#         super(Fusion_model_PXL_ATTN, self).__init__(CFG)
#         self.CFG = CFG
#         self.models = nn.ModuleDict()
#         for satellite in CFG.satellites.keys():
#             model = self.get_model(sat=satellite)
#             self.models[satellite] = model
        
#         self.attention_modules = nn.ModuleDict()
#         for satellite in CFG.satellites.keys():
#             attention_module = PixelWiseAttention(self.CFG.out_conv[-1])
#             # attention_module = MultiHeadAttention(num_heads, self.CFG.out_conv[-1])
#             self.attention_modules[satellite] = attention_module
        
#         self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

#     def forward(self, data):
#         y_preds = {}
#         attentions = {}
#         for satellite in self.CFG.satellites.keys():
#             (images, dates) = data[satellite]
#             model = self.models[satellite]
#             y_pred = model(images, batch_positions=dates)
#             y_preds[satellite] = y_pred
            
#             attention_module = self.attention_modules[satellite]
#             attention = attention_module(y_pred)
#             attentions[satellite] = attention
        
#         # Perform pixel-wise attention fusion
#         fused_features = torch.stack(list(y_preds.values()), dim=0)  # Shape: (num_satellites, batch_size, channels, height, width)
#         attention_weights = torch.stack(list(attentions.values()), dim=0)  # Shape: (num_satellites, batch_size, channels, height, width)
#         attention_weights = F.softmax(attention_weights, dim=0)
        
#         fused_features = (fused_features * attention_weights).sum(dim=0)
        
#         y_pred = self.conv_final(fused_features)
#         return y_pred

class Fusion_model_PXL_ATTN(Build_model):
    def __init__(self, CFG, num_heads=8):
        super(Fusion_model_PXL_ATTN, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model

        self.attention_modules = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            attention_module = MultiHeadAttention(num_heads, self.CFG.out_conv[-1])
            self.attention_modules[satellite] = attention_module

        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_pred = model(images, batch_positions=dates)
            y_preds[satellite] = y_pred
            
            attention_module = self.attention_modules[satellite]
            attention = attention_module(y_pred, y_pred, y_pred)
            attentions[satellite] = attention
        
        fused_features = torch.stack(list(attentions.values()), dim=0)
        fused_features = fused_features.mean(dim=0)
        
        y_pred = self.conv_final(fused_features)
        return y_pred
    
    
class Fusion_model_CONCAT_ATTN(Build_model):
    def __init__(self, CFG):
        super(Fusion_model_CONCAT_ATTN, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.attention = nn.Sequential(
            nn.Conv2d(len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.conv_final = nn.Conv2d(len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_preds[satellite] = model(images, batch_positions=dates)
        
        # Concatenate feature maps from all satellites
        y_pred = torch.cat(list(y_preds.values()), dim=1)
        
        # Apply attention
        attention_weights = self.attention(y_pred)
        y_pred = y_pred * attention_weights
        
        y_pred = self.conv_final(y_pred)
        return y_pred
    
class Fusion_model_CONCAT_ATTN_PIXELWISE(Build_model):
    def __init__(self, CFG):
        super(Fusion_model_CONCAT_ATTN_PIXELWISE, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.attention = nn.Sequential(
            nn.Conv2d(len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], kernel_size=1),
            nn.Sigmoid()
        )
        
        self.conv_final = nn.Conv2d(len(self.CFG.satellites.keys()) * self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_preds[satellite] = model(images, batch_positions=dates)
        
        # Concatenate feature maps from all satellites
        y_pred = torch.cat(list(y_preds.values()), dim=1)
        
        # Apply pixel-wise attention
        attention_weights = self.attention(y_pred)
        y_pred = y_pred * attention_weights
        
        y_pred = self.conv_final(y_pred)
        return y_pred