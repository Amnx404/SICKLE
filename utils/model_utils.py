from models import utae, pastis_unet3d, convlstm, convgru, fpn, utae_mid
import torch.nn as nn
import torch
import torch.nn.functional as F

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
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            # print(images.shape, dates.shape)
            y_preds[satellite] = model(images, batch_positions=dates)
            
    
        # here we cat data from UTAE --- implement attention here?
        # print(y_preds[self.sat].shape)
        y_pred = torch.cat(list(y_preds.values()), dim=1)  # Concatenating here
        y_pred = self.conv_final(y_pred)
        return y_pred

##################################  1 #############################################
class PixelWiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelWiseAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        # Xavier Initialisation
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        attention = self.conv(x)
        attention = torch.sigmoid(attention)
        return attention

class Fusion_model_PXL_ATTN(Build_model):
    def __init__(self, CFG):
    # def __init__(self, CFG, num_heads = 8):
        super(Fusion_model_PXL_ATTN, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.attention_modules = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            attention_module = PixelWiseAttention(self.CFG.out_conv[-1])
            # attention_module = MultiHeadAttention(num_heads, self.CFG.out_conv[-1])
            self.attention_modules[satellite] = attention_module
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_pred = model(images, batch_positions=dates)
            print(f"Shape of y_pred from {satellite}: {y_pred.shape}")  # Add this line
            y_preds[satellite] = y_pred
            
            attention_module = self.attention_modules[satellite]
            attention = attention_module(y_pred)
            print(f"Shape of attention output from {satellite}: {attention.shape}")  # Add this line
            attentions[satellite] = attention
        
        # Perform pixel-wise attention fusion
        fused_features = torch.stack(list(y_preds.values()), dim=0)  # Shape: (num_satellites, batch_size, channels, height, width)
        attention_weights = torch.stack(list(attentions.values()), dim=0)  # Shape: (num_satellites, batch_size, channels, height, width)
        print(f"Shape of fused_features: {fused_features.shape}")
        print(f"Shape of attention_weights: {attention_weights.shape}")
    
        attention_weights = F.softmax(attention_weights, dim=0)
        
        fused_features = (fused_features * attention_weights).sum(dim=0)
        print(f"Shape of final fused_features after weighting: {fused_features.shape}")
        y_pred = self.conv_final(fused_features)
        return y_pred 
# ################## 2 ########################################  
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
        
        # Xavier Initialisation
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.xavier_uniform_(self.attention[2].weight)
        
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

# ################## 3 ########################################  
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
        
        # Xavier Initialisation
        nn.init.xavier_uniform_(self.attention[0].weight)
        nn.init.xavier_uniform_(self.attention[2].weight)
        
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

########################## 4 #################################
class fusion_model_pxl_extralayers(Build_model):
    def __init__(self, CFG):
        super(fusion_model_pxl_extralayers, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.attention_modules = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            attention_module = PixelWiseAttention(self.CFG.out_conv[-1])
            self.attention_modules[satellite] = attention_module
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

        # Additional convolutional layers for increased complexity
        self.conv_extra1 = nn.Conv2d(self.CFG.out_conv[-1], self.CFG.out_conv[-1], kernel_size=3, stride=1, padding=1)
        self.conv_extra2 = nn.Conv2d(self.CFG.out_conv[-1], self.CFG.out_conv[-1], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        y_preds = {}
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_pred = model(images, batch_positions=dates)
            y_preds[satellite] = y_pred
            
            attention_module = self.attention_modules[satellite]
            attention = attention_module(y_pred)
            attentions[satellite] = attention
        
        # Perform pixel-wise attention fusion
        fused_features = torch.stack(list(y_preds.values()), dim=0)  # Shape: (num_satellites, batch_size, channels, height, width)
        attention_weights = torch.stack(list(attentions.values()), dim=0)  # Shape: (num_satellites, batch_size, channels, height, width)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        # Additional convolutional layers for increased complexity
        fused_features = self.relu(self.conv_extra1(fused_features))
        fused_features = self.relu(self.conv_extra2(fused_features))
        
        y_pred = self.conv_final(fused_features)
        return y_pred

########################## 5 #####################################
class CrossAttention(nn.Module):
    def __init__(self, num_features):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.key_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.value_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
        # Xavier Initialisation
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, query, key, value):
        query = self.query_conv(query)
        key = self.key_conv(key)
        value = self.value_conv(value)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = self.softmax(attention_scores)

        attended_value = torch.matmul(attention_scores, value)
        return attended_value

class CrossFusionModel(Build_model):
    def __init__(self, CFG):
        super(CrossFusionModel, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.cross_attention = CrossAttention(self.CFG.out_conv[-1])
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_pred = model(images, batch_positions=dates)
            y_preds[satellite] = y_pred
        
        # Perform cross-attention fusion
        satellite_keys = list(y_preds.keys())
        fused_features = y_preds[satellite_keys[0]]  # Initialize with the first satellite's features
        
        for i in range(1, len(satellite_keys)):
            query = fused_features
            key = y_preds[satellite_keys[i]]
            value = y_preds[satellite_keys[i]]
            
            attended_value = self.cross_attention(query, key, value)
            fused_features = fused_features + attended_value
        
        y_pred = self.conv_final(fused_features)
        return y_pred


class MultiHeadPixelAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadPixelAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(in_channels, num_heads)
        
        # Xavier Initialisation
        nn.init.xavier_uniform_(self.attention.in_proj_weight)

    def forward(self, x):
        # Reshape input tensor to (sequence_length, batch_size, channels)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height * width).permute(2, 0, 1)
        
        # Apply multi-head attention
        attn_output, _ = self.attention(x, x, x)
        
        # Reshape attention output back to (batch_size, channels, height, width)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, height, width)
        
        return attn_output
    
    
    
    
    
#Best model : self attn for each satellite then multiply all attn and satellite on stacking
class MultiheadFusionModel(Build_model):
    def __init__(self, CFG):
        super(MultiheadFusionModel, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.attention_modules = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            attention_module = MultiHeadPixelAttention(self.CFG.out_conv[-1], num_heads=4)
            self.attention_modules[satellite] = attention_module
            # Xavier Initialisation
            nn.init.xavier_uniform_(self.attention_modules[satellite].attention.in_proj_weight)
        
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
            attention = attention_module(y_pred)
            
            attentions[satellite] = attention
        
        # Perform pixel-wise attention fusion
        fused_features = torch.stack(list(y_preds.values()), dim=0)
        attention_weights = torch.stack(list(attentions.values()), dim=0)
    
        attention_weights = F.softmax(attention_weights, dim=0)
        
        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        y_pred = self.conv_final(fused_features)
        return y_pred
    
class CHN_ATTN(Build_model):
    def __init__(self, CFG):
        super(CHN_ATTN, self).__init__(CFG)
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
        
        self.attention_modules = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            attention_module = SEBlock(self.CFG.out_conv[-1])
            self.attention_modules[satellite] = attention_module
            
            # Xavier Initialisation
            nn.init.xavier_uniform_(self.attention_modules[satellite].excitation[0].weight)
            nn.init.xavier_uniform_(self.attention_modules[satellite].excitation[2].weight)
        
        # Example integration of a DilatedConvBlock
        self.dilated_conv_block = DilatedConvBlock(self.CFG.out_conv[-1], self.CFG.out_conv[-1])

        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            model = self.models[satellite]
            y_pred = model(images, batch_positions=dates)
            
            # Apply dilated convolution here if appropriate
            y_pred = self.dilated_conv_block(y_pred)
            
            attention_module = self.attention_modules[satellite]
            attention = attention_module(y_pred)
            y_preds[satellite] = y_pred
            attentions[satellite] = attention
        
        fused_features = torch.stack(list(y_preds.values()), dim=0)
        attention_weights = torch.stack(list(attentions.values()), dim=0)
        attention_weights = F.softmax(attention_weights, dim=0)
        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        y_pred = self.conv_final(fused_features)
        return y_pred

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y)
        if y.numel() == b * c:  # Check if the number of elements matches b*c
            y = y.view(b, c, 1, 1)
        else:
            raise RuntimeError("Excitation output has incorrect size: expected {}, got {}".format(b*c, y.numel()))
        return x * y


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=2):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        return F.relu(self.conv(x))

class CombinedFusionModel(Build_model):
    def __init__(self, CFG):
        super(CombinedFusionModel, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        
        # Initialize each satellite's model and multi-head self-attention module
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
            self_attention = MultiHeadPixelAttention(CFG.out_conv[-1], num_heads=8)
            self.attention_modules[satellite] = self_attention
        
        self.cross_attention = CrossAttention(self.CFG.out_conv[-1])
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        
        # First apply self-attention for each satellite independently
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            base_output = self.models[satellite](images, batch_positions=dates)
            y_pred = self.attention_modules[satellite](base_output)
            y_preds[satellite] = y_pred
        
        # Perform cross-attention fusion
        satellite_keys = list(y_preds.keys())
        fused_features = y_preds[satellite_keys[0]]  # Initialize with the first satellite's features
        
        for i in range(1, len(satellite_keys)):
            query = fused_features
            key = y_preds[satellite_keys[i]]
            value = y_preds[satellite_keys[i]]
            
            attended_value = self.cross_attention(query, key, value)
            fused_features = fused_features + attended_value
        
        y_pred = self.conv_final(fused_features)
        return y_pred
    
class CombinedFusionModel2(Build_model):
    def __init__(self, CFG):
        super(CombinedFusionModel2, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        
        # Initialize each satellite's model and multi-head self-attention module
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
            self_attention = MultiHeadPixelAttention(CFG.out_conv[-1], num_heads=8)
            self.attention_modules[satellite] = self_attention
            # layer_norm = nn.LayerNorm(CFG.out_conv[-1])
            # self.attention_modules[satellite] = nn.Sequential(self_attention, layer_norm)
        
        self.cross_attention = MultiHeadCrossAttention(self.CFG.out_conv[-1], num_heads=8)
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        
        # First apply self-attention for each satellite independently
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            base_output = self.models[satellite](images, batch_positions=dates)
            y_pred = self.attention_modules[satellite](base_output)
            y_preds[satellite] = y_pred
        
        # Perform cross-attention fusion
        satellite_keys = list(y_preds.keys())
        fused_features = y_preds[satellite_keys[0]]  # Initialize with the first satellite's features
        
        for i in range(1, len(satellite_keys)):
            query = fused_features
            key = y_preds[satellite_keys[i]]
            value = y_preds[satellite_keys[i]]
            
            attended_value = self.cross_attention(query, key, value)
            fused_features = fused_features + attended_value
        
        y_pred = self.conv_final(fused_features)
        return y_pred

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_features, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = num_features // num_heads
        
        # Ensure the division is valid
        assert self.dim_per_head * num_heads == num_features, "num_features must be divisible by num_heads"

        self.query_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.key_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.value_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Xavier Initialisation for all weights
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()
        # print("hi")

        # Apply the convolutions to the input
        query = self.query_conv(query)
        key = self.key_conv(key)
        value = self.value_conv(value)
        
        # Split the features into multiple heads
        query = self.split_heads(query, self.num_heads)
        key = self.split_heads(key, self.num_heads)
        value = self.split_heads(value, self.num_heads)

        # Compute the dot product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.dim_per_head ** 0.5
        attention_scores = self.softmax(attention_scores)

        # Apply attention to the value vector
        attended_value = torch.matmul(attention_scores, value)

        # Concatenate the heads together
        attended_value = self.combine_heads(attended_value)
        
        # Reshape to the original size
        attended_value = attended_value.view(batch_size, channels, height, width)
        
        return attended_value

    def split_heads(self, x, num_heads):
        batch_size, channels, height, width = x.size()
        new_shape = batch_size, num_heads, self.dim_per_head, height * width
        x = x.view(*new_shape).permute(0, 1, 3, 2)  # (batch_size, num_heads, seq_len, depth)
        return x

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, depth = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = batch_size, -1, seq_len * depth  # flatten the last two dimensions
        return x.view(*new_shape)

#########################################################################################################################################################


class PixelWiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelWiseAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        attention = self.conv(x)
        attention = torch.sigmoid(attention)
        return x * attention

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=2):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        return F.relu(self.conv(x))

class EnhancedFusionModel(Build_model):
    def __init__(self, CFG):
        super(EnhancedFusionModel, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(satellite)
            self.models[satellite] = model
        
        self.attention_modules = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            self.attention_modules[satellite] = nn.Sequential(
                PixelWiseAttention(CFG.out_conv[-1]),
                SEBlock(CFG.out_conv[-1])
            )
        
        self.dilated_conv = DilatedConvBlock(CFG.out_conv[-1], CFG.out_conv[-1])
        self.conv1 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            images, dates = data[satellite]
            model = self.models[satellite]
            y_pred = model(images, batch_positions=dates)
            
            attention_module = self.attention_modules[satellite]
            attention = attention_module(y_pred)
            y_preds[satellite] = y_pred
            attentions[satellite] = attention
        
        fused_features = torch.stack(list(y_preds.values()), dim=0)
        attention_weights = torch.stack(list(attentions.values()), dim=0)
        attention_weights = F.softmax(attention_weights, dim=0)
        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        fused_features = self.dilated_conv(fused_features)
        fused_features = F.relu(self.conv1(fused_features))
        
        y_pred = self.conv2(fused_features)
        return y_pred

    def get_model(self, satellite):
        return self.models.get(satellite, None)
    
    
    
###############################################################################################################################################################################################################################################################

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
        return x * attention

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # Linear projections
        query = self.query(x).view(batch_size, self.num_heads, self.head_dim, height, width)
        key = self.key(x).view(batch_size, self.num_heads, self.head_dim, height, width)
        value = self.value(x).view(batch_size, self.num_heads, self.head_dim, height, width)
        
        # Transpose spatial dimensions
        query = query.permute(0, 1, 3, 4, 2).contiguous().view(batch_size * self.num_heads, height * width, self.head_dim)
        key = key.permute(0, 1, 3, 4, 2).contiguous().view(batch_size * self.num_heads, height * width, self.head_dim)
        value = value.permute(0, 1, 3, 4, 2).contiguous().view(batch_size * self.num_heads, height * width, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.matmul(attention, value)
        out = out.view(batch_size, self.num_heads, height, width, self.head_dim)
        out = out.permute(0, 1, 4, 2, 3).contiguous().view(batch_size, -1, height, width)
        out = self.out(out)
        
        return out

class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)


    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        gates = torch.sigmoid(self.conv(x))
        return gates * x1 + (1 - gates) * (x2 + x3)

class EnhancedFusionModel1(Build_model):
    def __init__(self, CFG):
        super(EnhancedFusionModel1, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(satellite)
            self.models[satellite] = model

        self.pixel_attention = PixelWiseAttention(CFG.out_conv[-1])
        self.multi_head_attention = MultiHeadAttention(CFG.out_conv[-1], num_heads=4)
        self.gated_fusion = GatedFusion(CFG.out_conv[-1])

        self.conv1 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_final = nn.Conv2d(CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)


    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            images, dates = data[satellite]
            model = self.models[satellite]
            y_preds[satellite] = model(images, batch_positions=dates)

        l8_features = y_preds['L8']
        s1_features = y_preds['S1'] 
        s2_features = y_preds['S2']

        l8_features = self.pixel_attention(l8_features)
        s1_features = self.pixel_attention(s1_features)
        s2_features = self.pixel_attention(s2_features)

        fused_features = self.gated_fusion(l8_features, s1_features, s2_features)
        fused_features = self.multi_head_attention(fused_features)

        fused_features = self.relu(self.conv1(fused_features))
        fused_features = self.relu(self.conv2(fused_features))

        y_pred = self.conv_final(fused_features)
        return y_pred

    def get_model(self, satellite):
        config = self.CFG
        input_dim = len(config.satellites[satellite]["bands"])
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
        else:
            raise NotImplementedError
        return model
    
    
    
#############################################################################
# CROSS_ALL_NORM (Final 1)
# plot gradients to see if they are vanishing
# Rms norm
class CrossAttentionFusionBasic(Build_model):
    def __init__(self, CFG):
        super(CrossAttentionFusionBasic, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        self.crossattention_modules = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        
        # Initialize each satellite's model and multi-head self-attention module
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
            cross_attention = MultiHeadCrossAttention(CFG.out_conv[-1], num_heads=4)
            self.crossattention_modules[satellite] = cross_attention
            self_attention = MultiHeadPixelAttention(CFG.out_conv[-1], num_heads=4)
            self.attention_modules[satellite] = self_attention
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)
        
        self.ln = nn.LayerNorm([CFG.out_conv[-1], 32, 32])

    def forward(self, data):
        y_preds = {}
        
        # First apply utae model
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            y_pred = self.models[satellite](images, batch_positions=dates)
            y_preds[satellite] = y_pred
            
        satellite_keys = list(y_preds.keys())
        
        enhanced_features = {}

        # Perform cross-attention fusion for each satellite
        for i in range(len(satellite_keys)):
            fused_features = y_preds[satellite_keys[i]]  # Initialize with the first satellite's features
            query = fused_features
            for j in range(len(satellite_keys)):
                if j != i:
                    key = y_preds[satellite_keys[j]]
                    value = y_preds[satellite_keys[j]]
                    
                    attended_value = self.crossattention_modules[satellite_keys[i]](query, key, value)
                    fused_features = fused_features + attended_value
                    fused_features = self.ln(fused_features)
            enhanced_features[satellite_keys[i]] = fused_features

        # Multihead attention fusion
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            attention_module = self.attention_modules[satellite]
            attention = attention_module(enhanced_features[satellite])
            
            attentions[satellite] = attention

        fused_features = torch.stack(list(enhanced_features.values()), dim=0)
        attention_weights = torch.stack(list(attentions.values()), dim=0)
        attention_weights = F.softmax(attention_weights, dim=0)

        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        y_pred = self.conv_final(fused_features)
        return y_pred

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_features, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = num_features // num_heads
        
        # Ensure the division is valid
        assert self.dim_per_head * num_heads == num_features, "num_features must be divisible by num_heads"

        self.query_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.key_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.value_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Xavier Initialisation for all weights
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()

        # Apply the convolutions to the input
        query = self.query_conv(query)
        key = self.key_conv(key)
        value = self.value_conv(value)
        
        # Split the features into multiple heads
        query = self.split_heads(query, self.num_heads)
        key = self.split_heads(key, self.num_heads)
        value = self.split_heads(value, self.num_heads)

        # Compute the dot product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.dim_per_head ** 0.5
        attention_scores = self.softmax(attention_scores)

        # Apply attention to the value vector
        attended_value = torch.matmul(attention_scores, value)

        # Concatenate the heads together
        attended_value = self.combine_heads(attended_value)
        
        # Reshape to the original size
        attended_value = attended_value.view(batch_size, channels, height, width)
        
        return attended_value

    def split_heads(self, x, num_heads):
        batch_size, channels, height, width = x.size()
        new_shape = batch_size, num_heads, self.dim_per_head, height * width
        x = x.view(*new_shape).permute(0, 1, 3, 2)  # (batch_size, num_heads, seq_len, depth)
        return x

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, depth = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = batch_size, -1, seq_len * depth  # flatten the last two dimensions
        return x.view(*new_shape)
    
    
    
###############################################################################################################################################################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class EfficientChannelAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1(x)
        attention = self.sigmoid(self.conv2(x))
        a1, a2, a3 = torch.chunk(attention, 3, dim=1)
        return a1 * x1 + a2 * x2 + a3 * x3

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class EnhancedFusionModel2(Build_model):
    def __init__(self, CFG):
        super(EnhancedFusionModel2, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(satellite)
            self.models[satellite] = model

        self.eca = EfficientChannelAttention(CFG.out_conv[-1])
        self.multi_head_attention = MultiHeadAttention(CFG.out_conv[-1], num_heads=4)
        self.caf = ChannelAttentionFusion(CFG.out_conv[-1])

        self.conv1 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(CFG.out_conv[-1])
        self.conv2 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(CFG.out_conv[-1])
        self.activation = Mish()
        self.conv_final = nn.Conv2d(CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            images, dates = data[satellite]
            model = self.models[satellite]
            y_preds[satellite] = model(images, batch_positions=dates)

        l8_features = y_preds['L8']
        s1_features = y_preds['S1']
        s2_features = y_preds['S2']

        l8_features = self.eca(l8_features)
        s1_features = self.eca(s1_features)
        s2_features = self.eca(s2_features)

        fused_features = self.caf(l8_features, s1_features, s2_features)
        fused_features = self.multi_head_attention(fused_features)

        fused_features = self.activation(self.bn1(self.conv1(fused_features)))
        fused_features = self.activation(self.bn2(self.conv2(fused_features)))

        y_pred = self.conv_final(fused_features)
        return y_pred
    
    
####


import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialChannelGate(nn.Module):
    def __init__(self, in_channels, num_gates):
        super(SpatialChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_gates, kernel_size=1)
        self.conv2 = nn.Conv2d(num_gates, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sigmoid(self.conv2(self.conv1(x)))
        return x * gates

class ChannelAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.conv1(x)
        attention = self.sigmoid(self.conv2(x))
        a1, a2, a3 = torch.chunk(attention, 3, dim=1)
        return a1 * x1 + a2 * x2 + a3 * x3

class EnhancedFusionModel3(Build_model):
    def __init__(self, CFG):
        super(EnhancedFusionModel3, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        for satellite in CFG.satellites.keys():
            model = self.get_model(satellite)
            self.models[satellite] = model

        self.spatial_channel_gate = SpatialChannelGate(CFG.out_conv[-1], num_gates=32)
        self.channel_attention_fusion = ChannelAttentionFusion(CFG.out_conv[-1])

        self.conv1 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(CFG.out_conv[-1])
        self.conv2 = nn.Conv2d(CFG.out_conv[-1], CFG.out_conv[-1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(CFG.out_conv[-1])
        self.activation = nn.ReLU(inplace=True)
        self.conv_final = nn.Conv2d(CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        y_preds = {}
        for satellite in self.CFG.satellites.keys():
            images, dates = data[satellite]
            model = self.models[satellite]
            y_preds[satellite] = model(images, batch_positions=dates)

        l8_features = y_preds['L8']
        s1_features = y_preds['S1']
        s2_features = y_preds['S2']

        l8_features = self.spatial_channel_gate(l8_features)
        s1_features = self.spatial_channel_gate(s1_features)
        s2_features = self.spatial_channel_gate(s2_features)

        fused_features = self.channel_attention_fusion(l8_features, s1_features, s2_features)

        fused_features = self.activation(self.bn1(self.conv1(fused_features)))
        fused_features = self.activation(self.bn2(self.conv2(fused_features)))

        y_pred = self.conv_final(fused_features)
        return y_pred

    def get_model(self, satellite):
        config = self.CFG
        input_dim = len(config.satellites[satellite]["bands"])
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
        else:
            raise NotImplementedError
        return model
    
##########################################################################################
# CROSS_ALL_NORM_DROP (Final 2)
# scheduler visualise
# normalisation
# sattelite -> batch norm -> cross attention
# val acc
# trained weights
# test set
class CrossAttentionFusion(Build_model):
    def __init__(self, CFG):
        super(CrossAttentionFusion, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        self.crossattention_modules = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        
        # Initialize each satellite's model and multi-head self-attention module
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
            cross_attention = MultiHeadCrossAttention(CFG.out_conv[-1], num_heads=4)
            self.crossattention_modules[satellite] = cross_attention
            self_attention = MultiHeadPixelAttention(CFG.out_conv[-1], num_heads=4)
            self.attention_modules[satellite] = self_attention
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)
        
        self.ln = nn.LayerNorm([CFG.out_conv[-1], 32, 32])
        
        # Add batch normalization and dropout layers
        # self.bn = nn.BatchNorm2d(CFG.out_conv[-1])
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, data):
        y_preds = {}
        
        # First apply utae model
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            y_pred = self.models[satellite](images, batch_positions=dates)
            y_preds[satellite] = y_pred
            
        satellite_keys = list(y_preds.keys())
        
        enhanced_features = {}

        # Perform cross-attention fusion for each satellite
        for i in range(len(satellite_keys)):
            fused_features = y_preds[satellite_keys[i]]  # Initialize with the first satellite's features
            query = fused_features
            for j in range(len(satellite_keys)):
                if j != i:
                    key = y_preds[satellite_keys[j]]
                    value = y_preds[satellite_keys[j]]
                    
                    attended_value = self.crossattention_modules[satellite_keys[i]](query, key, value)
                    # attended_value = self.bn(attended_value)
                    attended_value = self.dropout(attended_value)
                    fused_features = fused_features + attended_value
                    fused_features = self.ln(fused_features)
            enhanced_features[satellite_keys[i]] = fused_features

        # Multihead attention fusion
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            attention_module = self.attention_modules[satellite]
            attention = attention_module(enhanced_features[satellite])
            
            attentions[satellite] = attention

        fused_features = torch.stack(list(enhanced_features.values()), dim=0)
        attention_weights = torch.stack(list(attentions.values()), dim=0)
        attention_weights = F.softmax(attention_weights, dim=0)

        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        # Decision fusion - average all the enhanced features
        # fused_features = torch.stack(list(enhanced_features.values()), dim=0)
        # fused_features = fused_features.mean(dim=0)
        
        y_pred = self.conv_final(fused_features)
        return y_pred, attention_weights

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_features, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = num_features // num_heads
        
        # Ensure the division is valid
        assert self.dim_per_head * num_heads == num_features, "num_features must be divisible by num_heads"

        self.query_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.key_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.value_conv = nn.Conv2d(num_features, num_features, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # Xavier Initialisation for all weights
        nn.init.xavier_uniform_(self.query_conv.weight)
        nn.init.xavier_uniform_(self.key_conv.weight)
        nn.init.xavier_uniform_(self.value_conv.weight)

    def forward(self, query, key, value):
        batch_size, channels, height, width = query.size()

        # Apply the convolutions to the input
        query = self.query_conv(query)
        key = self.key_conv(key)
        value = self.value_conv(value)
        
        # Split the features into multiple heads
        query = self.split_heads(query, self.num_heads)
        key = self.split_heads(key, self.num_heads)
        value = self.split_heads(value, self.num_heads)

        # Compute the dot product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.dim_per_head ** 0.5
        attention_scores = self.softmax(attention_scores)

        # Apply attention to the value vector
        attended_value = torch.matmul(attention_scores, value)

        # Concatenate the heads together
        attended_value = self.combine_heads(attended_value)
        
        # Reshape to the original size
        attended_value = attended_value.view(batch_size, channels, height, width)
        
        return attended_value

    def split_heads(self, x, num_heads):
        batch_size, channels, height, width = x.size()
        new_shape = batch_size, num_heads, self.dim_per_head, height * width
        x = x.view(*new_shape).permute(0, 1, 3, 2)  # (batch_size, num_heads, seq_len, depth)
        return x

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, depth = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = batch_size, -1, seq_len * depth  # flatten the last two dimensions
        return x.view(*new_shape)
    
    
#######
# New Model with Batch Norm

class CrossAttentionFusion3(Build_model):
    def __init__(self, CFG):
        super(CrossAttentionFusion3, self).__init__(CFG)
        self.CFG = CFG
        self.models = nn.ModuleDict()
        self.crossattention_modules = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        
        # Initialize each satellite's model and multi-head self-attention module
        for satellite in CFG.satellites.keys():
            model = self.get_model(sat=satellite)
            self.models[satellite] = model
            cross_attention = MultiHeadCrossAttention(CFG.out_conv[-1], num_heads=4)
            self.crossattention_modules[satellite] = cross_attention
            self_attention = MultiHeadPixelAttention(CFG.out_conv[-1], num_heads=4)
            self.attention_modules[satellite] = self_attention
        
        self.conv_final = nn.Conv2d(self.CFG.out_conv[-1], CFG.num_classes, kernel_size=3, stride=1, padding=1)
        
        # Replace LayerNorm with BatchNorm2d
        self.bn = nn.BatchNorm2d(CFG.out_conv[-1])
        
        # Keep the dropout layer as is
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, data):
        y_preds = {}
        
        # First apply utae model
        for satellite in self.CFG.satellites.keys():
            (images, dates) = data[satellite]
            y_pred = self.models[satellite](images, batch_positions=dates)
            y_preds[satellite] = y_pred
            
        satellite_keys = list(y_preds.keys())
        
        enhanced_features = {}

        # Perform cross-attention fusion for each satellite
        for i in range(len(satellite_keys)):
            fused_features = y_preds[satellite_keys[i]]  # Initialize with the first satellite's features
            query = fused_features
            for j in range(len(satellite_keys)):
                if j != i:
                    key = y_preds[satellite_keys[j]]
                    value = y_preds[satellite_keys[j]]
                    
                    attended_value = self.crossattention_modules[satellite_keys[i]](query, key, value)
                    attended_value = self.bn(attended_value)  # Apply BatchNorm2d instead of LayerNorm
                    attended_value = self.dropout(attended_value)
                    fused_features = fused_features + attended_value
            enhanced_features[satellite_keys[i]] = fused_features

        # Multihead attention fusion
        attentions = {}
        for satellite in self.CFG.satellites.keys():
            attention_module = self.attention_modules[satellite]
            attention = attention_module(enhanced_features[satellite])
            
            attentions[satellite] = attention

        fused_features = torch.stack(list(enhanced_features.values()), dim=0)
        attention_weights = torch.stack(list(attentions.values()), dim=0)
        attention_weights = F.softmax(attention_weights, dim=0)
        attention_weights_return = torch.stack(list(attentions.values()), dim=1)
        attention_weights_return = F.softmax(attention_weights_return, dim=1)

        fused_features = (fused_features * attention_weights).sum(dim=0)
        
        y_pred = self.conv_final(fused_features)
        return y_pred, attention_weights_return