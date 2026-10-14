# Copyright 2023 Haotian Liu. Apache-2.0 derived code; see LICENSES/Apache-2.0.txt
from abc import ABC,abstractmethod
import torch
import torch.nn as nn
from .projectors import build_vision_projector,build_oct_2d_projector
from .encoders import CFP_VisionTower,OCT_2DVisionTower

def build_vision_tower(args,**kw):
    return CFP_VisionTower(getattr(args,"vision_tower",getattr(args,"mm_vision_tower",None)),args,**kw)

def build_oct_2d_tower(args,**kw):
    return OCT_2DVisionTower(getattr(args,"oct_2d_tower",getattr(args,"mm_oct_2d_tower",None)),args,**kw)

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

        if hasattr(config, "mm_oct_2d_tower"):
            self.oct_2d_tower = build_oct_2d_tower(config, delay_load=True)
            self.oct_2d_projector = build_oct_2d_projector(config)

    def get_oct_2d_tower(self):
        oct_2d_tower = getattr(self, 'oct_2d_tower', None)
        if type(oct_2d_tower) is list:
            oct_2d_tower = oct_2d_tower[0]
        return oct_2d_tower

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_oct_2d_modules(self, model_args, fsdp=None):
        oct_2d_tower = model_args.oct_2d_tower
        pretrain_oct_2d_projector = model_args.pretrain_oct_2d_projector
        self.config.mm_oct_2d_tower = oct_2d_tower

        if self.get_oct_2d_tower() is None:
            print(f"Building 2D OCT tower...")
            oct_2d_tower = build_oct_2d_tower(model_args)
            # print(oct_2d_tower.is_loaded)

            if fsdp is not None and len(fsdp) > 0:
                self.oct_2d_tower = [oct_2d_tower]
            else:
                self.oct_2d_tower = oct_2d_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                oct_2d_tower = self.oct_2d_tower[0]
            else:
                oct_2d_tower = self.oct_2d_tower
            oct_2d_tower.load_model()

        self.config.use_mm_proj = True
        self.config.oct_2d_projector_type = getattr(model_args, 'oct_2d_projector_type', 'linear')

        if getattr(self, 'oct_2d_projector', None) is None:
            print("Building 2D-OCT projector...")
            self.oct_2d_projector = build_oct_2d_projector(self.config)

        else:
            # In case it is frozen by LoRA
            for p in self.oct_2d_projector.parameters():
                p.requires_grad = True

        if pretrain_oct_2d_projector is not None:
            oct_2d_projector_weights = torch.load(pretrain_oct_2d_projector, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.oct_2d_projector.load_state_dict(get_w(oct_2d_projector_weights, 'oct_2d_projector'))


    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            print(f"Building Vision tower...")
            vision_tower = build_vision_tower(model_args)
            print(vision_tower.is_loaded)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()
        # print(vision_tower.is_loaded)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            print("Building mm projector...")
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_oct_2d_tower(self):
        return self.get_model().get_oct_2d_tower()

    def encode_images(self, images):
        # print(images.device)
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_oct_2d_images(self, images):
        # print(images.device)
        oct_2d_features = self.get_model().get_oct_2d_tower()(images)
        oct_2d_features = self.get_model().oct_2d_projector(oct_2d_features)
        oct_2d_features = self.get_model().mm_projector(oct_2d_features)
        return oct_2d_features
