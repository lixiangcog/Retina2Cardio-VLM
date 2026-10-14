import torch
import torch.nn as nn
from types import SimpleNamespace
from torchvision import transforms
from .retfound import RETFound_mae
from .checkpoint_guard import load_patch_encoder_weights

class CustomImageProcessor:
    def __init__(self, transforms_callable, image_mean):
        self.transforms = transforms_callable
        self.image_mean = image_mean

    def preprocess(self, images, return_tensors=None):
        # 模仿Hugging Face processor的行为，确保能处理单个或多个图像
        if not isinstance(images, list):
            images = [images]

        processed_images = [self.transforms(image) for image in images]

        # 模仿Hugging Face processor的输出格式，返回一个字典
        if return_tensors == 'pt':
            return {"pixel_values": torch.stack(processed_images)}
        else:
            return {"pixel_values": processed_images}

class OCT_2DVisionTower(nn.Module):
    def __init__(self, oct_2d_tower_path, args, delay_load=False):
        super().__init__()

        self.oct_2d_tower_weight = oct_2d_tower_path
        self.is_loaded = False

        self.custom_config = SimpleNamespace(
            image_size=224,
            patch_size=16,
            hidden_size=1024
        )

        IMG_MEAN = (0.485, 0.456, 0.406)
        IMG_STD = (0.229, 0.224, 0.225)

        _transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        self.image_processor = CustomImageProcessor(transforms_callable=_transforms, image_mean=IMG_MEAN)

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.oct_2d_tower_weight} is already loaded, skipping.')
            return

        self.oct_2d_tower = RETFound_mae(global_pool=True)

        if self.oct_2d_tower_weight:
            print(f"正在从 {self.oct_2d_tower_weight} 加载 RETFound 权重")
            checkpoint = torch.load(self.oct_2d_tower_weight, map_location='cpu', weights_only=False)
            # print(checkpoint.keys() if checkpoint.keys() is not None else "Checkpoint has no keys.")

        # # 安全地获取 'model' 键，或者直接使用整个检查点字典
        # state_dict = checkpoint.get('model2d', checkpoint)

        # # --- **核心修正代码** --- 3d-2d
        # # 创建一个新的字典，并从每个键中移除 'vit.' 这个前缀
        # new_state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items()}
        # # --- 修正结束 --- 3d-2d对齐
        # 2. 创建一个新的字典，并从每个键中移除 'backbone.' 这个前缀
        # new_state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items()}

        print("正在加载模型权重...")
        # 加载修正后的权重字典
        loading_report = load_patch_encoder_weights(self.oct_2d_tower, checkpoint)

        print("权重加载完成。加载报告如下：")
        print(loading_report)

        self.oct_2d_tower.eval()
        self.oct_2d_tower.requires_grad_(False)
        self.is_loaded = True
        print("RETFound-OCT 模型加载完毕。")

    @torch.no_grad()
    def forward(self, images):
        image_features = self.oct_2d_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
        return image_features

    @property
    def config(self):
        return self.custom_config

    @property
    def dtype(self):
        return next(self.oct_2d_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.oct_2d_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class CFP_VisionTower(nn.Module):
    def __init__(self, cfp_tower, args, delay_load=False):
        super().__init__()

        self.cfp_tower_weight = cfp_tower
        self.is_loaded = False
        self.custom_config = SimpleNamespace(
            image_size=224,
            patch_size=16,   # RETFound_mae patch size
            hidden_size=1024
        )

        IMG_MEAN = (0.485, 0.456, 0.406)
        IMG_STD = (0.229, 0.224, 0.225)

        _transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])
        self.image_processor = CustomImageProcessor(
            transforms_callable=_transforms,
            image_mean=IMG_MEAN
        )

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.cfp_tower_weight} is already loaded, skipping.")
            return

        # 与 OCT 分支一致：直接用 RETFound_mae 编码器
        self.cfp_tower = RETFound_mae(global_pool=True)

        print(f"正在从 {self.cfp_tower_weight} 加载 RETFound-CFP 权重")
        checkpoint = torch.load(self.cfp_tower_weight, map_location="cpu", weights_only=False)

        # 兼容两种格式：
        # 1) 训练整包: {"model": ...}
        # 2) 纯 state_dict: {...}
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        print("正在加载 CFP 模型权重...")
        loading_report = load_patch_encoder_weights(self.cfp_tower, state_dict)
        print("CFP 权重加载报告：")
        print(loading_report)

        self.cfp_tower.eval()
        self.cfp_tower.requires_grad_(False)
        self.is_loaded = True
        print("RETFound-CFP 模型加载完毕。")

    @torch.no_grad()
    def forward(self, images):
        # 与 OCT 保持一致，返回 patch/token features
        image_features = self.cfp_tower.forward_features(
            images.to(device=self.device, dtype=self.dtype)
        )
        return image_features

    @property
    def config(self):
        return self.custom_config

    @property
    def dtype(self):
        return next(self.cfp_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.cfp_tower.parameters()).device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
