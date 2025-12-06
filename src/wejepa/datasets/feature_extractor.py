import torch
from torchvision import transforms

class FeatureExtractor:
    """Feature extractor using WE-JEPA model."""
    def __init__(self, checkpoint_path, device='cuda', resolution=96, backbone='vit_b_16'):
        from wejepa import IJEPA_base, default_config, adapt_config_for_backbone
        self.device = device
        self.resolution = resolution
        cfg = default_config()
        cfg = adapt_config_for_backbone(cfg, backbone)
        cfg.model.img_size = resolution
        cfg.data.image_size = resolution
        self.model = IJEPA_base(
            img_size=cfg.model.img_size,
            patch_size=cfg.model.patch_size,
            in_chans=3,
            embed_dim=cfg.model.embed_dim,
            enc_depth=cfg.model.enc_depth,
            pred_depth=cfg.model.pred_depth,
            num_heads=cfg.model.num_heads,
            post_emb_norm=cfg.model.post_emb_norm,
            M=cfg.model.M,
            mode='test',
            backbone=cfg.model.backbone,
            pretrained=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
        self.model.eval()
        self.model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model(x)
            pooled = feats.mean(dim=1)
        return pooled.cpu().numpy()[0]

    def extract_batch_features(self, images):
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            pooled = feats.mean(dim=1)
        return pooled.cpu().numpy()
