#!/usr/bin/env python
import os
import glob
import argparse
import time
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

import matplotlib.pyplot as plt


class MixedFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [p for p in glob.glob(os.path.join(image_dir, "*")) if os.path.isfile(p)]
        self.transform = transform
        assert len(self.image_paths) > 1, "The directory must contain at least two images."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        neutral_idx = idx % len(self.image_paths)
        expr_idx = random.randint(0, len(self.image_paths)-1)
        while expr_idx == neutral_idx:
            expr_idx = random.randint(0, len(self.image_paths)-1)

        neutral_img = Image.open(self.image_paths[neutral_idx]).convert("RGB")
        expr_img = Image.open(self.image_paths[expr_idx]).convert("RGB")

        if self.transform:
            neutral_img = self.transform(neutral_img)
            expr_img = self.transform(expr_img)

        return neutral_img, expr_img


class ExpressionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ExpressionClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits


class ExpressionEncoder(nn.Module):
    def __init__(self, input_channels=3, expr_dim=128):
        super(ExpressionEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, expr_dim)

    def forward(self, x):
        feat = self.features(x)
        expr_code = self.fc(feat.view(feat.size(0), -1))
        return expr_code


class ExpressionTransferGenerator(nn.Module):
    def __init__(self, expr_dim=128):
        super(ExpressionTransferGenerator, self).__init__()

        def conv_block(in_c, out_c, kernel=4, stride=2, pad=1, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel, stride, pad)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.enc1 = conv_block(3, 64, norm=False)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 512)
        self.enc6 = conv_block(512, 512)
        self.enc7 = conv_block(512, 512, norm=False)

        self.expr_encoder = ExpressionEncoder(input_channels=3, expr_dim=expr_dim)

        self.fuse_conv = nn.Conv2d(512+expr_dim, 512, kernel_size=1)

        def deconv_block(in_c, out_c, kernel=4, stride=2, pad=1, norm=True, dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            if dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.dec6 = deconv_block(512, 512, dropout=True)
        self.dec5 = deconv_block(512+512, 512, dropout=True)
        self.dec4 = deconv_block(512+512, 256)
        self.dec3 = deconv_block(256+512, 128)
        self.dec2 = deconv_block(128+256, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64+128, 3, 4, 2, 1),
            nn.Tanh()
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, neutral_img, expr_img):
        e1 = self.enc1(neutral_img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        expr_code = self.expr_encoder(expr_img)
        B = expr_code.size(0)
        expr_code = expr_code.view(B, -1, 1, 1).expand(B, expr_code.size(1), e7.size(2), e7.size(3))

        fused = torch.cat([e7, expr_code], dim=1)
        fused = self.fuse_conv(fused)

        d6 = self.dec6(fused)
        d5 = self.dec5(torch.cat([d6, e6], dim=1))
        d4 = self.dec4(torch.cat([d5, e5], dim=1))
        d3 = self.dec3(torch.cat([d4, e4], dim=1))
        d2 = self.dec2(torch.cat([d3, e3], dim=1))
        out = self.dec1(torch.cat([d2, e2], dim=1))
        out = self.upsample(out)
        return out


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        def disc_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        self.model = nn.Sequential(
            disc_block(in_channels, 64, norm=False),
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img):
        return self.model(img)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    dataset = SingleFolderFaceDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    G = ExpressionTransferGenerator(expr_dim=args.expr_dim).to(device)
    D = PatchDiscriminator().to(device)
    expression_classifier = ExpressionClassifier(num_classes=2).to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_expr = optim.Adam(expression_classifier.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    adv_criterion = nn.MSELoss()
    fm_criterion = nn.L1Loss()
    expr_criterion = nn.CrossEntropyLoss()
    
    def real_labels(x):
        return torch.ones_like(x).to(device)
    def fake_labels(x):
        return torch.zeros_like(x).to(device)
    
    print("Starting training on device:", device)
    start_time = time.time()

    epoch_G_losses = []
    epoch_D_losses = []
    epoch_acc = []

    for epoch in range(args.num_epochs):
        G.train()
        D.train()
        expression_classifier.train()
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, (neutral_imgs, expr_imgs) in enumerate(progress_bar):
            neutral_imgs = neutral_imgs.to(device)
            expr_imgs = expr_imgs.to(device)
            B = neutral_imgs.size(0)

            opt_D.zero_grad()
            real_out = D(expr_imgs)
            d_real_loss = adv_criterion(real_out, real_labels(real_out))
            fake_imgs = G(neutral_imgs, expr_imgs)
            fake_out = D(fake_imgs.detach())
            d_fake_loss = adv_criterion(fake_out, fake_labels(fake_out))
            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            opt_D.step()

            opt_expr.zero_grad()
            logits_real = expression_classifier(expr_imgs)
            target_real = torch.ones(B, dtype=torch.long).to(device)
            expr_cls_loss = expr_criterion(logits_real, target_real)
            expr_cls_loss.backward()
            opt_expr.step()

            opt_G.zero_grad()
            fake_out = D(fake_imgs)
            adv_loss = adv_criterion(fake_out, real_labels(fake_out))
            
            def get_intermediate_feature(model, x, num_layers=3):
                for i, layer in enumerate(model.model):
                    x = layer(x)
                    if i == num_layers - 1:
                        break
                return x
            real_feat = get_intermediate_feature(D, expr_imgs, num_layers=3)
            fake_feat = get_intermediate_feature(D, fake_imgs, num_layers=3)
            fm_loss_val = fm_criterion(fake_feat, real_feat.detach())

            logits_fake = expression_classifier(fake_imgs)
            target_fake = torch.ones(B, dtype=torch.long).to(device)
            expr_loss = expr_criterion(logits_fake, target_fake)

            g_loss = adv_loss + args.lambda_expr * expr_loss + args.lambda_fm * fm_loss_val
            g_loss.backward()
            opt_G.step()

            epoch_G_loss += g_loss.item() * B
            epoch_D_loss += d_loss.item() * B

            with torch.no_grad():
                real_pred = (D(expr_imgs) > 0.5).float()
                fake_pred = (D(fake_imgs.detach()) < 0.5).float()
                batch_acc = (real_pred.sum() + fake_pred.sum()) / (real_pred.numel() + fake_pred.numel())

            correct += (batch_acc.item() * B)
            total += B

            progress_bar.set_postfix(G_loss=f"{g_loss.item():.4f}", D_loss=f"{d_loss.item():.4f}")

        avg_G_loss = epoch_G_loss / len(dataset)
        avg_D_loss = epoch_D_loss / len(dataset)
        avg_acc = correct / total
        epoch_G_losses.append(avg_G_loss)
        epoch_D_losses.append(avg_D_loss)
        epoch_acc.append(avg_acc)

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Avg G_loss: {avg_G_loss:.4f}, Avg D_loss: {avg_D_loss:.4f}")

    plt.figure()
    plt.plot(range(len(epoch_G_losses)), epoch_G_losses, label="Generator Loss")
    plt.plot(range(len(epoch_D_losses)), epoch_D_losses, label="Discriminator Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator & Discriminator Loss (Combined)")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "loss_plot_combined.png"))
    plt.close()

    plt.figure()
    plt.plot(range(len(epoch_G_losses)), epoch_G_losses, color='blue', label="Generator Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Loss Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "loss_plot_generator.png"))
    plt.close()

    plt.figure()
    plt.plot(range(len(epoch_D_losses)), epoch_D_losses, color='orange', label="Discriminator Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss Over Epochs")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "loss_plot_discriminator.png"))
    plt.close()

    best_acc = max(epoch_acc)
    print("Best Discriminator Accuracy:", best_acc)

    plt.figure()
    plt.plot(epoch_acc, label="Discriminator Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "accuracy_plot.png"))
    plt.close()

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.2f} minutes")
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(args.save_dir, "generator.pth"))
    print("Saved generator model.")

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    G = ExpressionTransferGenerator(expr_dim=args.expr_dim).to(device)
    checkpoint = torch.load(os.path.join(args.save_dir, "generator.pth"), map_location=device)
    G.load_state_dict(checkpoint)
    G.eval()

    neutral_img = Image.open(args.neutral_img).convert("RGB")
    expr_img = Image.open(args.expr_img).convert("RGB")
    neutral_tensor = transform(neutral_img).unsqueeze(0).to(device)
    expr_tensor = transform(expr_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = G(neutral_tensor, expr_tensor)
    out_filename = os.path.join(args.save_dir, "inference_output.png")
    save_image((output * 0.5 + 0.5), out_filename)
    print("Saved inference output to", out_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Facial Expression Transfer (Single Folder Version)")
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], default='train')
    parser.add_argument('--data_dir', type=str, default='data', help="Folder of mixed face images")
    parser.add_argument('--neutral_img', type=str, default='testNeutral.png')
    parser.add_argument('--expr_img', type=str, default='testEmotion.png')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda_expr', type=float, default=20.0)
    parser.add_argument('--lambda_fm', type=float, default=10.0)
    parser.add_argument('--expr_dim', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='saved_models')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        inference(args)

