import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import model_ViT
import os
from util.pos_embed import interpolate_pos_embed
from sklearn.decomposition import PCA
import baseline_models as bm
import numpy as np

def main():

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    model_name = 'vit_tiny_patch16_nodp'
    depth = 9
    model_ckpt_1 = './ckpt_simsiamViT_tiny_new_local1/server/fed_checkpoint_final.pth'  # your first pretrained model
    model_ckpt_2 = './ckpt_MAEViT_tiny_local1/server/fed_checkpoint_final.pth'  # your second pretrained model
    model_ckpt_3 = './ckpt_MAEViT_MAR_tiny_local1/server/fed_checkpoint_final.pth'  # your second pretrained model
    model_ckpt_4 = './ckpt_simsiamViT_tiny_local99/server/fed_checkpoint_final.pth'  # your first pretrained model
    model_ckpt_5 = './ckpt_MAEViT_tiny_local99/server/fed_checkpoint_final.pth'  # your second pretrained model
    model_ckpt_6 = './ckpt_MAEViT_MAR_tiny_local99/server/fed_checkpoint_final.pth'  # your second pretrained model
    model_ckpt_7 = './ckpt_simsiamViT_tiny_new/server/fed_checkpoint_final.pth'  # your first pretrained model
    model_ckpt_8 = './ckpt_MAEViT_tiny/server/fed_checkpoint_final.pth'  # your second pretrained model
    model_ckpt_9 = './ckpt_MAEViT_MAR_tiny/server/fed_checkpoint_final.pth'  # your second pretrained model
    ckpts = [model_ckpt_1, model_ckpt_2, model_ckpt_3, model_ckpt_4, model_ckpt_5, model_ckpt_6, model_ckpt_7, model_ckpt_8, model_ckpt_9]

    dataset_name = 'cifar10'
    num_classes = 10

    # -------- DATASET --------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def model_load_ckpt(model, load_path):
        if load_path and os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % load_path)
            checkpoint_model = checkpoint
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)

    # Step 1: Load your pretrained models
    models = []
    for i in range(9):
        # if i == 0:
        #     model = bm.create_backbone(name='res%s-origin' % 18, num_classes=num_classes)
        #     checkpoint = torch.load(ckpts[i], map_location='cpu')
        #     print("Load pre-trained checkpoint from: %s" % ckpts[i])
        #     checkpoint_model = checkpoint
        #     if 'model' in checkpoint:
        #         checkpoint_model = checkpoint['model']
        #     model.load_state_dict(checkpoint_model, strict=False)
        # else:
        model = model_ViT.__dict__[model_name](
            depth=depth,
            num_classes=num_classes,
            global_pool=True,
        )
        model_load_ckpt(model, ckpts[i])
        model.head = torch.nn.Identity()  # remove head
        # model.eval().to(device)
        model.to(device)
        models.append(model)

    simsiamViT_local1_vs_global = l2_weight_distance(models[0], models[6])
    simsiamViT_local1_vs_local100 = l2_weight_distance(models[0], models[3])
    simsiamViT_local100_vs_global = l2_weight_distance(models[3], models[6])
    MAEViT_local1_vs_global = l2_weight_distance(models[1], models[7])
    MAEViT_local1_vs_local100 = l2_weight_distance(models[1], models[4])
    MAEViT_local100_vs_global = l2_weight_distance(models[4], models[7])
    MARViT_local1_vs_global = l2_weight_distance(models[2], models[8])
    MARViT_local1_vs_local100 = l2_weight_distance(models[2], models[5])
    MARViT_local100_vs_global = l2_weight_distance(models[5], models[8])
    print(f"simsiamViT L2 Distance (Local Model 1 vs Global): {simsiamViT_local1_vs_global:.2f}")
    print(f"simsiamViT L2 Distance (Local Model 1 vs Local Model 100): {simsiamViT_local1_vs_local100:.2f}")
    print(f"simsiamViT L2 Distance (Local Model 100 vs Global): {simsiamViT_local100_vs_global:.2f}")
    print(f"MAEViT L2 Distance (Local Model 1 vs Global): {MAEViT_local1_vs_global:.2f}")
    print(f"MAEViT L2 Distance (Local Model 1 vs Local Model 100): {MAEViT_local1_vs_local100:.2f}")
    print(f"MAEViT L2 Distance (Local Model 100 vs Global): {MAEViT_local100_vs_global:.2f}")
    print(f"MARViT L2 Distance (Local Model 1 vs Global): {MARViT_local1_vs_global:.2f}")
    print(f"MARViT L2 Distance (Local Model 1 vs Local Model 100): {MARViT_local1_vs_local100:.2f}")
    print(f"MARViT L2 Distance (Local Model 100 vs Global): {MARViT_local100_vs_global:.2f}")

    raise TypeError


   # -------- EXTRACT FEATURES --------
    def extract_features(model):
        features, labels = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(dataloader):
                imgs = imgs.to(device)
                feats = model(imgs)
                features.append(feats.cpu())
                labels.append(lbls)
        return torch.cat(features), torch.cat(labels)

    features1, labels = extract_features(models[0])
    features2, _ = extract_features(models[1])
    features3, _ = extract_features(models[2])
    features4, _ = extract_features(models[3])
    features5, _ = extract_features(models[4])
    features6, _ = extract_features(models[5])
    features7, _ = extract_features(models[6])
    features8, _ = extract_features(models[7])
    features9, _ = extract_features(models[8])

    # # Updated function with PCA + UMAP + subsampling + class centroids

    # # Extract features
    # features_list = []
    # labels = None
    # for model in models:
    #     feats, lbls = extract_features(model)
    #     features_list.append(feats)
    #     if labels is None:
    #         labels = lbls

    # labels_np = labels.numpy()
    # subsample_size = 5000
    # indices = np.random.choice(len(labels_np), subsample_size, replace=False)
    # labels_sub = labels_np[indices]

    # # Set up plot
    # fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    # if len(models) == 1:
    #     axes = [axes]

    # for i, features in enumerate(features_list):
    #     features_np = features.numpy()
    #     features_sub = features_np[indices]

    #     # PCA
    #     n_components = min(50, features_sub.shape[0], features_sub.shape[1])
    #     pca = PCA(n_components=n_components)
    #     features_pca = pca.fit_transform(features_sub)

    #     # UMAP
    #     umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    #     proj = umap_model.fit_transform(features_pca)

    #     # Plot
    #     ax = axes[i]
    #     sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels_sub, palette="tab10", s=10, ax=ax, legend=False)
    #     ax.set_title(f"UMAP: Model {i+1} Features")
    #     ax.set_aspect('equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # plt.tight_layout()
    # plt.savefig("umap_comparison.png", dpi=300)

    pca = PCA(n_components=20)
    features1 = pca.fit_transform(features1)
    features2 = pca.fit_transform(features2)
    features3 = pca.fit_transform(features3)
    features4 = pca.fit_transform(features4)
    features5 = pca.fit_transform(features5)
    features6 = pca.fit_transform(features6)    
    features7 = pca.fit_transform(features7)
    features8 = pca.fit_transform(features8)
    features9 = pca.fit_transform(features9)

    # -------- UMAP --------
    # umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    umap_model = umap.UMAP(n_components=2, random_state=seed)
    proj1 = umap_model.fit_transform(features1)
    proj2 = umap_model.fit_transform(features2)
    proj3 = umap_model.fit_transform(features3)
    proj4 = umap_model.fit_transform(features4)
    proj5 = umap_model.fit_transform(features5)
    proj6 = umap_model.fit_transform(features6)
    proj7 = umap_model.fit_transform(features7)
    proj8 = umap_model.fit_transform(features8)
    proj9 = umap_model.fit_transform(features9)

    # -------- PLOT --------
    plt.figure(figsize=(15, 15))
    title_size = 18
    legend_size = 12

    label_names = [f"class {i}" for i in labels]

    plt.subplot(3, 3, 1)
    sns.scatterplot(x=proj1[:, 0], y=proj1[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("Simsiam+ViT Client 1 Local Model", fontsize=title_size)

    plt.subplot(3, 3, 2)
    sns.scatterplot(x=proj2[:, 0], y=proj2[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("MAE+ViT Client 1 Local Model", fontsize=title_size)

    plt.subplot(3, 3, 3)
    sns.scatterplot(x=proj3[:, 0], y=proj3[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("MAR+ViT Client 1 Local Model", fontsize=title_size)

    plt.subplot(3, 3, 4)
    sns.scatterplot(x=proj1[:, 0], y=proj4[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("Simsiam+ViT Client 100 Local Model", fontsize=title_size)

    plt.subplot(3, 3, 5)
    sns.scatterplot(x=proj2[:, 0], y=proj5[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("MAE+ViT Client 100 Local Model", fontsize=title_size)

    plt.subplot(3, 3, 6)
    sns.scatterplot(x=proj3[:, 0], y=proj6[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("MAR+ViT Client 100 Local Model", fontsize=title_size)

    plt.subplot(3, 3, 7)
    sns.scatterplot(x=proj1[:, 0], y=proj7[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("Simsiam+ViT Global Model", fontsize=title_size)

    plt.subplot(3, 3, 8)
    sns.scatterplot(x=proj2[:, 0], y=proj8[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("MAE+ViT Global Model", fontsize=title_size)

    plt.subplot(3, 3, 9)
    sns.scatterplot(x=proj3[:, 0], y=proj9[:, 1], hue=label_names, palette="tab10", s=10)
    plt.legend(loc="lower left", fontsize=legend_size)
    plt.title("MAR+ViT Global Model", fontsize=title_size)

    
    plt.tight_layout()
    plt.savefig("umap_comparison.pdf", format='pdf', bbox_inches='tight', pad_inches=0)

def get_model_weight_vector(model):
    """Flatten and concatenate all parameters into a single vector."""
    return torch.cat([p.detach().cpu().view(-1) for p in model.parameters() if p.requires_grad])

def l2_weight_distance(model1, model2):
    """Compute the L2 norm between weights of two models."""
    w1 = get_model_weight_vector(model1)
    w2 = get_model_weight_vector(model2)
    return torch.norm(w1 - w2, p=2).item()

if __name__ == '__main__':
    main()