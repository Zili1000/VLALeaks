import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

# 1. Load data

features0 = torch.load('/home/tcs/4t01/lxk/openvla/log/member/libero_spatial_attention_features.pt')[:10000]
features1 = torch.load('/home/tcs/4t01/lxk/openvla/log/nonmember/libero_spatial_attention_features.pt')[:10000]

# features0 = torch.load('/home/tcs/4t01/lxk/openvla/log/member/libero_object_attention_features.pt')[:10000]
# features1 = torch.load('/home/tcs/4t01/lxk/openvla/log/nonmember/libero_object_attention_features.pt')[:10000]

# features0 = torch.load('/home/tcs/4t01/lxk/openvla/log/member/libero_goal_attention_features.pt')[:10000]
# features1 = torch.load('/home/tcs/4t01/lxk/openvla/log/nonmember/libero_goal_attention_features.pt')[:10000]

# features0 = torch.load('/home/tcs/4t01/lxk/openvla/log/member/libero_10_attention_features.pt')[:10000]
# features1 = torch.load('/home/tcs/4t01/lxk/openvla/log/nonmember/libero_10_attention_features.pt')[:10000]


# num = 500
# Convert data type
# if isinstance(features0[0], torch.Tensor) and features0[0].dtype == torch.bfloat16:
#     features0 = [f.float() for f in features0]
#     features1 = [f.float() for f in features1]

# Merge
X = torch.cat([torch.stack(features0), torch.stack(features1)], dim=0)
y = torch.cat([torch.zeros(len(features0)), torch.ones(len(features1))], dim=0)

# Convert to numpy
X_np = X.numpy()
y_np = y.numpy()

print(f"Data shape: {X_np.shape}")
print(f"Feature statistics: mean={X_np.mean():.4f}, std={X_np.std():.4f}")
print(f"Feature range: min={X_np.min():.4f}, max={X_np.max():.4f}")

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.9, stratify=y_np, random_state=123
)

# ==================== Utility function: Calculate TPR@1%FPR ====================
def calculate_tpr_at_fpr(y_true, y_pred_proba, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    # Find the index closest to the target FPR
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx], fpr[idx]

# ==================== Random Forest Evaluation ====================
print("\n" + "="*50)
print("Random Forest Model")
print("="*50)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]  # Get probability for positive class

# Calculate metrics
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_tpr_at_1fpr, rf_fpr_at_thresh = calculate_tpr_at_fpr(y_test, rf_proba, target_fpr=0.01)

print(f"Accuracy: {rf_acc:.4f}")
print(f"AUC: {rf_auc:.4f}")
print(f"TPR@1%FPR: {rf_tpr_at_1fpr:.4f} (actual FPR={rf_fpr_at_thresh:.4f})")
print(classification_report(y_test, rf_pred))

# Plot ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# 3. Feature importance
feature_importance = rf.feature_importances_
top_k = 20
top_indices = np.argsort(feature_importance)[-top_k:]

print(f"\nTop {top_k} important features:")
for idx in top_indices[::-1]:
    print(f"  Feature {idx}: {feature_importance[idx]:.6f}")

# ==================== MLP Model Evaluation ====================
class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[32,16]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

# Prepare data
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_t, y_train_t), 
    batch_size=128, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test_t, y_test_t), 
    batch_size=128
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImprovedMLP(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

print("\n" + "="*50)
print("Improved MLP Model")
print("="*50)

best_acc = 0
best_auc = 0
best_tpr_at_1fpr = 0
train_losses = []
val_accs = []
val_aucs = []
val_tprs = []

for epoch in range(500):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    tpr_at_1fpr, _ = calculate_tpr_at_fpr(all_labels, all_probs, target_fpr=0.01)
    
    train_losses.append(train_loss/len(train_loader))
    val_accs.append(acc)
    val_aucs.append(auc_score)
    val_tprs.append(tpr_at_1fpr)
    
    scheduler.step(1 - acc)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/500, Loss: {train_loss/len(train_loader):.4f}, Acc: {acc:.4f}, AUC: {auc_score:.4f}, TPR@1%FPR: {tpr_at_1fpr:.4f}')
    
    if acc > best_acc:
        best_acc = acc
        best_auc = auc_score
        best_tpr_at_1fpr = tpr_at_1fpr
        torch.save(model.state_dict(), 'best_model.pth')

print(f"\nBest test accuracy: {best_acc:.4f}")
print(f"Best test AUC: {best_auc:.4f}")
print(f"Best test TPR@1%FPR: {best_tpr_at_1fpr:.4f}")

# ==================== Final Evaluation and Visualization ====================
print("\n" + "="*50)
print("Final Evaluation")
print("="*50)

# Get final model predictions
model.eval()
all_probs_mlp = []
all_labels_mlp = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        output = model(batch_x)
        probs = torch.softmax(output, dim=1)
        all_probs_mlp.extend(probs[:, 1].cpu().numpy())
        all_labels_mlp.extend(batch_y.numpy())

final_acc = accuracy_score(all_labels_mlp, [1 if p > 0.5 else 0 for p in all_probs_mlp])
final_auc = roc_auc_score(all_labels_mlp, all_probs_mlp)
final_tpr_at_1fpr, _ = calculate_tpr_at_fpr(all_labels_mlp, all_probs_mlp, target_fpr=0.01)

print(f"VLALeaks-MLP - Accuracy: {final_acc:.4f}, AUC: {final_auc:.4f}, TPR@1%FPR: {final_tpr_at_1fpr:.4f}")
print(f"VLALeaks-RF - Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}, TPR@1%FPR: {rf_tpr_at_1fpr:.4f}")

# ==================== Visualization ====================
plt.figure(figsize=(15, 5))

# 1. PCA visualization
plt.subplot(1, 3, 1)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_np)
plt.scatter(X_pca[y_np==0, 0], X_pca[y_np==0, 1], alpha=0.5, label='Member', s=1)
plt.scatter(X_pca[y_np==1, 0], X_pca[y_np==1, 1], alpha=0.5, label='Non-Member', s=1)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA Visualization')
text = f'Explained variance: {pca.explained_variance_ratio_.sum():.3f}'
plt.text(0.05, 0.05, text, transform=plt.gca().transAxes, 
         verticalalignment='bottom', horizontalalignment='left',
         fontsize=8, color='gray')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. t-SNE visualization (if not too many samples)
plt.subplot(1, 3, 2)
print("\nComputing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_np)
plt.scatter(X_tsne[y_np==0, 0], X_tsne[y_np==0, 1], alpha=0.5, label='Member', s=1)
plt.scatter(X_tsne[y_np==1, 0], X_tsne[y_np==1, 1], alpha=0.5, label='Non-Member', s=1)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE Visualization')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. ROC curve comparison
plt.subplot(1, 3, 3)
# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.plot(fpr_rf, tpr_rf, label=f'VLALeaks-RF (AUC = {roc_auc_rf:.4f})', linewidth=2)

# MLP ROC
fpr_mlp, tpr_mlp, _ = roc_curve(all_labels_mlp, all_probs_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
plt.plot(fpr_mlp, tpr_mlp, label=f'VLALeaks-MLP (AUC = {roc_auc_mlp:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_analysis_with_auc.png', dpi=150)
plt.show()

# ==================== Print Detailed Results ====================
print("\n" + "="*50)
print("Detailed Results Summary")
print("="*50)
print(f"Random Forest:")
print(f"  - Accuracy: {rf_acc:.4f}")
print(f"  - AUC: {rf_auc:.4f}")
print(f"  - TPR@1%FPR: {rf_tpr_at_1fpr:.4f}")
print(f"\nMLP (Best):")
print(f"  - Accuracy: {best_acc:.4f}")
print(f"  - AUC: {best_auc:.4f}")
print(f"  - TPR@1%FPR: {best_tpr_at_1fpr:.4f}")
print(f"\nMLP (Final):")
print(f"  - Accuracy: {final_acc:.4f}")
print(f"  - AUC: {final_auc:.4f}")
print(f"  - TPR@1%FPR: {final_tpr_at_1fpr:.4f}")

# If AUC is close to 0.5, the model has no learning capability
if final_auc < 0.55:
    print("\n⚠️ Warning: AUC close to 0.5, the model has no effective classification capability!")
    print("Possible reasons:")
    print("  1. Extracted attention features have no discriminative power")
    print("  2. Feature distributions of the two classes are identical")
    print("  3. Bug in feature extraction process")
elif final_auc < 0.7:
    print("\n⚠️ Hint: AUC is low, model has limited classification capability")
else:
    print("\n✓ Good AUC, model has good classification capability")