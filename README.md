# t-SNE-Scratch-Implementation-on-Wine-Dataset
"ML models implemented from scratch using NumPy and Pandas only"

🧭 t-SNE (t-Distributed Stochastic Neighbor Embedding) — From Theory to Implementation

📊 Applied on: Wine Dataset (Unsupervised Learning)

This project demonstrates how t-SNE can be used to visualize high-dimensional data by projecting it into a lower-dimensional space (2D).
It includes both mathematical intuition and practical implementation using Scikit-learn.


---

🧠 1. What is t-SNE?

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique mainly used for data visualization.
It preserves the local structure of data — i.e., points that are close in high-dimensional space remain close in low-dimensional space.


---

### ⚙️ 2. Why t-SNE?

Problem	                                                   PCA	               t-SNE

Captures linear structure                                 	✅                 	✅
Captures non-linear structure	                              ❌                	✅
Preserves global variance	                                  ✅	                ❌
Preserves local similarity	                                ⚠️ Partial	          ✅ Strong
Use case	                                                 Feature reduction	   Cluster visualization
###

t-SNE is ideal for understanding clustering and visualizing embeddings in complex datasets like Wine, MNIST, and others.


---

🧩 3. Mathematical Intuition

Step 1 — Similarities in High-Dimensional Space

For each pair of points $$x_i$$, $$x_j$$ compute the probability that $$x_i$$  is a neighbor of $$x_j$$ :

$$p_{j|i}$$ = $$\frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

Symmetrize it:

$$p_{ij}$$ = $$\frac{p_{j|i} + p_{i|j}}{2n}$$


---

Step 2 — Similarities in Low-Dimensional Space

Map the data to low-dimensional space as $$y_i$$, $$y_j$$ and compute:

$$q_{ij}$$ = $$\frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

Here we use Student’s t-distribution to allow for better cluster separation.


---

Step 3 — Objective Function (KL Divergence)

The algorithm minimizes the Kullback-Leibler Divergence (KL Divergence) between the two distributions:

C = KL(P || Q) = $$\sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

This ensures that local relationships from high-dimensional space are preserved in the low-dimensional map.


---

### 🧮 4. Implementation Steps

🧰 Step 1 — Import Libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

###
---

### 🧪 Step 2 — Load and Preprocess Dataset

wine = load_wine()
X = wine.data
y = wine.target

# Standardize data
X_scaled = StandardScaler().fit_transform(X)


---

🔧 Step 3 — Apply PCA (Optional but Recommended)

Speeds up t-SNE and reduces noise.

pca = PCA(n_components=min(30, X_scaled.shape[1]), random_state=42)
X_pca = pca.fit_transform(X_scaled)


---

🚀 Step 4 — Apply t-SNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,
    init='pca',
    random_state=42
)
X_tsne = tsne.fit_transform(X_pca)


---

🎨 Step 5 — Visualize Results

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', s=60, edgecolor='k')
plt.title("t-SNE Visualization on Wine Dataset")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(label="Wine Class")
plt.show()


---

🧭 5. PCA vs t-SNE Comparison

X_pca2 = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(12,5))

# PCA
plt.subplot(1,2,1)
plt.scatter(X_pca2[:,0], X_pca2[:,1], c=y, cmap='viridis', s=60, edgecolor='k')
plt.title("PCA (2D Projection)")
plt.xlabel("PCA1"); plt.ylabel("PCA2")

# t-SNE
plt.subplot(1,2,2)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', s=60, edgecolor='k')
plt.title("t-SNE (2D Projection)")
plt.xlabel("tSNE1"); plt.ylabel("tSNE2")

plt.tight_layout()
plt.show()

###
---

📊 6. Observations

t-SNE reveals well-separated clusters corresponding to different wine types.

PCA provides linear separation, while t-SNE exposes non-linear boundaries.

Each color represents a class of wine; close points are similar in chemical composition.



---

🧠 7. Insights & Use Cases

✅ Used for visualizing high-dimensional data like:

Word embeddings (Word2Vec, BERT)

Image embeddings (CNN feature maps)

Customer segmentation

Genomic or medical data analysis


✅ Great for:

Exploratory data analysis (EDA)

Checking class separability

Visualizing results of unsupervised models



---

📘 8. Summary Table

Stage	Description	Formula / Tool

Step 1	Compute similarities in high-dim	Gaussian 
Step 2	Compute similarities in low-dim	Student-t 
Step 3	Minimize KL divergence	KL(P
Step 4	Gradient descent optimization	max_iter, learning_rate
Step 5	Visualization	Matplotlib/Seaborn



---

🧾 9. References

van der Maaten, L. & Hinton, G. (2008): “Visualizing Data using t-SNE”

Scikit-learn Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html



---

💡 10. Note

This project is part of my Unsupervised Machine Learning series, where I’m learning and implementing each algorithm from mathematical theory to practical visualization — focusing on intuition, math, and real-world usage.
