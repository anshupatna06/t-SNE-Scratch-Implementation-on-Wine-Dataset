# t-SNE-Scratch-Implementation-on-Wine-Dataset
"ML models implemented from scratch using NumPy and Pandas only"

🧭 t-SNE (t-Distributed Stochastic Neighbor Embedding) — From Theory to Implementation

📊 Applied on: Wine Dataset (Unsupervised Learning)

This project demonstrates how t-SNE can be used to visualize high-dimensional data by projecting it into a lower-dimensional space (2D).
It includes both mathematical intuition and practical implementation using Scikit-learn.


---

🧠  What is t-SNE?

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a non-linear dimensionality reduction technique mainly used for data visualization.
It preserves the local structure of data — i.e., points that are close in high-dimensional space remain close in low-dimensional space.


---



t-SNE is ideal for understanding clustering and visualizing embeddings in complex datasets like Wine, MNIST, and others.


---

🧩  Mathematical Intuition

Step 1 — Similarities in High-Dimensional Space

For each pair of points $$x_i$$, $$x_j$$compute the probability that $$x_j$$ is a neighbor of $$x_i$$:

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
⚙️ Workflow

1. Data Preprocessing — Standardized the Wine dataset.


2. PCA Reduction — Reduced noise and optimized computation.


3. t-SNE Embedding — Transformed high-dimensional data into 2D space.


4. Visualization — Compared PCA vs t-SNE for structure clarity.




---

🧠 Key Insights

t-SNE reveals clearer class separation than PCA.

Great for exploring non-linear relationships in data.

Preserves local neighborhood similarity effectively.



---

🧩 Tools & Libraries

Python · NumPy · Matplotlib · Scikit-learn · PCA · t-SNE


---



---

📊  Observations

t-SNE reveals well-separated clusters corresponding to different wine types.

PCA provides linear separation, while t-SNE exposes non-linear boundaries.

Each color represents a class of wine; close points are similar in chemical composition.



---

🧠  Insights & Use Cases

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


---

🧾  References

van der Maaten, L. & Hinton, G. (2008): “Visualizing Data using t-SNE”

Scikit-learn Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html



---
💡 Learning Goal

Understanding how t-SNE works mathematically (KL Divergence, Gaussian & Student-t distributions) and practically (cluster visualization) as part of my Unsupervised ML learning series toward a Microsoft AI Internship goal.

💡  Note

This project is part of my Unsupervised Machine Learning series, where I’m learning and implementing each algorithm from mathematical theory to practical visualization — focusing on intuition, math, and real-world usage.

