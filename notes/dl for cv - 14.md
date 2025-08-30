
---
======
## Lecture 14 – Visualizing & Understanding CNNs

### 1. **Why bother visualizing CNNs?**

- CNNs often called _black-boxes_—good performance, poor interpretability.
    
- Visualization helps us:
    
    - **Diagnose** mistakes (e.g. shortcut learning or biases).
        
    - **Interpret** learned features — “what does this filter detect?”
        
    - **Refine models** effectively, by understanding learned primitives.
        

_Analogy:_ It’s like peeking into a recipe while cooking—seeing ingredients (filters), steps (layers), and how they combine—not just the final dish.

---

### 2. **Visualizing First-Layer Filters**

- These filters are human-interpretable: edges, colors, gradients, simple textures.
    
- We can directly visualize them as small image patches.
    

_Analogy:_ They’re like seeing the primary colors on an artist’s palette—it tells you what the network mixes later.

---

### 3. **Activation Maps & Max-Activating Patches**

- **Activation maps**: outputs of filters across images—show how strongly each responds spatially.
    
- **Max-activating patches**: for each filter, find image regions that trigger it most.  
    → Reveals what high-level features (e.g. dog eyes, textures) that filter is tuned to.
    

---

### 4. **DeconvNet / Input Reconstruction**

- Technique from Zeiler & Fergus to invert feature maps → reconstruct input patterns.
    
- Helps localize what pixel patterns caused a high-level signal.
    

_Simplified intuition:_ Undo the CNN’s steps to “see what pattern lit up this neuron.”

---

### 5. **Occlusion Experiments**

- Systematically slide a blank patch over input image.
    
- Track how predicted class confidence drops—build a heatmap of sensitivity.  
    → Reveals critical regions the network relies on, without gradient math.
    

---

### 6. **Saliency Maps**

- Compute gradient of class score wrt input pixels.
    
- Visualize absolute, max-channel gradient as heatmap—shows which pixels affect output most.
    

_Metaphor:_ It’s like checking where your fingers tingle when someone lightly taps different parts of your body—denoting critical zones.

---

### 7. **CAM & Grad-CAM – Class Activation Mapping**

- **CAM** (requires GAP): uses classifier weights to produce class-sensitive heatmaps.
    
- **Grad-CAM** (flexible): computes gradient of target class wrt feature maps for localization.  
    Use ReLU to highlight positive contribution regions ([Wikipedia](https://en.wikipedia.org/wiki/Class_activation_mapping?utm_source=chatgpt.com "Class activation mapping"), [arXiv](https://arxiv.org/abs/1610.02391?utm_source=chatgpt.com "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization")).
    

_Analogy:_ Imagine spotlighting where on the input the model looked to guess "cat".

---

### 8. **Guided Backpropagation & Guided Grad-CAM**

- **Guided backprop** filters gradients: only positive forward and backward activations pass—generates crisp edges.
    
- Multiply with Grad-CAM maps → **Guided Grad-CAM** gives class-sensitive, high-resolution maps.
    

---

### 9. **Feature Embedding Visualizations (PCA / t-SNE)**

- Extract features (e.g. FC7 responses) across dataset → project to 2D with t-SNE or PCA.
    
- Visual clusters reveal semantic groupings (cats cluster, dogs cluster, etc.).
    

---

### 10. **Transfer Learning – Feature Generalization**

- Feature embedding from ImageNet-trained CNN applied to new datasets (e.g. Caltech-101).
    
- Retrieval of visually and semantically similar images (nearest-neighbor) reveals generality of features.
    

_Analogy:_ How well a language translator for English works when applied to similar languages—shows robustness.

---

## Visual Skeleton (Ready to Fill with Phrases/Analogies)

```
1. Motivation: why peek inside CNNs?
2. First-layer filter gallery
3. Activation maps & exemplars
4. Input reconstruction via DeconvNet
5. Spatial importance via occlusion
6. Pixel-level saliency with input gradients
7. Class-level localization: CAM / Grad-CAM
8. High-res fusion: Guided Grad-CAM
9. Feature clustering: PCA / t-SNE
10. Feature transfer: generalization via retrieval
```

