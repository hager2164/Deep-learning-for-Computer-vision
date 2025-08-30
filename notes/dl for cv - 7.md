
---

## Lecture 7: Convolutional Networks — Detailed Skeleton

---

### 0. **Lecture Metadata**

- **Instructor:** Justin Johnson, University of Michigan, 2019 Deep Learning for Computer Vision
    
- **Topics Covered:** Convolution, Pooling, Batch Normalization
    
- **Video Duration:** ~1:12:03 (YouTube) ([web.eecs.umich.edu](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html?utm_source=chatgpt.com "Schedule | EECS 498-007 / 598-005: Deep Learning for ..."))
    
- **Korean note summary**: available for conceptual guidance ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-7-Convolutional-Networks?category=1148106 "[EECS 498-007 / 598-005] Lecture 7: Convolutional	Networks"))
    

---

### 1. **Motivation & Biological Inspiration** (0:00 – ~5:00)

- **Problem with MLPs on images:**
    
    - Flattening 2D spatial data loses structure.
        
    - Parameter explosion: e.g., 400×700×3 image → 840,000 inputs, millions of weights even for a small hidden layer. ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-7-Convolutional-Networks?category=1148106 "[EECS 498-007 / 598-005] Lecture 7: Convolutional	Networks"))
        
- **Biological analogy (Hubel & Wiesel):**
    
    - Neurons respond to local regions (receptive fields) in the visual cortex. Circa 1960s—foundation for convolution. ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
        

---

### 2. **Convolutional Layer Fundamentals** (~5:00 – ~20:00)

- **Definition**:
    
    - Filter = small spatial window (e.g., 5×5), applied across image via dot product.
        
        - Takes full depth of input.
            
    - Results in a feature map (activation map). ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
        
- **Stacking multiple filters** → multiple activation maps → form new depth channels.
    
- **Parameter computation example**:
    
    - Input: 32×32×3, filter: 5×5×3, stride 1 → output: 28×28 spatially.
        
    - If 10 filters: output = 28×28×10. ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
        
    - Number of parameters = (filter_size × depth + bias) × number_filters. Example: 76 × 10 = 760. ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
        
- **Hyperparameters**:
    
    - Stride, padding, filter size control resolution and spatial shrinkage. Using zero-padding of (F−1)/2 preserves spatial size. ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
        

---

### 3. **Activation & Pooling Layers** (~20:00 – ~30:00)

- **ReLU (nonlinearity)**:
    
    - Applied elementwise after convolution, keeps positive values, introduces nonlinearity.
        
- **Pooling (e.g., 2×2 max pooling)**:
    
    - Downsamples spatial dimensions, retains depth.
        
    - Introduces modest translation invariance. ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
        
- **Layer Pattern Summary**:
    
    - Build CNNs via repeating **[Conv → ReLU → Pool]** blocks.
        

---

### 4. **CNN as a Hierarchical Feature Extractor** (~30:00 – ~40:00)

- Early filters detect edges/colors.
    
- Subsequent layers detect textures, parts, and eventually semantic concepts.
    
- CNN layers form a **hierarchy of feature abstraction**—all feed into deeper layers (shown via visuals) ([cs231n.stanford.edu](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf?utm_source=chatgpt.com "Lecture 7: Convolutional Neural Networks"))
    

---

### 5. **CNN Forward & Backward Mechanics** (~40:00 – ~50:00)

- **Forward pass**: Convolution → ReLU → Pool → (repeat).
    
- **Backward pass**:
    
    - Shared filter weights receive gradients from multiple positions.
        
    - Pooling layers backward-pass only to maxima locations.
        
    - Backprop remains chain-rule — Conv differs only in weight sharing/local connectivity.
        

---

### 6. **Flatten + Fully Connected + Softmax** (~50:00 – ~55:00)

- After several conv/pool stages, flatten the activation volumes.
    
- Feed into FC layers for classification.
    
- Same softmax-cross-entropy loss as in Lecture 4.
    

---

### 7. **Batch Normalization (BN)** (~55:00 – ~65:00)

- **Purpose**: Mitigates internal covariate shift—normalizes activations within mini-batches.
    
- **Process**:
    
    - Compute batch mean and variance, normalize activations.
        
    - Scale and offset via learned parameters (γ, β).
        
    - In testing: use running averages, not batch stats.
        
- BN **stabilizes training**, allows for higher learning rates, acts like a regularizer.
    

---

### 8. **Summary Slide & In-class FAQ** (~65:00 – ~72:00)

- **Typical CNN architecture formula**:
    
    ```
    [(Conv → ReLU) * N → Pool?] * M → (FC → ReLU) * K → Softmax
    ```
    
- Jonathan may illustrate via CIFAR-10 demo (ConvNetJS) or show practical filters learned visually.
    
- He might also link to Assignment 2 where CNN modules build upon what you already coded as FC.
    

---

## Summary Outline (For Your Notes)

```
Lecture 7 – Convolutional Neural Networks
A. Motivation (MLPs inefficient for images; biological receptive fields)
B. Convolutional Layer
   1. Filter sliding = local connectivity
   2. Multiple filters = activation map depth
   3. Parameter count & example math
   4. Hyperparameters (stride, padding, size)
C. Activation & Pooling
   5. ReLU nonlinearity
   6. Max pooling and translation invariance
   7. ConvNet block pattern
D. Hierarchical Feature Extraction
E. Forward/Backward Mechanics (backprop with conv/pool)
F. Flatten → FC → Softmax (classification layer)
G. Batch Normalization: training vs test behavior, benefits
H. Recap Slide & Demo Examples
```

---

###
![[Pasted image ٢٠٢٥٠٨١٧٢٢٥٤١٦.png]]
#### 🔹 1. Input + Convolution

- The **image (input)** is passed through a **kernel (filter of weights)**.
    
- Each filter learns to detect a certain feature (edges, curves, textures, colors, etc.).
    
- The result is a **feature map** (also called activation map).
    

👉 **Key:** Multiple filters = multiple feature maps. They don’t collapse — they coexist as separate channels.

---

#### 🔹 2. Non-linearity (ReLU)

- After convolution, you apply a **ReLU (Rectified Linear Unit)** to each feature map **independently**.
    
- ReLU sets all negative activations to 0 but keeps positive values.
    
- This introduces **non-linearity**, which prevents the CNN from being just a linear transformation (which would collapse feature maps into something like one big linear combination).
    

👉 **So:** ReLU doesn’t “connect” different feature maps. It just cleans and sharpens each one separately.

---

#### 🔹 3. Pooling

- Pooling (often max pooling) reduces the **spatial dimension** (height × width), but **not the depth** (number of feature maps).
    
- If you had 32 feature maps before pooling, you’ll still have 32 after pooling — just smaller (compressed) ones.
    

👉 Pooling makes the representation more compact and translation-invariant.

---

#### 🔹 4. Stacking Layers

- You then apply another convolution layer → more filters → **new set of feature maps**.
    
- This process repeats: Convolution → ReLU → Pooling → …
    
- Each deeper layer combines lower-level features (edges) into higher-level features (shapes, eyes, wheels, etc.).
    

👉 So the feature maps stack like hidden layers in a regular NN, but they preserve spatial structure.

---

#### 🔹 5. Flatten → Fully Connected Layers

- After several convolution + pooling stages, you **flatten** the 3D tensor (height × width × depth of feature maps) into a long vector.
    
- This vector goes into a **fully connected layer** (like a normal NN).
    
- The last fully connected layer feeds into a **softmax**, which outputs probabilities for each class.
    

---

#### 🔹 To Answer Your Question Directly:

- **Do activation maps collapse into one?** → No. Each filter produces its own activation map, and they stack as channels.
    
- **What connects them?** → Nothing mixes them at the convolution stage; they are parallel. Mixing happens later when multiple filters feed into the next convolution (each new filter takes inputs across _all previous feature maps_).
    
- **What does ReLU do?** → Applied independently per feature map, introduces nonlinearity, keeps positive signals only.
    
- **Is there anything else besides ReLU?** → Yes, sometimes other nonlinearities are used (Sigmoid, Tanh, Leaky ReLU, ELU, GELU…), but ReLU is standard because it’s simple and avoids vanishing gradients.
    

---

👉 So the diagram shows **Representation (left: conv+ReLU+pool)** and **Classification (right: flatten + FC + softmax)**.  
The feature maps don’t collapse into one — they grow into a richer, deeper set until flattening.

---
- MLPs require flattening images, **destroying spatial structure**—like treating every pixel as unrelated. This makes learning both inefficient and prone to overfitting + to actually connect them in fully connected nn => no. of parameters will increase exponentially.
    
![[Recording ٢٠٢٥٠٨١٧١٦٤٥٤٠.webm]]

) L2–L6 (what you already built)

- You learned the **generic pipeline**: input → features → classifier → loss → backprop/optim.
    
- In practice you used **fully connected (FC) layers** (your hand-drawn W₁/W₂ in A2).
    
- The slide you translated shows **why FC on images explodes parameter count** (e.g., 400×700×3 inputs → 840k inputs → billions of weights if you use a modest hidden layer).
    
    - Consequences: slow learning, overfitting risk, and you **ignore spatial structure** when you flatten.
        

### 2) L7–L9 (what you’re starting now)

- CNNs are introduced **exactly to fix those issues**:
    
    - **Local connectivity** (small receptive fields) and **weight sharing** (same filter slides) → **huge parameter drop**.
        
    - **Translation tolerance** via weight sharing + (often) pooling → features detected anywhere.
        
    - Same loss/optim/backprop as L2–L6, but with **new layer types** (conv/pool) and **structured inductive bias**.
        

> Mental model:- 
> FC on images = parameter explosion + loses geometry.
> CNN = **keeps geometry**, **slashes parameters**, **generalizes better**.

- ReLU activations and backprop remain consistent with earlier lectures, but **this structural constraint is what makes CNNs efficient and powerful**, especially visible around the start of Lecture 7 (~0:00–8:30)

---

### 🔹 Step 1: Filter → Activation Map (Linear Step)

- Each filter does a **linear operation** (dot product).
    
- Without anything else, this is just like a big linear layer:
    
    - multiple filters would just be different linear projections of the same image.
        
    - If you stacked only linear filters, the entire CNN would still be **linear overall**.
        

That means: **stacking convolutions alone ≈ still a linear model** → no extra expressive power, no nonlinearity.

---

### 🔹 Step 2: Nonlinear Function (ReLU)

- After each convolution, we pass the activation map through a **nonlinear activation function** (commonly ReLU).
    
- **ReLU(x) = max(0, x)**.
    

👉 What this does:

1. **Separates features**: keeps positive activations (strong matches with the filter), kills negative ones.
    
2. **Prevents collapse**: if everything stayed linear, all filters could be combined into one equivalent linear transformation. ReLU makes the system **piecewise linear** and therefore _non-collapsible_.
    
3. **Interaction between feature maps**: indirectly, yes.
    
    - Each feature map is individually ReLU’d.
        
    - But because ReLU is nonlinear, when the next layer takes _all_ feature maps as input, the resulting filters can combine them in **nonlinear ways** (edges + curves = corners, etc).
        

---

### 🔹 Step 3: Stacking Multiple Feature Maps

- Each filter gives its own activation map.
    
- After ReLU, you stack them along the depth axis.
    
- Next layer’s filters slide not just over the image pixels, but over **all the activation maps at once**.
    
    - So a filter in the 2nd layer is 3D: it spans (height × width × depth of input feature maps).
        
    - That’s where maps “interact” — not directly with each other, but through the **next set of filters**.
        

---

### 🔹 Step 4: Beyond ReLU (Other Nonlinearities)

- CNNs can use other nonlinearities, though ReLU is most common:
    
    - **Sigmoid / Tanh** (older, but vanish gradients).
        
    - **Leaky ReLU / ELU** (variants that let negative info leak through).
        
    - **Maxout** (rare, but learns its own activation).
        

But Jonathan (2019) focuses on **ReLU**, because it’s simple and works really well.

---

### 🌱 To tie this back to your question:

- Filters by themselves would indeed just give different _linear_ maps.
    
- ReLU makes them **nonlinear + separable**.
    
- Next-layer filters mix them together again → giving rise to richer representations.
    

That’s why CNNs work:  
**Linear (conv) → Nonlinear (ReLU) → Linear (conv) → Nonlinear (ReLU)…**  
= hierarchical feature learning.

---
