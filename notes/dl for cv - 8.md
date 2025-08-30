

---

## Lecture 8 – CNN Architectures (Deep Learning for CV)

---

### 0. **Metadata**

- **Title:** CNN Architectures
    
- **Instructor:** Justin Johnson, University of Michigan, Fall 2019
    
- **Materials:** YouTube video, slides (`lecture08.pdf`), and Korean summary from Life AI Learning
    
- **Main Themes:** AlexNet, ZFNet, VGG, GoogLeNet (Inception), ResNet, efficiency vs performance, architectural trends
    

---

### A. **ImageNet & AlexNet as the Turning Point** (~0–10 min)

- **ImageNet Quiz Foundation:**
    
    - Over 1 million images, 1,000 categories; annual ILSVRC competition (Motivates deep CNNs) ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        
- **AlexNet’s Breakthrough (2012):**
    
    - Input: 227×227 RGB image
        
    - Architecture: 5 conv layers (with ReLU), max-pooling, 3 FC layers
        
    - Trained using 2× GTX 580 GPUs ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        
    - **Parameter & computation breakdown:**
        
        - First conv layer: 64 filters → output 56×56 activations
            
        - Memory cost: ~784 KB; Params: ~23.3k; FLOPs: ~72.8M for one conv application ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
            
- **Key insight: bigger model = better accuracy—but at high compute cost**
    

---

### B. **Architectural Variants: Efficiency vs Accuracy**

#### 1. **ZFNet (2013)** (~10–20 min)

- Tweaks on AlexNet:
    
    - Smaller stride (7×7 stride 2 vs AlexNet’s 11×11 stride 4)
        
    - More filters (e.g., 512, 1024, 512 vs AlexNet’s 384–384–256) ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        
    - Result: more compute, better performance.
        

#### 2. **VGG (2014)** (~20–30 min)

- Introduces simplicity via uniform 3×3 conv filters:
    
    - Stacked convs replace larger kernels; same receptive field but fewer parameters
        
    - Pattern: `conv-conv-pool` repeated across 5 stages (depths like 16, 19)
        
    - Pooling doubles channels, halves spatial size ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        

#### 3. **GoogLeNet / Inception (2014)** (~30–40 min)

- Efficiency-focused:
    
    - Aggressive downsampling early (“stem”) to cut memory and compute
        
    - Inception modules combine 1×1, 3×3, 5×5 convs + 1×1 for dimensionality reduction
        
    - Uses **global average pooling** instead of flattening → reduces FC parameters
        
    - Adds **auxiliary classifiers** mid-network to ease training pre-BatchNorm ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        

---

### C. **Architectural Trends & Model Comparison** (~40–45 min)

- Performance vs compute:
    
    - **AlexNet:** Low compute, high parameters
        
    - **VGG:** High compute, high parameters
        
    - **GoogLeNet:** Efficient compute, better accuracy
        
    - Visualized via ImageNet contest results (accuracy vs computational cost) ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        

---

### D. **ResNet – Solving Depth Optimization Issues** (~45–55 min)

- **Depth does not guarantee better performance** (deeper models underfit training data) ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
    
- **Problem diagnosis:** inability to learn identity functions; makes deeper nets hurt performance
    
- **Residual Block Solution:**
    
    - `output = F(x) + x` allows identity mapping if convolution learns zero weights
        
    - Supports very deep learning (e.g., ResNet-18, ResNet-34)
        
    - Bottleneck version: 1×1 → 3×3 → 1×1 for efficiency and nonlinearity ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        
    - Applies global average pooling + aggressive downsampling as in previous models ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
        
![[Recording ٢٠٢٥٠٨١٨٠٨٤٧٤٨.webm]]


---

### E. **Beyond ResNet: ResNeXt & Modern Designs** (~55–60 min)

- **ResNeXt:** parallel residual branches (grouped convolutions) improve efficiency and performance ([The man in the Arena](https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-8-CNN-Architectures "[EECS 498-007 / 598-005] Lecture 8: CNN Architectures"))
    
- Architectural message: stacking modular, repeatable blocks with identity shortcuts and efficient convs works best.
    

---

### F. **Detailed Outline for Your Notes**

```
Lecture 8: CNN Architectures

A. ImageNet & AlexNet
   - Importance of ImageNet & deep CNNs
   - AlexNet architecture, compute/param analysis

B. Evolution of Architectures
   1. ZFNet (improved stride & filter counts)
   2. VGG (use of uniform 3×3 convs & deep stacks)
   3. GoogLeNet (efficient Inception modules, global pooling, auxiliary classifiers)

C. Model Trade-offs
   - AlexNet vs VGG vs GoogLeNet: compute vs accuracy

D. ResNet
   - Depth optimization challenges
   - Residual blocks with identity shortcuts
   - Variants: ResNet-18 vs ResNet-34 with bottlenecks

E. ResNeXt & Emerging Modules
   - Grouped convolution and modular design

F. Recap
   - Deep CNN design principles: module-based, efficient, trainable at depth
```

---
