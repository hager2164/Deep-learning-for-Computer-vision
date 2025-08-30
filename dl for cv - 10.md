
---

## Lecture 10: Training Neural Networks I — Detailed + Annotated Notes
---

### 1. **Weight Initialization – That Goldilocks Zone**

- **Key insight:** Avoid weights that are _too small_ (values collapse to zero) or _too large_ (values saturate in nonlinearities). It needs to be “just right.”
    
- **Xavier Initialization**:  
    — Defines the standard deviation as 1/Din1/\sqrt{D_{\text{in}}}, ensuring signal variance is maintained layer to layer.  
    — Generalizes to Conv layers using Din=input channels×kernel height×kernel widthD_{\text{in}} = \text{input channels} \times \text{kernel height} \times \text{kernel width}.
    
- **Derivation insight:**  
    From the transcript:
    
    > _"If we choose the magical value ... variance = 1/Din1/D_{\text{in}}, then the variance of the output yy is equal to the variance of xx."_
    
- **ResNet-specific tweak:**  
    — Best practice: initialize the **last layer of each residual block to zero**, ensuring each block behaves as an identity function at initialization. This avoids exploding variance across multiple blocks.
    

---

### 2. **Registration of Activation Distributions**

The transcript walks through experiments using **tanh** (symmetric) versus **ReLU**:

- With tanh, Xavier init works well.
    
- With ReLU, Xavier causes activations to collapse (many zeros).
    
- **Correction for ReLU:** Multiply initialization variance by 2 → known as **MSR** or **He initialization**.
    

---

### 3. **Regularization Techniques**

Johnson highlights a powerful pattern: introduce randomness during training, then average it out at test time.

#### a) **Dropout**

- Randomly zeroes neuron outputs during training.
    
- Encourages robust, non-coadaptive feature learning.
    
- At test time: average out randomness—commonly done by scaling activations (a.k.a. "inverted dropout").  
    Johnson says:
    
    > _"This expectation is exact only for individual layers ... but it works well enough in practice."_
    

#### b) **Generalization of the pattern: Other stochastic regularizers**

- **DropConnect**: Randomly zero weights instead of activations.
    
- **Stochastic Depth**: Randomly drop entire residual blocks.
    
- **Cutout**: Randomly zero square patch in input images.
    
- **MixUp**: Randomly blend two images and their labels (e.g., 95% cat + 5% dog).  
    MixUp is surprising but effective in small-data regimes.
    

#### c) **Dominant techniques today (especially for ConvNets)**

- **Batch Normalization**: Adds stochasticity via batch stats; regularizes and stabilizes training.
    
- **Weight Decay (L2)**: Penalizes large weights—simple and effective.
    
- **Data Augmentation**: Random flips, crops, and scaling at training time; deterministic average over transformations at test time.
    

---

### 4. **Architectural Relevance**

- For large, fully connected models (e.g., early AlexNet/VGG), **Dropout** was essential.
    
- Modern Conv architectures (ResNet, Inception variants) lean heavily on **batch normalization + weight decay + data augmentation**.
    
- Less commonly: Dropout, but techniques like **Cutout** and **MixUp** shine in low-data regimes like CIFAR-10.
    

---

## At-a-Glance: Key Teaching Moments from Transcript

|Concept|Insight from Justin Johnson’s Words|
|---|---|
|Weight Initialization|“Goldilocks regime” keeps variance stable; applies to both FC and Conv.|
|Activation Stability|Xavier init for tanh works; for ReLU must adjust via MSR (He init).|
|Driving Regularization|Dropout as randomness during training, expectation averaged at test-time.|
|Specialized Augments|DropConnect, Cutout, MixUp as stochastic regularizers for different tasks.|
|Feature Robustness|Encourages models to _learn robust features_, not co-adapted shortcuts.|
