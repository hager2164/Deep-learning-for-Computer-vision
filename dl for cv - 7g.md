
---
### 📚 **Big Framework: CNN Block (Lectures 7–9)**

#### 1. **Motivation & Transition from MLP to CNNs**

- (L7 ~0:00–8:30)
    
    - MLPs treat images as flat vectors → lose spatial structure.
        
    - CNN idea = exploit **locality** + **weight sharing** → fewer parameters, better inductive bias.
        
    - Circle back: same ReLU, same backprop pipeline as lectures 3–6, but new layer types.
        

---

#### 2. **Core CNN Building Blocks**

- (L7 ~8:30–35:00)
    
![[Recording ٢٠٢٥٠٨١٧٢٣٠٢٥٢.webm]]
![[Pasted image ٢٠٢٥٠٨١٧٢٣١٩٤٨.png]]

- **Convolutional Layer**
	
	- Local receptive fields (patches).
		
	- Weight sharing: one filter slides across image.
		
	- Output = **feature maps**.
		
- **Hyperparameters**: stride, padding, kernel size.
	
- **Parameter count reduction**: compare FC layer vs conv filter.
	
- Equation form (important slide).
        
- **Pooling Layer** (L7 ~35:00–47:00)
    
    - Max pooling vs average pooling.
        
    - Provides **translation invariance**.
        
    - Circle back: reduces dimensionality similar to earlier regularization tricks.
        
- **Non-linearity**
    
    - ReLU after conv, just like after FC.
        
    - Nothing new mathematically → but crucial in deep CNNs.
        

---


#### 3. **CNN Forward & Backprop Simplification**

- (L7 ~47:00–1:05:00)
    
    - Conv backprop = FC backprop but with shared weights + local connectivity.
        
    - Introduce **“upstream / downstream gradients”** idea (same as lecture 6 recap).
        
    - Key remark: you don’t need to manually derive conv backprop in practice → frameworks handle it.
        

---

#### 4. **Designing a CNN for Classification**

- (L7 ~1:05:00–end)
    
    - Typical block: Conv → ReLU → Pool → repeat.
        
    - At the end: Flatten → Fully Connected → Softmax.
        
    - Circle back: same softmax cross-entropy loss as Lecture 4.
        

---

#### 5. **Training CNNs in Practice**

- (Lecture 8 ~0:00–35:00)
    
    - **Data preprocessing**: normalization, augmentation.
        
    - **Weight initialization**: He/Xavier (ties to Lecture 5).
        
    - **Regularization**:
        
        - Dropout inside CNNs.
            
        - Weight decay still relevant.
            
    - **Optimization**: SGD vs SGD+Momentum vs Adam.
        
- **BatchNorm** (L8 ~35:00–1:00:00)
    
    - Normalizes activations inside network.
        
    - Helps optimization, stabilizes training.
        
    - Circle back: relates to covariate shift problem in MLPs.
        

---

#### 6. **CNN Architectures**

- (Lecture 9)
    

**a. Early Architectures**

- (L9 ~0:00–25:00)
    
    - **LeNet-5** (1998): first CNN for digits.
        
    - **AlexNet** (2012): breakthrough in ImageNet.
        
        - Tricks: ReLU, Dropout, Data Augmentation, GPU training.
            

**b. Modern Architectures**

- (L9 ~25:00–1:00:00)
    
    - **VGG**: deeper, small conv filters (3x3).
        
    - **GoogLeNet / Inception**: parallel filters, 1x1 convolutions.
        
    - **ResNet**: skip connections → solves vanishing gradient.
        
        - Circle back: ties directly to Lecture 5 vanishing gradients.
            
    - Key remark: architecture innovations = engineering + inductive bias.
        

---

#### 7. **Big Picture & Circle-Back to Lectures 2–6**

- MLPs = general function approximators.
    
- CNNs = MLPs with **structured constraints**:
    
    - Local receptive fields.
        
    - Weight sharing.
        
    - Pooling invariance.
        
- Training CNNs still uses:
    
    - Backprop (Lecture 5–6).
        
    - Cross-entropy loss (Lecture 4).
        
    - SGD optimization (Lecture 2–3).
        

---


# CNN Big Framework (Concept Map with Details)

---

## A. **Why Move from MLPs to CNNs?**

- CNNs impose two important **inductive biases**:
    
    - **Locality**: nearby pixels are related—filters focus on small neighborhoods.
        
    - **Translation invariance**: features found in one region also apply elsewhere, through **weight sharing**.
### 1 B. **Core CNN Components**
![[Pasted image ٢٠٢٥٠٨١٧١٦٠٦٠٠.png]]

### 1. **Convolutional Layer**

- Learns multiple **filters** (e.g., 3×3 kernels) that slide across the image to produce **activation maps**, capturing specific features like edges.
    
- **Hyperparameters** include:
    
    - **Stride**: how far filter moves each step.
        
    - **Padding**: adding zeros around edges to control output size.
        
- This layer is **much more parameter-efficient** than a fully connected layer of similar input dimensions. Slides around halfway through Lecture 7 illustrate the parameter count reduction.
    

### 2. **Activation Function**

- Just like in MLPs, ReLU (`max(0, x)`) is used after convolutions to maintain non-linearity—critical for deep nets. This is emphasized throughout all CNN lectures.
    

### 3. **Pooling Layer**

- Typically **max pooling**, e.g., 2×2 with stride 2, which **downsamples activations** to reduce spatial size and computation.
    
- Provides **translation invariance**—small shifts won’t change pooled output (Lecture 7 mid-section).
    

### 4. **Fully Connected & Output**

- At the end, feature maps are **flattened**, passed through fully connected layers, and fed into **softmax** for classification. This mirrors the final stage of MLPs demonstrated in earlier lectures.
    

---

## C. **Forward & Backward Pass Mechanics in CNNs**

- **Forward Pass**: convolution → ReLU → pooling → flatten → FC → softmax.
    
- **Backward Pass**: gradients are propagated:
    
    - For convolutions: **shared weights** affect multiple positions.
        
    - For pooling: gradient assigned only to max location.
        
- Backprop is operationally similar to MLPs—it uses the **chain rule (Lecture 6)**. It gets highlighted in Lecture 7 when discussing gradient flow through conv/pool layers.
    

---

## D. **Design Patterns for CNN Architectures**

- Standard pattern:  
    **[Conv → ReLU → (Optional Pool)] repeated** several times, then **Flatten → FC → Softmax**.
    
- **Hierarchical feature extraction**:
    
    - Early layers detect edges.
        
    - Middle layers detect motifs or parts.
        
    - Late layers capture object-level representations (shown with visuals in Lecture 9).
        

---

## E. **Practical Training of CNNs**

### 1. **Data & Preprocessing**

- Normalize image inputs (zero mean, unit variance) and apply **data augmentations** like flips and crops to improve generalization—demonstrated in Lectures 8–9.
    

### 2. **Initialization**

- Use **He/Xavier initialization** to keep signal variance stable across layers, essential for CNN depth (Lecture 8).
    

### 3. **Regularization**

- **Weight decay (L2)** prevents overly sharp filters.
    
- **Dropout** (particularly in FC layers) reduces overfitting—similar to its role in MLPs.
    

### 4. **Optimization**

- **SGD**, often with **momentum**, remains the base optimizer.
    
- **Adam** used for faster convergence in CNN training.
    
- **Batch Normalization**: normalizes activations during training, speeding up convergence and improving stability (Lecture 8).
    

---

## F. **Notable CNN Architectures**

- **LeNet-5**: early CNN for digit recognition—conv layers + pooling + FC layers (Lecture 9 early segment).
    
- **AlexNet**: introduced ReLU, dropout, and GPU training (breakthrough in 2012).
    
- **VGG**: deep models using only 3×3 convs—ease of structure, heavy compute.
    
- **GoogLeNet (Inception)**: parallel convs, 1×1 convs for dimensionality reduction.
    
- **ResNet**: introduced **skip (residual) connections** to solve vanishing gradients in very deep nets—direct reference to Lecture 5 concerns with deep nets.
    

---

## G. **Integration with Earlier Lecture Concepts**

- Loss function: **softmax cross-entropy** (Lecture 4).
    
- Optimization: **SGD family and learning rate schedules** (Lecture 4).
    
- Gradient flow: **backprop through layers including spatial conv logic** (Lecture 6).
    
- Activations: **ReLU and nonlinear stacking** (Lecture 5).
    
- Filters = **learnable templates**, building on weight visualization from Lecture 3–5.
    

---

