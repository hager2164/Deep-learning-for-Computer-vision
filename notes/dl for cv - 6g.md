

---

# 📒 Lecture 2–6 Notes (Clean Skeleton)

---

### **1. Input → Features → Classifier → Loss → Optimizer**

- Input: raw pixels.
    
- Features: raw or hand-crafted.
    
- Classifier: linear score function.
    
- Loss: how wrong are we?
    
- Optimizer: update weights (backprop).
    

---

### **2. Linear Classifiers (Lecture 2–3)**

- Score function: f(x,W,b)=Wx+bf(x, W, b) = Wx + b.
    
- WW: rows = classes, cols = features/pixels.
    
- Bias bb.
    
- **Visual view**: reshape WW to image template.
    
- **Limitations**:
    
    - Single template per class.
        
    - Can’t capture pose/background variations.
        

---

### **3. Loss Functions (Lecture 3–4)**

#### 🔹 SVM Loss (hinge)

Li=∑j≠yimax⁡(0,sj−syi+Δ)L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)

- Margin-based (wants correct score > incorrect by Δ).
    
- Insensitive to small changes once margin satisfied.
    

#### 🔹 Softmax + Cross-Entropy

Li=−log⁡esyi∑jesjL_i = -\log \frac{e^{s_{y_i}}}{\sum_j e^{s_j}}

- Probabilistic interpretation.
    
- Logits → exponentiate → normalize → probabilities.
    
- Loss = negative log of correct class probability.
    

---

### **4. Regularization (Lecture 5)**

- Prevents overfitting, controls complexity.
    
- **L2**: λ∑W2\lambda \sum W^2 → smooth shrinkage.
    
- **L1**: λ∑∣W∣\lambda \sum |W| → sparsity (feature selection).
    

---

### **5. Optimization (Lecture 5–6)**

#### Gradients

- Gradient = direction of steepest ascent (↑).
    
- Descent = step opposite (↓).
    
- Backprop = chain rule with Jacobians.
    

#### Gradient Checking

- Numeric ≈ check.
    
- Analytic = exact (used in training).
    

#### Gradient Descent Variants

- **Batch GD**: all data.
    
- **Stochastic GD**: 1 sample at a time.
    
- **Mini-batch GD**: small random subset (default in practice).
    

#### Tricks

- Learning rate schedule.
    
- **Momentum**: smooths updates, accelerates downhill.
    
- **Adam**: adaptive learning rate + momentum.
    

---

### **6. Evaluation (Lecture 6)**

- Train / Validation / Test splits.
    
- Accuracy = % correct.
    
- Overfitting vs Underfitting tradeoff.
    

---

⚡ That’s the **high-level skeleton for Lectures 2–6**.  
Next, I can:

1. Expand each section into a **1-page “cheat sheet”** with equations + tiny visuals.
    
2. Or keep it like this as a **quick revision list**.
    

👉 Do you want me to make the **cheat sheet (visual + compressed equations)** version for you now?