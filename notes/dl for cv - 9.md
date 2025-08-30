
---

### Lecture 9: Hardware & Software — Detailed Skeleton

---

#### 0. **Lecture Metadata**

- **Lecture Title:** Hardware and Software
    
- **Course:** Deep Learning for Computer Vision, Fall 2019, University of Michigan
    
- **Instructor:** Justin Johnson
    
- **Focus:** Understanding the computational tools and infrastructure that make training and deploying deep neural networks possible.
     
---

#### 1. **Why Hardware & Software Matter**

- Deep learning surged because of **parallelization hardware** (like GPUs).
    
- Modern CNNs require huge compute and memory bandwidth—mastering hardware is key to performance.
    

---

#### 2. **Hardware Platforms Overview**

**a. CPUs (Central Processing Units)**

- Versatile but limited in parallelism → slow for large matrix ops typical in deep learning.
    

**b. GPUs (Graphics Processing Units)**

- Highly parallel—excellent for mini-batch matrix multiplications.
    
- Designed for **floating-point ops**, ideal for conv layers.
    

**c. TPUs (Tensor Processing Units)**

- Custom-designed for deep learning workloads (e.g., TensorFlow).
    
- Optimize for **matrix-tensor multiplications**, with high throughput and energy efficiency.
    

---

#### 3. **Software Abstractions for Deep Learning**

**a. Static Graph Frameworks**

- Examples: TensorFlow (pre-eager), MXNet.
    
- Build computation graphs ahead-of-time → optimized execution.
    

**b. Dynamic Graph Frameworks**

- Examples: PyTorch (used in this course), Chainer.
    
- Build graphs on-the-fly during execution—flexible for control flow and debugging.
    

---

#### 4. **Trade-offs: Static vs Dynamic**

- **Static Graphs:** Better compiler optimization, can deploy to specialized hardware seamlessly.
    
- **Dynamic Graphs:** Easier to write/debug; more interactive—great for research and experimentation.
    

---

#### 5. **Deep Learning Framework Key Features**

- **Auto-Differentiation:** Automates gradient computation, no need for manual backprop.
    
- **Hardware Abstraction:** Let frameworks manage device placement (CPU vs GPU).
    
- **Prebuilt Layers & Optimizers:** Reusable modules like Conv2D, optim.SGD, etc.
    

---

#### 6. **Putting It All Together**

By pairing **powerful hardware** (GPUs/TPUs) with **flexible deep learning frameworks**, you can:

- Train large models quickly.
    
- Experiment rapidly with architectural changes.
    
- Deploy optimized models to production systems.
    

---

#### Lecture 9 Skeleton Summary

```
Lecture 9 – Hardware and Software

A. Motivation
   - Hardware enabled deep learning revolution

B. Hardware Platforms
   1. CPU
   2. GPU
   3. TPU

C. Software Frameworks
   1. Static Graphs (TensorFlow, MXNet)
   2. Dynamic Graphs (PyTorch)

D. Comparison & Trade-offs
   - Performance vs flexibility

E. Key Framework Utilities
   - Auto-diff, device placement, built-in modules

F. Takeaways
   - Hardware x software synergy enables modern deep learning
```

---
