
## Lecture 12: Recurrent Neural Networks — Detailed Skeleton

---

### 1. **Why RNNs? When Sequential Data Needs Memory**

- **Problem**: Traditional feedforward networks treat each input independently and can’t handle sequences like text, speech, or time series effectively.
    
![[Recording ٢٠٢٥٠٨٢٠١٩٢٤١٦.webm]]

- **RNN’s Solution**: Maintains an evolving **hidden state**, `h_t`, that summarizes past context and gets updated at each time step (`h_t = f_W(h_{t–1}, x_t)`).
       x_t + h_(t-1) ---> [RNN Cell] ---> h_t ---> y_t
	                ^          |
                    |--------|


---

### 2. **Vanilla RNN Architecture & Unfolding Over Time**

- **Core idea**: Share a single set of parameters across all time steps.
    
- **Illustration**: Unfold the recurrent cell into a chain—a deep network unrolled through time.
    - recurrent => same weight matrices Wx,Wh every step.
    - ![[Pasted image ٢٠٢٥٠٨٢٠١٩٣٠٤٧.png]]
- **Common Use Cases**:
    
    - **Many-to-One**: Sentiment classification (e.g., "text → sentiment").
        
    - **One-to-Many**: Caption generation (e.g., "image encoding → sentence").
        
    - **Many-to-Many**: Sequence translation (e.g., "English sequence → French sequence").  
        ([Gautam Kunapuli](https://gkunapuli.github.io/files/cs6375/18-RNNs.pdf?utm_source=chatgpt.com "Recurrent Neural Networks"), [Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network?utm_source=chatgpt.com "Recurrent neural network"))
        

---

### 3. **Training RNNs via Backpropagation Through Time (BPTT)**

- **Process**: Unroll the RNN across time steps, compute loss, then backpropagate through this unrolled graph—summing gradients across all copies of the recurrent cell.  
    ([Wikipedia](https://en.wikipedia.org/wiki/Backpropagation_through_time?utm_source=chatgpt.com "Backpropagation through time"))
    
- **Truncated BPTT**: To manage compute, carry hidden states forward but only backpropagate through a limited window of steps.
    

---

### 4. **The Vanishing & Exploding Gradient Problem**

- **Issue**: Repeated multiplication through time causes gradients to diminish (vanish) or escalate (explode).
    
- **Key theoretical insight**: If the largest singular value of the recurrent weight matrix is less than 1 → vanishing gradients; if greater than 1 → exploding.  
    ([Gautam Kunapuli](https://gkunapuli.github.io/files/cs6375/18-RNNs.pdf?utm_source=chatgpt.com "Recurrent Neural Networks"), [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation_through_time?utm_source=chatgpt.com "Backpropagation through time"), [arXiv](https://arxiv.org/abs/1211.5063?utm_source=chatgpt.com "On the difficulty of training Recurrent Neural Networks"))
    
- **Solution**:
    
    - Gradient clipping: Scale back large gradients to keep them in check.
        
    - Employ specialized RNN architectures like LSTM and GRU that manage memory flow better.
        

---

### 5. **Gated RNNs: LSTM & GRU**

- **LSTM (Long Short-Term Memory)**:
    
    - Uses **input**, **forget**, and **output gates** to maintain long-term memory via additive pathways—mitigating vanishing gradients.  
        ([Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network?utm_source=chatgpt.com "Recurrent neural network"))
        
- **GRU (Gated Recurrent Unit)**:
    
    - Combines gates into **update** and **reset** mechanisms—simpler and fewer parameters compared to LSTM.  
        ([Wikipedia](https://en.wikipedia.org/wiki/Gated_recurrent_unit?utm_source=chatgpt.com "Gated recurrent unit"))
        

---

### 6. **RNNs Applied to Vision + Language Tasks**

- **Image Captioning**: Feed an image’s CNN encoding as an initial hidden state to an RNN, which then generates a sequence of words (caption).  
    ([Gautam Kunapuli](https://gkunapuli.github.io/files/cs6375/18-RNNs.pdf?utm_source=chatgpt.com "Recurrent Neural Networks"))
    
- **Visual Question Answering**: Combine visual features with RNN-encoded questions to predict answers—bridging visual understanding and language.
    

---

### 7. **Summarizing the Role of RNNs & Looking Forward**

- RNNs introduce structured, sequential modeling capabilities—key when order and context matter.
    
- Gating mechanisms like LSTM/GRU are central to training deeper temporal models effectively.
    
- Upcoming Lecture 13 will cover **Attention mechanisms**, which complement RNNs by letting models “focus” adaptively across sequence steps.
    
