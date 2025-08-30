
## Computer Vision Classifier Pipeline (L2 → L6)

###### **1. Input**  
📥 Image (raw pixels, e.g. 32×32×3)

⬇️

###### **2. Classifier**  
🔹 Linear Score Function
- (level3 SVM loss) => linear score (loss) function = f(x,w,b) = wx + b 
- L1, L2 regularization -> L2(keep all features but shrinks them), L1(kills irrelevant features).
- regression:
	- linear reg: straight line/hyperplane.
	- logistic reg: sigmoid(linear reg) + squash between 0 -> 1
	- NNs: stack multiple logistic functions.
- WW: weights (rows = classes, cols = pixels/features)
    
- bb: bias

- linear classifiers: view points
	- visual view point: reshape the weight matrix w into same shape as the image to create a template of each category, then multiply by x.
	- fails similar context backgrounds.
	- single template, no capture for different poses, all poses in one template.
	- ![[Pasted image ٢٠٢٥٠٨١٦١٠٠٨١٠.png]]
	
- NNs:
	- 5.1
		- problems with linear classifiers: [^1]
			- can't linearly seperate all data types + uses one template/class => can't capture all different modes of a class.
		- solutions:
			- feature transformation.
			- image features:
				- color histogram
				- bag of words
				- HoG
			- feature extract => includes human => slow operation.
			- NNs computer does it auto.
	- 5.2
		- **activation function:**
			- if not used => no "layering" actually happens, it'll collapse into one matrix multiplication again!
			- effect of 1 hidden layer -> linear func, several hidden layers -> nonlinear function.
		- in **features extraction**
			- sometimes even feature transformation doesn't create linearly separable classes, we need representative activation function=> ReLU => results linearly separable classes. 
		- **hidden layers:**
			- more hl => more overfitting the data => don't reduce the hl size, use REGULARIZATION(L2).
		- **universal approximation**:
	- 5.3 convex function
		- like a bowl shaped functions.
		- easier to optim
		- ![[Pasted image ٢٠٢٥٠٨١٦٢٢٢٧٢٨.png]]
		- 
- Neural Networks (L5)
	├─ Why Nonlinear? → transform data (e.g. polar) for separability
	├─ Learnable Feature Transforms (vs hand-crafted)
	├─ Stack Layers + ReLU → build complex representations
	│   └─ Space folding visual: folds flatten complex patterns
	├─ Universal Approximation → any continuous function (in theory)
	│   └─ Bump functions via shifted ReLU sums
	├─ Non-convexity → no optimal guarantee, but works well in practice
	└─ Contrast with convex linear classifiers

    

⬇️

###### **3. Loss Functions** _(Lecture 3 & 4)_

-  give some badness score between the calculated label and the actual truth(y).

- **Multiclass SVM (hinge loss)**
    
    Li=∑j≠yimax⁡(0,sj−syi+Δ)L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)
    - Margin-based, wants correct class score higher by Δ aka 1.
    - once calculated, changing the classes a bit won't change the overall predicted class.
    - we sum over the incorrect classes only(correct will add 1)
        
- **Softmax / Cross-Entropy**
    
    Li=−log⁡esyi∑jesjL_i = -\log \frac{e^{s_{y_i}}}{\sum_j e^{s_j}}
    - Probabilistic interpretation.
    - logits -> e(un-norm) -> sum to 1(norm) -> 
    - then cross-entropy loss -> -log(probs).

⬇️

###### **4. Regularization** _(L2 recap from Lecture 2, fully discussed L5)_

- **L2**: λ∑W2\lambda \sum W^2 → smooth weight decay.
    
- **L1**: λ∑∣W∣\lambda \sum |W| → sparsity.
    

⬇️

###### **5. Optimization**(grads) _(Lectures 5 & 6)_

back prop:
	- downstream grad: grad regarding the input var
	- local grad
	- upstream grad: grad regarding output var
	- too much computation =>use sigmoid as local grad.
	- ![[Pasted image ٢٠٢٥٠٨١٦٢٣٣٤١٥.png]]
	- [^2]
	- can use ReLU instead of sigmoid for the grad calculations => one of grade hot coded matrix => any<0 = 0 ===> bigger matrix, greater loss (only diagonal is numbers!)
	- to solve: grad = grad if x>0, 0 otherwise.
- **gradient** -> where's the DIRECTION OF MAX of function 
	- hidden layers: jacobian matrix => derivative of neurons' vector parameters.
	- last layer: gradient vector => derivative of loss (scalar) wrt parameters == mul(jacobian matrices) => scalar new loss value.
	- computing gradient: 
		- numeric: approximate, slow
		- analytic: exact, fast
		- grad check => use analytic, check with numeric
- **Gradient Descent**: update weights with derivative(gradient) of loss.
    - we got all gradients => update parameters: w = w - a*grad(w).
    - 
- **Variants**:
    - Stochastic Gradient Descent (SGD)
        - ![[Pasted image ٢٠٢٥٠٨١٦١٣٥٧٣٥.png]]
    - Mini-batch GD
		- approximate sum using mini-batch of examples => 1 data point/step
    - Learning rate schedules, momentum, etc.
        - ![[Recording ٢٠٢٥٠٨١٦١٥٤٤٥٦.webm]]
        - ![[Recording ٢٠٢٥٠٨١٦١٥٤٧٤١.webm]]
        - ![[Recording ٢٠٢٥٠٨١٦١٥٥٣٠١.webm]]
- **computational graph:** 
⬇️

###### **6. Results / Evaluation** _(covered at end L6+)_

- Accuracy on train/val/test.
    
- Overfitting vs underfitting.

lec2-6:
- input -> features -> classifier -> loss -> optim
- (raw pixels) -> (raw or hand-crafted) -> linear(scores) -> how wrong -> adjust w,b (backprop)

- non-linearity:
	- ![[Recording ٢٠٢٥٠٨١٦١٢٤٣٤٤.webm]]
[^1]: https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-5-Neural-Network?utm_source=chatgpt.com
	

[^2]: https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-6-Backpropagation




