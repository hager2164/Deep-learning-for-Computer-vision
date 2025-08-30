![[Pasted image ٢٠٢٥٠٨١٩١٩٣٣٠٨.png]]
# Comprehensive Notes on Training Neural Networks

## Introduction

This document provides an in-depth explanation of key concepts and techniques involved in training neural networks, drawing insights from the "EECS 498-007 / 598-005 Lecture 11: Training Neural Network" video by Justin Johnson and supplementary notes from the "life-ai-learning.tistory.com" blog. It is designed for beginners in the AI field, particularly computer science students with no prior AI background, aiming to clarify complex topics, explain underlying reasons, and visualize abstract ideas through various methods.

## Prerequisites for Understanding Neural Network Training

To fully grasp the concepts discussed in the lecture and notes, a foundational understanding of several areas is beneficial. This section will break down these prerequisites, explaining them in simple terms.

### 1. What is Machine Learning?

Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn from data without being explicitly programmed. Instead of writing rules for every possible scenario, ML algorithms build a model based on sample data, known as "training data," to make predictions or decisions without being explicitly programmed to perform the task [1]. The core idea is to feed a machine a large amount of data and let it learn patterns and relationships within that data. For example, if you want to teach a computer to identify cats in pictures, you would show it thousands of images labeled as "cat" or "not cat." The machine learning algorithm would then learn to recognize the features that distinguish cats from other objects.

### 2. What is Deep Learning?

Deep learning (DL) is a specialized subfield of machine learning that uses artificial neural networks with multiple layers (hence "deep") to learn from data. Inspired by the structure and function of the human brain, deep learning models can process vast amounts of data and discover intricate patterns [2]. Unlike traditional machine learning algorithms that often require human intervention to identify relevant features in data, deep learning models can automatically learn these features from raw data. This capability makes deep learning particularly powerful for tasks involving complex data like images, audio, and text.

### 3. What are Neural Networks?

At the heart of deep learning are neural networks, computational models designed to mimic the interconnected neurons of the human brain. A neural network consists of layers of interconnected "neurons" or "nodes." Each neuron takes inputs, performs a simple calculation, and then passes the result as an output to other neurons. These connections have associated "weights" that determine the strength of the connection, and the network learns by adjusting these weights during training [3].

**Analogy:** Imagine a neural network as a complex decision-making system. Each neuron is like a tiny decision-maker. When you give it information (input), it processes that information and passes its conclusion to other decision-makers. The strength of the connections between these decision-makers (weights) determines how much influence one decision-maker has on another. Through training, the network learns to adjust these influences to make better overall decisions.

### 4. Basic Calculus and Linear Algebra Concepts

While a deep mathematical background isn't strictly necessary to begin understanding neural networks conceptually, a basic familiarity with calculus (especially derivatives for understanding gradient descent) and linear algebra (vectors, matrices, and their operations) can provide a deeper appreciation of how these networks function and learn. These mathematical tools are the language in which the learning processes of neural networks are described.

## Training Neural Networks: Key Concepts from Lecture 11

This section delves into the core topics covered in Lecture 11, providing detailed explanations and integrating insights from the Korean blog post.

### 1. Learning Rate Schedules

The learning rate is one of the most crucial hyperparameters in training neural networks. It determines the step size at which the model's weights are updated during optimization. A constant learning rate can lead to problems: if it's too high, the model might overshoot the optimal solution and diverge; if it's too low, training can be extremely slow and get stuck in local minima [4]. Learning rate schedules are techniques that adjust the learning rate over time (or epochs) during training to achieve better performance and faster convergence.

**Why we do this:**
The idea behind varying the learning rate is that early in training, when the model is far from optimal, a larger learning rate can help it explore the loss landscape more quickly. As training progresses and the model gets closer to the optimal solution, a smaller learning rate is needed to fine-tune the weights and avoid overshooting. This dynamic adjustment helps the model converge more effectively and efficiently.

**How it works:** Different schedules modify the learning rate based on various functions or conditions:

*   **Step Decay:** This method reduces the learning rate by a certain factor at predefined intervals (e.g., every few epochs). For instance, the learning rate might be halved every 30 epochs. While effective, it requires careful tuning of when and by how much to decay the rate [5].
*   **Cosine Decay:** This schedule uses a cosine function to gradually decrease the learning rate from an initial value to a minimum value. It's often preferred for its smooth decay, which can lead to better performance. It typically only requires setting the initial learning rate and the total number of epochs [6].
*   **Linear Decay:** Similar to cosine decay, but the learning rate decreases linearly over time.
*   **Inverse Sqrt Decay:** Less commonly used, this method decays the learning rate inversely proportional to the square root of the iteration number.
*   **Constant Learning Rate:** While often outperformed by schedules, a constant learning rate can still be effective, especially with adaptive optimizers like Adam or RMSprop [7]. The Korean notes emphasize that for complex optimizers like Adam, a constant learning rate is often recommended.
![[Pasted image ٢٠٢٥٠٨١٩١٦٥٨٠١.png]]

**Impact:** Choosing the right learning rate schedule can significantly impact the training speed, stability, and final performance of a neural network. It helps prevent oscillations, allows for finer adjustments as the model converges, and can lead to better generalization.

### 2. Early Stopping

Training a neural network for too long can lead to a phenomenon called overfitting, where the model performs exceptionally well on the training data but poorly on unseen data. Early stopping is a regularization technique designed to prevent overfitting by monitoring the model's performance on a separate validation set during training. When the performance on the validation set starts to degrade (e.g., validation loss increases or validation accuracy decreases), training is stopped, and the model weights from the best-performing epoch on the validation set are saved [8].

**Why we do this:** 
The goal of training is not just to perform well on the data it has seen (training data) but to generalize well to new, unseen data. Overfitting occurs when the model essentially memorizes the training data, including its noise and peculiarities, rather than learning the underlying patterns. Early stopping acts as a safeguard, ensuring that the model doesn't continue to learn these irrelevant details, thus improving its ability to generalize.

**How it works:** 
During training, in addition to the training loss, the model's performance on a validation set (a portion of the data not used for training) is periodically evaluated. If the validation performance does not improve for a certain number of epochs (known as "patience"), training is halted. The model state (weights) that yielded the best validation performance is then restored and used as the final model [9]. The Korean notes highlight the importance of not stopping based on a single epoch's performance but rather observing a consistent trend of degradation in validation performance.

**Impact:** 
Early stopping is a simple yet powerful technique to prevent overfitting, save computational resources by stopping unnecessary training, and improve the generalization ability of the model.

### 3. Choosing Hyperparameters: Grid Search vs. Random Search

Hyperparameters are settings that are external to the model and whose values cannot be estimated from data. They are set before the training process begins and significantly influence the model's performance. Examples include the learning rate, the number of layers in a neural network, the number of neurons per layer, and regularization strengths. Finding the optimal combination of hyperparameters is crucial for building effective models.

**Why we do this:** 
The performance of a neural network is highly sensitive to its hyperparameters. A poorly chosen set of hyperparameters can lead to a model that either underfits (doesn't learn enough) or overfits (learns too much noise). Hyperparameter tuning is the process of finding the best set of hyperparameters for a given task and dataset.

**How it works:**

*   **Grid Search:** This method involves exhaustively searching through a manually specified subset of the hyperparameter space. You define a discrete set of values for each hyperparameter, and the algorithm evaluates the model's performance for every possible combination of these values [10].
    *   **Pros:** Guaranteed to find the best combination within the defined grid.
    *   **Cons:** Computationally very expensive, especially with many hyperparameters or a wide range of values, as the number of combinations grows exponentially.
*   **Random Search:** Instead of evaluating all combinations, random search samples a fixed number of random combinations from the specified hyperparameter distributions. The Korean notes emphasize that random search is often more efficient than grid search because it explores a wider range of values for each hyperparameter, increasing the chance of finding better performing combinations, especially when only a few hyperparameters are truly important [11].
    *   **Pros:** More efficient than grid search for high-dimensional hyperparameter spaces, as it's more likely to find good values for important hyperparameters.
    *   **Cons:** Not guaranteed to find the absolute best combination, but often finds a good enough solution much faster.

**Impact:** 
Effective hyperparameter tuning is critical for achieving high-performing neural networks. Random search is generally recommended over grid search for its efficiency in exploring the hyperparameter space.

### 4. Diagnosing Training Problems with Loss Curves

Monitoring learning curves—plots of training and validation loss/accuracy over epochs—is essential for understanding the training dynamics and diagnosing common problems like overfitting and underfitting.

**Why we do this:** Learning curves provide a visual representation of how well the model is learning and generalizing. By observing their behavior, we can identify issues early and take corrective actions.

**How it works:**

*   **Overfitting:** Indicated by a large and growing gap between training accuracy (high) and validation accuracy (low), or training loss (low) and validation loss (high). The model is memorizing the training data but failing to generalize to new data. Solutions include increasing regularization (e.g., dropout, weight decay), getting more data, or early stopping [12].
*   **Underfitting:** Indicated by both training and validation accuracy being low, or both training and validation loss being high. The model is not complex enough to capture the underlying patterns in the data. Solutions include using a larger model (more layers, more neurons), training for more epochs, or using a more sophisticated architecture [13]. The Korean notes specifically mention that if there's no gap between training and validation curves and both are increasing, it suggests underfitting.
*   **High Learning Rate Issues:** If the loss does not decrease or even diverges at the beginning of training, it often indicates a learning rate that is too high. The Korean notes illustrate this with a graph where the loss initially increases or oscillates wildly.
*   **Poor Initialization:** If the loss remains stagnant at the beginning, it might suggest issues with weight initialization.
*   **Premature Learning Rate Decay:** The Korean notes point out that applying learning rate decay too early can hinder learning if the model hasn't fully converged with the initial learning rate.

**Impact:** Analyzing learning curves is a fundamental skill for any AI practitioner. It allows for informed decisions about model architecture, hyperparameter tuning, and regularization strategies.

### 5. Model Ensembles

Model ensembling is a technique where multiple independent models are trained, and their predictions are combined to make a final prediction. This often leads to improved performance compared to using a single model.

**Why we do this:** 
The idea is that different models might make different errors, and by averaging their predictions, these errors can cancel each other out, leading to a more robust and accurate overall prediction. It's a way to reduce variance and improve generalization.

**How it works:** 
Typically, several models with different initializations, architectures, or training data subsets are trained independently. During inference (when making predictions on new data), the predictions from all individual models are averaged (for regression tasks) or voted on (for classification tasks) to produce the final output [14]. The Korean notes mention that ensembling can provide a modest performance boost, typically around 2%.

**Impact:** Model ensembling is a simple yet effective way to squeeze out additional performance from trained models, often used in competitive machine learning scenarios.
![[Pasted image ٢٠٢٥٠٨١٩١٨٣٣٢٥.png]]
### 6. Transfer Learning

Transfer learning is a powerful technique that leverages knowledge gained from training a model on one task and applies it to a different but related task. This is particularly useful when you have limited data for your target task.
![[Pasted image ٢٠٢٥٠٨١٩١٩٣٦٣١.png]]
**Why we do this:** 
Training deep neural networks from scratch requires massive amounts of data and computational resources. Transfer learning addresses this challenge by allowing us to benefit from pre-trained models that have already learned rich, generalizable features from very large datasets (e.g., ImageNet for image recognition) [15]. This saves significant time and resources and enables high performance even with small datasets.
![[Pasted image ٢٠٢٥٠٨١٩١٩٣٦٥١.png]]
**How it works:** 
The most common approach involves taking a pre-trained neural network (e.g., a convolutional neural network trained on ImageNet), removing its final output layer (which is specific to the original task), and then adding a new, simpler classifier layer on top. The pre-trained layers are often kept frozen (their weights are not updated during training), and only the new classifier layer is trained on the target dataset. In some cases, a few of the top pre-trained layers might also be fine-tuned (their weights are slightly adjusted) to better adapt to the new task [16].

**Impact:** 
Transfer learning has revolutionized many areas of AI, especially computer vision, by making it possible to achieve state-of-the-art performance on new tasks with significantly less data and training time. It debunks the myth that massive datasets are always required for deep learning.

![[Pasted image ٢٠٢٥٠٨١٩١٩٣٥٢٨.png]]


### 7. parallelism
1. GPU = layer![[Pasted image ٢٠٢٥٠٨١٩١٩٣٩١٢.png]]
2. GPU = model![[Pasted image ٢٠٢٥٠٨١٩١٩٤٠٠٩.png]]
3. GPU = same model![[Pasted image ٢٠٢٥٠٨١٩١٩٤١٣٤.png]]
## Conclusion

Training neural networks is a nuanced process that involves understanding and effectively managing various hyperparameters and techniques. From dynamically adjusting the learning rate to preventing overfitting with early stopping, and from efficiently tuning hyperparameters to leveraging pre-trained models with transfer learning, each concept plays a vital role in building robust and high-performing AI models. By grasping these fundamentals, beginners can navigate the complexities of neural network training and build a strong foundation for further exploration in the field of artificial intelligence.

## References

[1] GeeksforGeeks. (2025, August 7). *What is a Neural Network?*. Retrieved from https://www.geeksforgeeks.org/machine-learning/neural-networks-a-beginners-guide/
[2] IBM. (n.d.). *What Is Deep Learning?*. Retrieved from https://www.ibm.com/think/topics/deep-learning
[3] AWS. (n.d.). *What is a Neural Network?*. Retrieved from https://aws.amazon.com/what-is/neural-network/
[4] IBM. (2024, November 27). *What is Learning Rate in Machine Learning?*. Retrieved from https://www.ibm.com/think/topics/learning-rate
[5] Medium. (2023, July 9). *A (Very Short) Visual Introduction to Learning Rate Schedulers (With Code)*. Retrieved from https://medium.com/@theom/a-very-short-visual-introduction-to-learning-rate-schedulers-with-code-189eddffdb00
[6] Neptune.ai. (n.d.). *How to Choose a Learning Rate Scheduler for Neural Networks*. Retrieved from https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler
[7] life-ai-learning.tistory.com. (2023, February 1). *[EECS 498-007 / 598-005] Lecture 11: Training Neural Network*. Retrieved from https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-11-Training-Neural-Network
[8] Machine Learning Mastery. (2019, August 6). *A Gentle Introduction to Early Stopping to Avoid Overtraining Neural Network Models*. Retrieved from https://www.machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/
[9] GeeksforGeeks. (2025, July 18). *Using Early Stopping to Reduce Overfitting in Neural Networks*. Retrieved from https://www.geeksforgeeks.org/deep-learning/using-early-stopping-to-reduce-overfitting-in-neural-networks/
[10] Your Data Teacher. (2021, May 19). *Hyperparameter tuning. Grid search and random search*. Retrieved from https://www.yourdatateacher.com/2021/05/19/hyperparameter-tuning-grid-search-and-random-search/
[11] KDnuggets. (2023, November 3). *Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV Explained*. Retrieved from https://www.kdnuggets.com/hyperparameter-tuning-gridsearchcv-and-randomizedsearchcv-explained
[12] Simplilearn. (2025, June 9). *The Complete Guide on Overfitting and Underfitting in Machine Learning*. Retrieved from https://www.simplilearn.com/tutorials/machine-learning-tutorial/overfitting-and-underfitting
[13] TensorFlow Core. (2024, April 3). *Overfit and underfit*. Retrieved from https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
[14] life-ai-learning.tistory.com. (2023, February 1). *[EECS 498-007 / 598-005] Lecture 11: Training Neural Network*. Retrieved from https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-11-Training-Neural-Network
[15] DigitalOcean. (2025, April 24). *What is Deep Learning? A Beginner's Guide to Neural Networks*. Retrieved from https://www.digitalocean.com/resources/articles/what-is-deep-learning
[16] life-ai-learning.tistory.com. (2023, February 1). *[EECS 498-007 / 598-005] Lecture 11: Training Neural Network*. Retrieved from https://life-ai-learning.tistory.com/entry/EECS-498-007-598-005-Lecture-11-Training-Neural-Network

