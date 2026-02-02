# OpenCL: Zero to Transformer 🤖

## Objective
To build a deep understanding of Heterogeneous Computing and Machine Learning optimization by implementing a Transformer model from scratch using raw OpenCL.

## 🛠 Phase 1: The OpenCL Foundation
*Goal: Understand the "boilerplate" and the hardware model.*
1.  **Platform & Device Discovery:** Querying the host for available hardware (NVIDIA GPU, CPU, etc.).
2.  **The Context & Command Queue:** Setting up the environment where work happens.
3.  **Memory Management:** * Host (CPU) vs. Device (GPU) memory.
    * Reading/Writing buffers.
    * **Concept:** Global vs. Local (Cache) vs. Private memory.
4.  **The Kernel:** Writing our first parallel function ("Worker").

## 🧮 Phase 2: Parallel Math & Algorithms
*Goal: Learn to "think" in parallel.*
1.  **Vector Addition:** CPU vs. GPU implementation. 
2.  **Parallel Reduction:** Summing arrays (handling concurrency and race conditions).
3.  **Matrix Multiplication (GEMM):** * Naive implementation.
    * Tiled implementation (using `local_cache` for speed).
    * *This is the core engine of all Deep Learning.*

## 🧠 Phase 3: The Neural Building Blocks
*Goal: Translate Math to ML layers.*
1.  **Linear Layers:** Utilizing our Matrix Multiplication.
2.  **Activation Functions:** Implementing ReLU/Softmax as kernels.
3.  **The Backward Pass:** Calculating gradients (calculus in C++/OpenCL).

## 🚀 Phase 4: The Transformer
*Goal: The Final Boss.*
1.  **Self-Attention Mechanism:** Implementing the $Q, K, V$ matrix logic.
2.  **Feed Forward Network:** Connecting layers.
3.  **Training Loop:** Updating weights on the GPU.
4.  **Inference:** Running the model.

## 📱 Phase 5: The Mobile Migration (Qualcomm)
*Goal: Porting to the Edge.*
1.  Adapting code for Snapdragon (Adreno GPU).
2.  Optimization tips for mobile power/thermal constraints.