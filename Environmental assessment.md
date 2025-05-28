##  Assessment of the Environmental Cost

Deep learning projects, especially those involving large datasets and complex models, can have an environmental impact primarily due to the electricity consumed during model training and experimentation. It's important to be mindful of this.

### 🔍 Key Factors Contributing to Environmental Cost:

#### 1. Computational Resources (Training)
- Training deep learning models like **`wide_resnet50_2`** requires significant GPU or CPU processing time. This is the largest contributor.
- The energy consumption depends on the hardware used (GPUs are power-intensive), the duration of training, and the efficiency of the code.
- For this project, training for 10 epochs on the EuroSAT dataset (27,000 images of 64x64, upscaled to 224x224) using a platform like Google Colab still consumes energy.

#### 2. Data Storage and Transfer
- The EuroSAT dataset (RGB version) is a few gigabytes in size. Storing this data on local drives or cloud storage consumes energy continuously, albeit less than active computation.
- Downloading the dataset from its source also consumes energy through network infrastructure.

#### 3. Hyperparameter Tuning
- Multiple training runs with different hyperparameters (learning rates, batch sizes, model variations) multiply the computational cost.

---

### ✅ Mitigation Strategies

- **Transfer Learning**: Using a pre-trained model (**`wide_resnet50_2`**) significantly reduces the required training time and data compared to training a large model from scratch. This is a major energy-saving practice.

- **Efficient Data Loading**: Using `num_workers` in `DataLoader` can speed up data loading, potentially reducing overall training time if I/O is a bottleneck.

- **Early Stopping**: The implemented early stopping mechanism prevents unnecessary training epochs if the model's performance on the validation set ceases to improve, saving computational resources.

- **Appropriate Batch Size**: Choosing a reasonable batch size helps balance memory usage and training speed.
