# Book Rating Prediction

A project to predict **Book-Ratings** using user-book interaction data, simulating real-world recommendation systems.

## Objectives
- Analyze rating distributions and user/book activity.
- Preprocess data: handle missing values, encode features, address sparsity.
- Build models: Matrix Factorization, Collaborative Filtering, Neural Networks.
- Evaluate using RMSE and MAE.
- Optimize: tune hyperparameters and handle class imbalances.

## Tech Stack
- **Python**, **Pandas**, **NumPy**
- **Seaborn**, **Matplotlib**
- **Surprise**, **TensorFlow/Keras**
- **Google Colab**

## Usage
1. **Run on Google Colab**:
   - Clone the repo:
     ```bash
     git clone https://github.com/your-repo/book-recommendation-system.git
     ```
   - Upload the notebook to Google Drive and open it in Colab.

2. **Contribute**:
   - Fork the repo, create a feature branch, make changes, and submit a pull request.

## License
Licensed under the [MIT License](LICENSE).

## STATS

### Rating Statistics:
- **Strong zero-inflation**: Many books have 0 ratings.
- **count**: 1.149780e+06
- **mean**: 2.866950e+00
- **std**: 3.854184e+00
- **min**: 0.000000e+00

### User Activity Statistics:
- **Average ratings per user**: 10.92 times
- **Median ratings per user**: 1.0
- **Max ratings by a single user**: 13602 times

### Book Activity Statistics:
- **Average ratings per book**: 3.38 times
- **Median ratings per book**: 1.0
- **Max ratings for a single book**: 2502 times

### Data Preprocessing:
- **No missing values**.
- **Encoded ISBNs to numerical values**.
- **Filtered users with 5+ ratings** and books with 5+ ratings.
- **Normalized ratings** to 0-1 scale.
- **Reduced dataset from 1M to 541K ratings** while maintaining quality interactions.

### Model 1: Collaborative Filtering using SVD (Singular Value Decomposition)
- **Reason for choosing SVD**: It's a widely used matrix factorization technique that effectively handles sparse datasets, capturing latent factors for users and items to predict ratings accurately.
- **Performance**:
  - **RMSE**: 3.54
  - **MAE**: 2.81
- **SVD Errors Observations**:
  - **Peak at 0 Error**: Many near-perfect predictions, which is a good sign.
  - **Negative Skew (Overestimation)**: The model tends to over-predict more often, as observed in the range of -5 to -2.5.
  - **Positive Errors (Underestimation)**: Some significant underestimation between 5 and 10.
  
### After Tuning SVD:
1. **Clustering Along Vertical Lines**:
   - The model struggles to differentiate among books with similar features, indicating that further adjustments are needed.
   - After tuning, the clustering is less pronounced, but there is still room for improvement.
   
2. **Scattering Around the Red Line**:
   - Points scatter around the red line (perfect predictions), indicating prediction errors. A tighter alignment would show higher accuracy, reflecting the effect of optimized hyperparameters.

---

### Model 2: Neural Collaborative Filtering (NCF)
- **Why NCF?**:
  - NCF leverages neural networks to learn complex, non-linear relationships between users and items, making it more accurate and personalized than traditional methods.
  - NCF captures intricate user preferences and can predict ratings that traditional collaborative filtering methods struggle with.
  
- **Performance**:
  - **Accuracy (Exact Match)**: 16.95%
  - **RMSE** and **MAE** for NCF were also calculated.
  - **Challenges**:
    - Exact match accuracy can be low due to the nature of continuous rating scales (1-5).
    - Recommendation systems generally focus on minimizing errors (RMSE, MSE) or ranking metrics like Precision@K and Recall@K.
  
---

### Comparison of SVD and NCF

| Model     | RMSE   | MAE   | Accuracy (Exact Match) |
|-----------|--------|-------|------------------------|
| **SVD**   | 3.54   | 2.81  | N/A                    |
| **NCF**   | 3.91   | 2.95  | 16.95%                 |

- **SVD**: Performs reasonably well with moderate accuracy but is limited by its linear nature in capturing user-item interactions.
- **NCF**: Shows potential for personalized recommendations but has low exact match accuracy. This is common in recommendation systems, where minimizing prediction error is usually prioritized over exact accuracy.

### Conclusion
- Both SVD and NCF have their strengths and weaknesses.
- **SVD** is fast and interpretable, but struggles with capturing complex, non-linear user-item interactions.
- **NCF** can handle these complexities, but exact match accuracy may not always be high due to the continuous nature of ratings.
- **Future Steps**: Consider further tuning of the NCF model, adding additional features, and exploring other recommendation system techniques like **Factorization Machines** or **Deep Learning-based models**.
