### Algorithm Implementation Overview

In this project, the final algorithm aimed to predict star ratings for Amazon movie reviews based on a combination of textual and numeric features from the review dataset. The model employed a combination of Natural Language Processing (NLP) techniques for text data and machine learning algorithms to predict the final star ratings. Below, I will describe the steps involved and any techniques used to improve my model's accuracy.

### Key Components of the Algorithm

1. **Data Preprocessing**
    - The dataset was preprocessed by handling missing values, extracting relevant features, and converting them into a format suitable for the model.
    - For text fields such as `reviewText` and `summary`, I used TF-IDF (Term Frequency-Inverse Document Frequency) to represent text as numeric vectors. The TF-IDF vectorizer transformed the unstructured text into a high-dimensional sparse matrix, capturing the importance of words relative to their occurrence in other documents.
    - Sentiment analysis was conducted on both the `reviewText` and `summary` fields, generating sentiment scores. These scores captured the emotional tone of the text, ranging from negative to positive, providing additional features for the model.

2. **Feature Engineering**
    - The dataset was enhanced with user-level features by aggregating information from the training data. Features such as the average rating a user had given (`avg_user_rating`), total number of reviews a user had written (`total_user_reviews`), and total helpfulness votes received (`total_helpfulness_votes`) were calculated for each user.
    - Numeric features were also added, such as `review_age`, which calculated the age of the review based on the difference between the current year and the year the review was written.
    - These user- and review-specific features were essential to the model's accuracy, as they captured patterns in reviewer behavior, which can be strong indicators of star ratings.

3. **Model Training**
    - The core model used for predictions was **XGBoost**, a gradient-boosting algorithm that excels in structured data. XGBoost was chosen for its performance, ability to handle sparse data, and flexibility with a combination of text and numeric features.
    - During the training process, the features were separated into two groups: 
        - Text features (e.g., TF-IDF matrix).
        - Numeric features (e.g., sentiment scores, user-level statistics).
      These were combined into a single sparse matrix using `hstack`, which allowed the model to consider both the textual and numeric data.
    - The model was trained on the combined matrix, where the star rating was the target variable. The star ratings were originally provided on a 1–5 scale, but the target was transformed to a 0–4 scale for training purposes to match XGBoost’s default outputs. After predictions, the values were adjusted back to the 1–5 scale.

4. **Model Evaluation**
    - The predictions were evaluated primarily through measuring Accuracy in addition to a confusion matrix to assess the model's weaknesses. The model's final predictions were output to a CSV file for submission, following the format required.

### Specific Techniques and Optimizations

1. **Sentiment Analysis**
    - One of the most effective techniques used was generating sentiment scores from both the `reviewText` and `summary`. Sentiment can be a strong indicator of the star rating (e.g., a highly positive review tends to correlate with a higher rating). By including these scores as numeric features, the model had additional context that directly related to the target variable.

2. **Feature Aggregation for User-Level Data**
    - Aggregating user-specific data helped capture behavioral patterns. For example, users who tend to write more reviews or give higher average ratings may have a different influence on the final star rating than occasional reviewers. The inclusion of `avg_user_rating`, `total_user_reviews`, and `total_helpfulness_votes` allowed the model to generalize better for specific users.
  
3. **Combining Text and Numeric Features**
    - The combination of text features (from TF-IDF) and numeric features (such as sentiment and user-level stats) into a single feature matrix was key to the model's ability to handle different data types effectively. The `hstack` function allowed us to feed both text and numeric data into XGBoost without compromising the model’s flexibility.

### Assumptions

1. **User Behavior Assumption**
    - It was assumed that user behavior (e.g., average rating given, total number of reviews) remained consistent across different types of reviews. This assumption may not hold for all users, as their reviewing habits may change over time or depending on the genre of the movie, but it provided useful signal for the model.
    
2. **Linear Sentiment Impact**
    - It was assumed that the relationship between sentiment scores and star ratings was approximately linear, meaning that more positive sentiment would directly lead to higher ratings. However, some reviews might express subtle nuances (e.g., sarcasm) that sentiment analysis cannot capture fully.

3. **No Temporal Dynamics**
    - The model did not account for potential changes in user behavior over time. For instance, user tastes might evolve, or certain movie genres may receive different types of reviews depending on trends or movie releases at specific times.

### Mistakes and Possible Improvements

1. **Usage of `nrows` when loading the dataset**
    - When developing initial iterations of the model, smaller batches of training data were loaded in by adjusting the value of the `nrows` parameter for `read_csv()`. A more robust approach would have been to use the `.sample()` method. 
    - Alternative models and pipelines were developed using sampled datasets, yet their performance did not exceed that of this model.

2. **Lack of Version Control**
    - This notebook, despite being the highest achieving model on Kaggle, represents my first submission. Due to this, I did not pay particular attention to keeping track of changes made throughout my attempts to improve the model under the assumption that at least one subsequent submission would improve upon my first.
    - When the competition ended and I failed to make an improvement on my first submission, I had to iteratively backtrace my steps until I could reproduce the same submission file as my first attempt.

3. **Class Balancing, Hyperparameter Tuning, and other Optimizations**
    - In other, less accurate models, I experimented with class balancing to reduce the model's bias towards predicting 5s or 1s excessively. 
    - Hyperparameter tuning through a randomized search was also experimented with, but long run times made the practice infeasible overall.
    - Additional performance optimization methods, such as cross validation, could have been used to improve the robustness of more complex models and improve accuracy.


### Conclusion

The best performing model effectively utilized a combination of natural language processing techniques (TF-IDF, sentiment analysis) and machine learning (XGBoost) to predict star ratings from Amazon movie reviews. Techniques sentiment extraction, user-level feature aggregation, and combining text with numeric data were pivotal in improving model performance. The assumptions regarding user behavior and sentiment scores simplified the model but provided meaningful improvements in accuracy. More complex models, such as those that incorporate class balancing and additional language processing features, failed to reach the level of performance of this relatively simple model. 

### Notes

Due to time constraints and the desire to explore more complex features, this model was never trained on the entire dataset, but only a subset of 500,000 entries (`nrows = 500000`). While attempting to reconstruct this notebook, a submission file was created after training this model on all valid data in `train.csv`, yielding public and private scores of 0.62168 and 0.61981, respectively.