# NLP-Sentiment-Analyze
### Purpose:
To analyze the commentary of NBA teams, judging which comments belong to which teams and whether each comment is positive, negative, or neutral to the team, in order to investigating supportiveness of a team and the trend that the public supports.

### Method:
- **Tokenizer:**
    Using tf-idf tokenizer and transformer pre-trained tokenizer to make raw text into word vector.
- **Models:**
    - **Base Line:**
        Use SVM, Random-forest, Naive Bayes
    - **Main Approach:** 
        Transformers Bert-base-chinese
### Performance:
Our results are in the performance directory, each image contains f1-score, precision, and recall score.
- **Naming rules**
    - **teams, sent**
        Teams classification and Sentiment analysis tasks
    - **aug, unaug**
        The results of models using augmented data or unaugmented data
    - **new_data**
        The results of input new datas

### Utilize this project:
    1. git clone
    2. pip install requiremetns.txt (python 3.7)
    3. python transformer.py/svm.py/random_forest.py/naive_bayes.py
    