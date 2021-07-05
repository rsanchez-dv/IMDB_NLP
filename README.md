# IMDB_NLP
Trained a model to recognize positive or negative comments using the IMDB data set. Experimented detecting positive or negative reviews on Rotten Tomatoes.
# IMDB Assignment

## Importing Libraries


```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

# Just don't want warnings.
import warnings
warnings.filterwarnings('ignore')
```

## Importing Data


```python
df = pd.read_csv('./IMDB_Dataset.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



### Initial Observations in the data


```python
display(df['sentiment'].value_counts())
```


    negative    25000
    positive    25000
    Name: sentiment, dtype: int64



```python
df.shape
```




    (50000, 2)



## Creating pipeline


```python
import string
import re
import nltk
stopword = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [wn.lemmatize(word) for word in tokens if word not in stopword]
    return text

pipe = Pipeline([
    ('cvec',TfidfVectorizer()),
    ('lr',LogisticRegression())
])
# Source: https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning/
pipe2 = Pipeline([
    ('cvec2',TfidfVectorizer()),
    ('multi_nb',MultinomialNB())
])
pipe3 = Pipeline([
    ('cvec3', CountVectorizer()),
    ('bern_nb',BernoulliNB())
])
pipe4 = Pipeline([
    ('cvec4', CountVectorizer()),
    ('rand_forest',RandomForestClassifier())
])
pipe_params = {
    'cvec__max_features': [2500,3000,3500],
    'cvec__min_df': [2,3],
    'cvec__ngram_range': [(1,1),(1,2)]
}
```

## Split Data


```python
X = df['review']
y = df['sentiment']
X.head()
y.head()
```




    0    positive
    1    positive
    2    positive
    3    negative
    4    positive
    Name: sentiment, dtype: object




```python
y = y.apply(lambda x: 1 if x == 'positive' else 0 )
y
```




    0        1
    1        1
    2        1
    3        0
    4        1
            ..
    49995    1
    49996    0
    49997    0
    49998    0
    49999    0
    Name: sentiment, Length: 50000, dtype: int64




```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state=42)
```

## Train Models

### Logistric Regression With GridSearch


```python
gs = GridSearchCV(pipe, param_grid = pipe_params, cv = 3, n_jobs=-1)
start = time.time()
gs.fit(X_train,y_train)
print(f'Time: {time.time() - start}')
print(f'Training Score: {gs.score(X_train,y_train)}')
print(f'Training Score: {gs.score(X_test,y_test)}')
print(classification_report(y_test,gs.predict(X_test)))
```

    Time: 77.85983371734619
    Training Score: 0.9086
    Training Score: 0.8912
                  precision    recall  f1-score   support
    
               0       0.90      0.88      0.89      7411
               1       0.89      0.90      0.89      7589
    
        accuracy                           0.89     15000
       macro avg       0.89      0.89      0.89     15000
    weighted avg       0.89      0.89      0.89     15000



### Multinomial NB 


```python
start = time.time()
pipe2.fit(X_train,y_train)
print(f'Time: {time.time() - start}')
print(f'Training Score: {pipe2.score(X_train,y_train)}')
print(f'Training Score: {pipe2.score(X_test,y_test)}')
print(classification_report(y_test,pipe2.predict(X_test)))
```

    Time: 4.6989686489105225
    Training Score: 0.9037714285714286
    Training Score: 0.8608666666666667
                  precision    recall  f1-score   support
    
               0       0.84      0.89      0.86      7411
               1       0.89      0.83      0.86      7589
    
        accuracy                           0.86     15000
       macro avg       0.86      0.86      0.86     15000
    weighted avg       0.86      0.86      0.86     15000



### Bernoulli NB


```python
start = time.time()
pipe3.fit(X_train,y_train)
print(f'Time: {time.time() - start}')
print(f'Training Score: {pipe3.score(X_train,y_train)}')
print(f'Training Score: {pipe3.score(X_test,y_test)}')
print(classification_report(y_test,pipe3.predict(X_test)))
```

    Time: 4.696046829223633
    Training Score: 0.9016285714285714
    Training Score: 0.8561333333333333
                  precision    recall  f1-score   support
    
               0       0.84      0.88      0.86      7411
               1       0.88      0.83      0.85      7589
    
        accuracy                           0.86     15000
       macro avg       0.86      0.86      0.86     15000
    weighted avg       0.86      0.86      0.86     15000



### Random Forest Classifier


```python
start = time.time()
pipe4.fit(X_train,y_train)
print(f'Time: {time.time() - start}')
print(f'Training Score: {pipe4.score(X_train,y_train)}')
print(f'Training Score: {pipe4.score(X_test,y_test)}')
print(classification_report(y_test,pipe4.predict(X_test)))
```

    Time: 82.92627286911011
    Training Score: 1.0
    Training Score: 0.8515333333333334
                  precision    recall  f1-score   support
    
               0       0.85      0.85      0.85      7411
               1       0.85      0.85      0.85      7589
    
        accuracy                           0.85     15000
       macro avg       0.85      0.85      0.85     15000
    weighted avg       0.85      0.85      0.85     15000



## Save Model


```python
import pickle
with open('IMDB_model.pkl','wb') as file:
    pickle.dump(gs,file)
```

## Load Model


```python
with open('IMDB_model.pkl', 'rb') as file:
    IMDB_model = pickle.load(file)
```

## Testing Model


```python
reviews = [
    # NEG
    'Absolute trash, dont spend you money on this...',
    'A great survival-on-an-island movie. Tom Hanks is superb. A sad story, but one that most people will like.',
    # NEG
    'No one asked for Mary Poppinss return to modern consciousness, but her reappearance unmistakably proves that Hollywood Boomers are desperate to justify their own mediocrity through nostalgic sentiment',
    '"Cast Away" is an exceptionally well-crafted exploration of the survival of the human spirit. Its a movie unafraid to consider the full complexity of life.',
    'Somewhat entertaining especially with a lot of the unintended comedy. At times very tedious and the main concept of the film was completely lost.',
    # NEG
    'The film shows shallow -- fake -- empathy with the Appalachian background that begins Vances humble brag about leaving backwoods hollers and winding up at Yale University',
    'A great movie that shows the progress of human development through Tom Hanks character while he is stranded on the desert island. But...all that is overshadowed by Wilson, who will remain in our hearts for all eternity.',
    # NEG
    'Trash like Red Sparrow, the Jennifer Lawrence spy movie, represents the garbagey essence of most Hollywood movies',
    'Probably one of the best disaster emotional films ever. A classic game of survival that is played absolutely perfectly.',
    # NEG
    'The film disastrously focuses on Udays outrages and does so without any moral perspective. "Rape, torture, disembowelment, killing, drinking, drugs and decadence" is practically the films synopsis',
    'While not a new idea to anyone plot-wise, a brilliantly executed film with lots of great themes/moments to enjoy, mostly just with one man!',
]
```


```python
print(f'Grid Search Logistic Regression')
display(gs.predict(reviews))
```

    Grid Search Logistic Regression



    array([0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=int64)



```python
print(f'MultinomialNB:')
display(pipe2.predict(reviews))
```

    MultinomialNB:



    array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=int64)



```python
print(f'BernoulliNB')
display(pipe3.predict(reviews))
```

    BernoulliNB



    array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=int64)



```python
print(f'RandomForestClassifier:')
display(pipe4.predict(reviews))
```

    RandomForestClassifier:



    array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int64)



```python
print(f'Actual: [0,1,0,1,1,0,1,0,1,0,1]')
```

    Actual: [0,1,0,1,1,0,1,0,1,0,1]



```python

```


