#Nazim Belabbaci aka NazimBL
#Summer 2022

from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_table("data")
#print(gata3_data)

##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
#print(gata3_data)

kmer_texts = list(gata3_data['kmers'])
for item in range(len(kmer_texts)):
    kmer_texts[item] = ' '.join(kmer_texts[item])

y_data = gata3_data.iloc[:, 0].values
#print(human_texts[0])
#print(human_texts[1])

# Creating the Bag of Words model using CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmer_texts)

#print(X)
gata3_data['label'].value_counts().sort_index().plot.bar()
seed = 7
test_size = 0.3
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=seed)

## Hyper Parameter Optimization
params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]

}

best_params={'min_child_weight': 7, 'max_depth': 10, 'learning_rate': 0.25, 'gamma': 0.0, 'colsample_bytree': 0.7}

from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# instantiate the classifier
classifier = XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X,y_data)
timer(start_time) # timing ends here for "start_time" variable

print(random_search.best_estimator_)
print(random_search.best_params_)

#Input best_estimator output to classifier's param

"""
classifier=XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.7,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.0, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.25, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=10, max_leaves=0, min_child_weight=7,
              missing=None, monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,subsample=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y_data,cv=10)
print(score)
"""
