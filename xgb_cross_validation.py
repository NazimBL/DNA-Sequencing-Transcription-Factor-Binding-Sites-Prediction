#Nazim Belabbaci
#Summer 2022

import xgboost as xgb
import pandas as pd
from numpy import  nan


def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_table("data")
#print(gata3_data)

##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
print(gata3_data)

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



best_params={'min_child_weight': 7, 'max_depth': 10, 'learning_rate': 0.25, 'gamma': 0.0, 'colsample_bytree': 0.7}

classifier=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5,
              enable_categorical=False, gamma=0.3, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=8,
              min_child_weight=7, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=36, num_parallel_tree=1,
              predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, subsample=1, tree_method='exact',
              validate_parameters=1, verbosity=None)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y_data,cv=10)
print(score.mean())
