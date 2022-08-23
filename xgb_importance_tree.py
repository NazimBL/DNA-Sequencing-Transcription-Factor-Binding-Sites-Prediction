# Nazim Belabbaci
# Summer 2022
from sklearn.metrics import plot_confusion_matrix
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_table("final")

gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)
#print(gata3_data)

kmer_texts = list(gata3_data['kmers'])
for item in range(len(kmer_texts)):
    kmer_texts[item] = ' '.join(kmer_texts[item])


y_data = gata3_data.iloc[:, 0].values

# Creating the Bag of Words model using CountVectorizer()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(kmer_texts)

gata3_data['label'].value_counts().sort_index().plot.bar()

seed = 7
test_size = 0.3
# Splitting the human dataset into the training set and test set
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_data,
                                                    test_size=test_size,
                                                    random_state=seed)

params={'min_child_weight': 7,
        'max_depth': 10, 'learning_rate': 0.25,
        'gamma': 0.0, 'colsample_bytree': 0.7}

#save this shit
classifier = xgb.XGBClassifier(**params)

classifier.fit(X_train,y_train,verbose=True ,
               early_stopping_rounds=20,eval_metric='aucpr',
               eval_set=[(X_test,y_test)])

plot_confusion_matrix(classifier,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Not Bond", "Bond"])


bst =classifier.get_booster()

for importance_type in ('weight','gain','cover','total_gain','total_cover'):
  print('%s:'% importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape':'box', ##make the nodes fancy
               'style':'filled, rounded',
               'fillcolor':'#78cbe'}

leaf_params = {'shape':'box', ##make the nodes fancy
               'style':'filled',
               'fillcolor':'#e48038'}

graph_data=xgb.to_graphviz(classifier, num_trees=0, size="10,10",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)

graph_data.view(filename='xgb_tree')
plt.show()
