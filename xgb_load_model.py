import pickle
import xgboost as xgb
import pandas as pd

#kmers function
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

##preprocessing
gata3_data=pd.read_table("gata3_er_test")


##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)

kmers_texts = list(gata3_data['kmers'])
for item in range(len(kmers_texts)):
    kmers_texts[item] = ' '.join(kmers_texts[item])

loaded_vectorizer = pickle.load(open('vectorizer_gataless.pickle', 'rb'))
X = loaded_vectorizer.transform(kmers_texts)

print(X.shape)
model = xgb.XGBClassifier()
model.load_model("xgb_model_gataless.json")

#y_pred=model.predict(X, ntree_limit=model.best_ntree_limit)
#print(y_pred)

count=0
y_pred=model.predict(X)
print(y_pred)
for result in y_pred:
    if(result==1): count = count +1
    print(str(result))
print(count)
