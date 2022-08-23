import pickle
import xgboost as xgb
import pandas as pd


def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#read preprocessed table, see github repository
gata3_data=pd.read_table("gata3_alone_test")

##replace seuence column with kmers words
gata3_data['kmers'] = gata3_data.apply(lambda x: getKmers(x['sequence']), axis=1)
gata3_data = gata3_data.drop('sequence', axis=1)


kmer_texts = list(gata3_data['kmers'])
for item in range(len(kmer_texts)):
    kmer_texts[item] = ' '.join(kmer_texts[item])


loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
X = loaded_vectorizer.transform(kmer_texts)

print(X.shape)
model = xgb.XGBClassifier()
model.load_model("xgb_model_last.json")

fn = open('GATA3alone2.bed.txt', 'r')

# open other file in write mode
fn1 = open('GATA_alone_Predicted', 'w')

count = 0
y_pred = model.predict(X)
print(y_pred)

cont = fn.readlines()
type(cont)
for i,pred in zip(range(0, len(cont)-1) , y_pred):
    if (pred==1):
        fn1.write(cont[i].upper())
    else:pass

# close the file
fn1.close()
