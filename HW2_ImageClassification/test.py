import pickle

root='./data/'

data=pickle.load(open(root+'traintest.pkl','rb'))

print(data)
print(data.keys())