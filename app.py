import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import operator
import itertools 


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
le1=pickle.load(open('le1.pkl','rb'))
le2=pickle.load(open('le2.pkl','rb'))
le3=pickle.load(open('le3.pkl','rb'))
le4=pickle.load(open('le4.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')
	
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #df = pd.DataFrame({"State": [''], "District": [''],"Year":[2020],"Season":[''],"Crop":[''],"Rainfall":[1000]})
    #i=0
    features2=[0,0,0,0,0,0]
    features=[ x for x in request.form.values()]
    features2[0]=le1.transform([features[1]])[0]
    features2[1]=le2.transform([features[2]])[0]
    features2[2]=int(features[3])
    if features[4]=='Autumn':
    	features2[3]=0
    elif features[4]=='Kharif':
    	features2[3]=1
    elif features[4]=='Rabi':
    	features2[3]=2
    elif features[4]=='Summer':
    	features2[3]=3
    elif features[4]=='Winter':
    	features2[3]=5
    else :
    	features2[3]=4
    	
    features2[5]=float(features[5])
    
    df=pd.read_csv('maindata.csv')
    df=df[df['District_Name']==features[2]]
    unique_crops=df['Crop'].unique()
    
    out_list={}
    for crop in unique_crops:
    	features2[4]=le4.transform([crop])[0]
    	final_features = [np.array(features2)]
    	prediction = model.predict(final_features)

    	out_list[crop]=prediction[0]
    
    
    out_list=dict(sorted(out_list.items(),key=operator.itemgetter(1),reverse=True))
    out_list = dict(itertools.islice(out_list.items(), 5))
    op=[]
    for x in out_list.keys():
    	op.append(x)
    return render_template('output.html', len = len(out_list.keys()),prediction_text=op)
    
    #return render_template('index.html', prediction_text='Yield will be {} <br> {} <br> {} <br> {}'.format(le1.classes_,le2.classes_,le3.classes_,le4.classes_))
    

if __name__ == "__main__":
    app.run(debug=True)


