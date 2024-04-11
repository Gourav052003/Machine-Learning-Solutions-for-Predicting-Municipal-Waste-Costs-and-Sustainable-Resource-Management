from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('bin/LGBM_model','rb'))
transformer = pickle.load(open('bin/transformer_file','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_placement():
     
    region = str(request.form.get('region'))
    tc = float(request.form.get('tc'))
    cres = float(request.form.get('cres'))
    csor = float(request.form.get('csor'))
    area = float(request.form.get('area'))
    pop = int(request.form.get('pop'))
    alt = float(request.form.get('alt'))
    sea = float(request.form.get('sea'))
    pden = float(request.form.get('pden'))
    urb = float(request.form.get('urb'))
    d_fee = int(request.form.get('d_fee'))
    sample = int(request.form.get('sample'))
    organic = float(request.form.get('organic'))
    paper = float(request.form.get('paper'))
    glass = float(request.form.get('glass'))
    wood = float(request.form.get('wood'))
    metal = float(request.form.get('metal'))
    plastic = float(request.form.get('plastic'))
    texile = float(request.form.get('texile'))
    wage = float(request.form.get('wage'))
    
    data = pd.DataFrame(data = np.array([region,tc,cres,csor,area,pop,alt,sea,pden,urb,d_fee,sample,organic,paper,glass,wood,metal,plastic,texile,wage]).reshape(1,20),
    columns = ["region","tc","cres","csor","area","pop","alt","sea","pden","urb","d_fee","sample","organic","paper","glass","wood","metal","plastic","texile","wage"])
    
    transformed_data = transformer.transform(data)
    
    # prediction
    result = model.predict(transformed_data)
    
    return render_template('results.html',result=2.718281828459045**result[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050,debug=True)