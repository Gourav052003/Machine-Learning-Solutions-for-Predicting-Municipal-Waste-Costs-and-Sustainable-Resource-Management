from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('bin/LGBM_model','rb'))
transformer = pickle.load(open('bin/transformer_file','rb'))
WASTE_GENERATION_df = pd.read_csv("data/cleaned_data.csv")

numerical_features = ["tc","cres","csor","area","pop","alt","pden","organic","paper","glass","wood","metal","plastic","texile","wage"]
categorical_features = ["region","sea","urb","d_fee","sample"]

data_types = ["str","float","float","float","float","int","float","float","float","float","int","int","float","float","float","float","float","float","float","float"]    

def random_sample_replacement(data_row):
    
    for feature in numerical_features:
        
        if data_row[feature].values[0] == "":
            
            fill_value = WASTE_GENERATION_df[feature].dropna().sample(1).values[0]
            data_row[feature] = str(fill_value)
            
    return data_row  


def outlier_treatment(data_row):
    
    
    for feature in numerical_features:
        
        q1 = np.percentile(WASTE_GENERATION_df[feature],25)
        q3 = np.percentile(WASTE_GENERATION_df[feature],75)
        
        iqr = q3-q1
        
        l = q1-(1.5*iqr)
        u = q3+(1.5*iqr)
        
    
        median = int(np.median(WASTE_GENERATION_df[feature].values))
        mean = int(np.mean(WASTE_GENERATION_df[feature].values))
        population = None
        
        if mean>median:
            population = WASTE_GENERATION_df[(WASTE_GENERATION_df[feature]>=np.percentile(WASTE_GENERATION_df[feature],20)) &
                                             (WASTE_GENERATION_df[feature]<=np.percentile(WASTE_GENERATION_df[feature],45))]

        else:
            population = WASTE_GENERATION_df[(WASTE_GENERATION_df[feature]>=np.percentile(WASTE_GENERATION_df[feature],60)) & 
                                             (WASTE_GENERATION_df[feature]<=np.percentile(WASTE_GENERATION_df[feature],85))]

        # print(population)
        data_value = float(data_row[feature].values[0])
               
        if  data_value<l or data_value>u:
            data_row[feature] = population[feature].sample(1).values[0]
        
                  
    return data_row


def data_type_parsing(data_row):
    
    features = data_row.columns
    
    for data_type,feature in zip(data_types,features):
       
        if data_type == "float":
            data_row[feature] = data_row[feature].astype(float)
        elif data_type == "int": 
            data_row[feature] = data_row[feature].astype(int)  
    
    return data_row

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict_placement():
     
    region = str(request.form.get('region'))
        
    tc = request.form.get('tc')
    
    cres = request.form.get('cres')
    csor = request.form.get('csor')
    area = request.form.get('area')
    pop = request.form.get('pop')
    alt = request.form.get('alt')
    
    sea = request.form.get('sea')
    
    pden = request.form.get('pden')
    
    urb = request.form.get('urb')
    d_fee = request.form.get('d_fee')
    sample = request.form.get('sample')
    
    organic = request.form.get('organic')
    paper = request.form.get('paper')
    glass = request.form.get('glass')
    wood = request.form.get('wood')
    metal = request.form.get('metal')
    plastic = request.form.get('plastic')
    texile = request.form.get('texile')
    wage = request.form.get('wage')
    
    
    
    data = pd.DataFrame(data = np.array([region,tc,cres,csor,area,pop,alt,sea,pden,urb,d_fee,sample,organic,paper,glass,wood,metal,plastic,texile,wage]).reshape(1,20),
    columns = ["region","tc","cres","csor","area","pop","alt","sea","pden","urb","d_fee","sample","organic","paper","glass","wood","metal","plastic","texile","wage"])
    
    
    data = random_sample_replacement(data)
    data = data_type_parsing(data)
    data = outlier_treatment(data)
    
    transformed_data = transformer.transform(data)
    
    # prediction
    result = model.predict(transformed_data)
    result = 2.718281828459045**result[0]
    result = round(result,2)
    return render_template('results.html',result=result)

@app.route('/research')
def research():
    return render_template('research.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050,debug=True)

