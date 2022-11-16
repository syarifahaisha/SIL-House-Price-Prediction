from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)


def load_model():
    model_from_pickle = pickle.load(open('./model.pkl', 'rb'))
    # print(classifier_from_pickle)
    return model_from_pickle

# def preprocess_input(input_dict:dict):
#     df = pd.read_csv("../dataset/healthcare-dataset-stroke-data.csv")

#     #Remove every row that contains NaN value
#     df = df.dropna()

#     #Remove unecessary column
#     df = df.drop('id', axis=1)
#     df = df.drop('stroke', axis=1)

#     df_input = pd.DataFrame.from_dict([input_dict], orient='columns')
    
#     df = pd.concat([df, df_input])

#     le = preprocessing.LabelEncoder()
#     df['gender'] = le.fit_transform(df['gender'])
#     df['ever_married'] = le.fit_transform(df['ever_married'])
#     df['work_type'] = le.fit_transform(df['work_type'])
#     df['Residence_type'] = le.fit_transform(df['Residence_type'])
#     df['smoking_status'] = le.fit_transform(df['smoking_status'])

    return df.tail(1).values.tolist()

def convert_output(class_pred, confidence_val):
    print("Class", class_pred)
    conv_class_pred = "Stroke" if (class_pred == 1) else "Not Stroke"
    conv_confidence_val = "{0:.2f}%".format(confidence_val * 100)

    return conv_class_pred, conv_confidence_val

def predict(model, input):
    print(input)
    y_pred = model.predict(input)

    pred, confidence_val = convert_output(y_pred[0], model.predict_proba(input).max())

    return pred, confidence_val

@app.route('/', methods=('GET', 'POST'))
def index():
    if (request.method == 'POST'):

        input_raw = {}
        input_raw["date"] = request.values.get('date')
        input_raw["bedrooms"] = request.values.get('bedrooms')
        input_raw["bathrooms"] = request.values.get('bathrooms')
        input_raw["sqft_living"] = request.values.get('sqft_living')
        input_raw["sqft_lot"] = request.values.get('sqft_lot')
        input_raw["floors"] = request.values.get('floors')
        input_raw["waterfront"] = request.values.get('waterfront')
        input_raw["view"] = request.values.get('view')
        input_raw["condition"] = request.values.get('condition')
        input_raw["grade"] = request.values.get('grade')
        input_raw["sqft_above"] = request.values.get('sqft_above')
        input_raw["sqft_basement"] = request.values.get('sqft_basement')
        input_raw["yr_built"] = request.values.get('yr_built')
        input_raw["yr_renovated"] = request.values.get('yr_renovated')
        input_raw["zipcode"] = request.values.get('zipcode')
        input_raw["lat"] = request.values.get('lat')
        input_raw["long"] = request.values.get('long')
        input_raw["sqft_living15"] = request.values.get('bsqft_living15mi')
        input_raw["sqft_lot15"] = request.values.get('sqft_lot15')


        model = load_model()

        predicted_class, confidence = predict(model, input)

        return render_template("index.html", predicted_class=predicted_class, confidence=confidence)
    else:
        return render_template("index.html", predicted_class=None, confidence=None)
