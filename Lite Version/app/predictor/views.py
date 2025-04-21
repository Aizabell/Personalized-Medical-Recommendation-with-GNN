from django.shortcuts import render
import torch
import joblib
import json , os
import numpy as np
from .model import DeepMultiLabelNet

# Load model and assets once
model = DeepMultiLabelNet(472, 4525)
model.load_state_dict(torch.load("predictor/model/best_f1_model.pth", map_location=torch.device('cpu')))
model.eval()

with open("predictor/model/column_index.json") as f:
    column_index = json.load(f)
with open("predictor/model/admission_type_mapping.json") as f:
    admission_map = json.load(f)
with open("predictor/model/ethnicity_mapping.json") as f:
    ethnicity_map = json.load(f)
with open("predictor/model/disease_category_mapping.json") as f:
    disease_map = json.load(f)
with open("predictor/model/procedure_category_mapping.json") as f:
    procedure_map = json.load(f)
    
with open("predictor/model/labtests.json") as f:
    lab_tests = json.load(f)

scaler = joblib.load("predictor/model/scaler_valuenum_labevents.pkl")
mlb = joblib.load("predictor/model/mlb_encoder.pkl")

def predict_view(request):
    if request.method == 'POST':
        # Get form data
        age = float(request.POST['age'])
        gender = int(request.POST['gender'])
        ethnicity = ethnicity_map[request.POST['ethnicity']]
        admission_type = admission_map[request.POST['admission_type']]
        diseases = request.POST.getlist('diseases')
        procedures = request.POST.getlist('procedures')
        lab_vals = [float(val) for val in request.POST.getlist('lab_vals') if val.strip() != '']
        length_of_stay = float(request.POST['length_of_stay'])

        # Create input vector
        input_vector = np.zeros(len(column_index))
        input_vector[column_index['age']] = age
        input_vector[column_index['length_of_stay']] = length_of_stay
        input_vector[column_index['gender']] = gender
        input_vector[column_index['ethnicity']] = ethnicity
        input_vector[column_index['admission_type']] = admission_type
        input_vector[column_index['avg_lab_value']] = scaler.transform([[np.mean(lab_vals)]])[0][0]

        for d in diseases:
            if d in disease_map:
                col = f'disease_category_encoded_{disease_map[d]}'
                if col in column_index:
                    input_vector[column_index[col]] += 1

        for p in procedures:
            if p in procedure_map:
                col = f'procedure_category_encoded_{procedure_map[p]}'
                if col in column_index:
                    input_vector[column_index[col]] += 1

        # Predict
        X_input = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            preds = model(X_input).numpy()[0]

        top_indices = preds.argsort()[-10:][::-1]
        top_results = [
            {"drug": mlb.classes_[i], "prob": round(preds[i] * 100, 2)}  # probability as %
            for i in top_indices
        ]

        return render(request, 'predictor/predict.html', {
            'top_results': top_results,
            'ethnicity_choices': list(ethnicity_map.keys()),
            'admission_type_choices': list(admission_map.keys()),
            'disease_choices': list(disease_map.keys()),
            'procedure_choices': list(procedure_map.keys()),
            'labtest_choices': lab_tests
        })

    # GET request
    return render(request, 'predictor/predict.html', {
        'ethnicity_choices': list(ethnicity_map.keys()),
        'admission_type_choices': list(admission_map.keys()),
        'disease_choices': list(disease_map.keys()),
        'procedure_choices': list(procedure_map.keys()),
        'labtest_choices': lab_tests
    })

def home_view(request):
    return render(request, 'predictor/home.html')