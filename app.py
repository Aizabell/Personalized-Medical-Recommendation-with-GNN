import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flask import Flask, request, render_template, jsonify
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

app = Flask(__name__)

# ---------------------------
# Load Data, Mappings, and Model
# ---------------------------

# Set device (adjust if you are not using MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the graph data
data = torch.load("./models/processed_graph.pt", weights_only=False)
data = data.to(device)

# Build temporary output dims for model instantiation
with torch.no_grad():
    temp_conv1 = HeteroConv({
        edge_type: SAGEConv((-1, -1), out_channels=64)
        for edge_type in data.edge_types
    }, aggr='sum').to(device)
    temp_output = temp_conv1(data.x_dict, data.edge_index_dict)
    output_dims = {node_type: feat.shape[1] for node_type, feat in temp_output.items()}

# Define your model class
class HeteroSAGELinkPredictor(nn.Module):
    def __init__(self, metadata, output_dims, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((hidden_channels, hidden_channels), out_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.patient_encoder = nn.Sequential(
            nn.Linear(44, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_channels * 5, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

    def encode_patient(self, patient_features):
        return self.patient_encoder(patient_features)

    def decode(
        self,
        z_patient,
        z_medication,
        z_disease,
        z_procedure,
        z_lab,
        edge_index,
        disease_ids=None,
        procedure_ids=None,
        lab_ids=None,
    ):
        src, dst = edge_index

        # Get context embeddings or return zero vectors if missing
        def get_context_embeddings(z, ids):
            if ids is None:
                return torch.zeros_like(z_patient[src])
            return z[ids]

        disease_emb = get_context_embeddings(z_disease, disease_ids)
        proc_emb    = get_context_embeddings(z_procedure, procedure_ids)
        lab_emb     = get_context_embeddings(z_lab, lab_ids)

        combined = torch.cat([
            z_patient[src],
            z_medication[dst],
            disease_emb,
            proc_emb,
            lab_emb
        ], dim=1)

        return self.decoder(combined).squeeze()

# Instantiate model and load weights
model = HeteroSAGELinkPredictor(data.metadata(), output_dims=output_dims).to(device)
ckpt_path = "./checkpoints"
model.load_state_dict(torch.load(os.path.join(ckpt_path, "best_model_acc.pt"), map_location=device))
model.eval()

# Load mappings (medication names, disease mapping, etc.)
with open(os.path.join("mappings", "id_to_medication.json"), 'r') as file:
    med_map = json.load(file)

with open(os.path.join("mappings", "id_to_disease.json"), "r") as f:
    disease_mapping = json.load(f)

with open(os.path.join("mappings", "id_to_procedure.json"), "r") as f:
    procedure_mapping = json.load(f)

# Load patient info encoders/scalers
encoder = joblib.load(os.path.join("models", "patient_gender_ethnicity_encoder.pkl"))
age_scaler = joblib.load(os.path.join("models", "patient_age_scaler.pkl"))

# ---------------------------
# Helper functions from your inference code
# ---------------------------

def build_first_association_map(src_nodes, dst_nodes):
    assoc_map = {}
    for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
        if src not in assoc_map:
            assoc_map[src] = dst
    return assoc_map

def get_first_associated_node(assoc_map, patient_ids, default_val=-1):
    return torch.tensor(
        [assoc_map.get(pid.item(), default_val) for pid in patient_ids],
        dtype=torch.long
    ).to(patient_ids.device)

def predict_with_node_addition(
    model,
    data,
    new_patient_features,     # shape: [1, input_dim]
    disease_ids,              # list or tensor of disease node indices
    procedure_ids,            # list or tensor of procedure node indices
    lab_ids,                  # list or tensor of lab node indices
    disease_map_fn,           # function to build disease_map
    procedure_map_fn,         # function to build procedure_map
    lab_map_fn,               # function to build lab_map
    med_map,                  # medication mapping (id -> name)
    device,
    topk=15
):
    model.eval()
    with torch.no_grad():
        # Clone original data so as not to modify the graph
        inference_data = data.clone()

        # Assign new patient node ID and add new patient features
        new_patient_tensor = torch.tensor(new_patient_features, dtype=torch.float).to(device)
        new_patient_id = inference_data["patient"].x.shape[0]
        inference_data["patient"].x = torch.cat(
            [inference_data["patient"].x, new_patient_tensor], dim=0
        )

        # Add edges from the new patient to context nodes
        def add_edges(edge_type, target_ids):
            edge_index = torch.stack([
                torch.full((len(target_ids),), new_patient_id, dtype=torch.long),  # source
                torch.tensor(target_ids, dtype=torch.long)
            ], dim=0).to(device)
            inference_data[edge_type].edge_index = torch.cat([
                inference_data[edge_type].edge_index.to(device),
                edge_index
            ], dim=1)

        add_edges(("patient", "has_disease", "disease"), disease_ids)
        add_edges(("patient", "underwent", "procedure"), procedure_ids)
        add_edges(("patient", "has_lab", "lab"), lab_ids)

        # Reapply ToUndirected to recreate reverse edges
        inference_data = ToUndirected()(inference_data)

        # Rebuild context maps
        disease_map = disease_map_fn(
            inference_data["patient", "has_disease", "disease"].edge_index[0],
            inference_data["patient", "has_disease", "disease"].edge_index[1]
        )
        procedure_map = procedure_map_fn(
            inference_data["patient", "underwent", "procedure"].edge_index[0],
            inference_data["patient", "underwent", "procedure"].edge_index[1]
        )
        lab_map = lab_map_fn(
            inference_data["patient", "has_lab", "lab"].edge_index[0],
            inference_data["patient", "has_lab", "lab"].edge_index[1]
        )

        z_dict = model(inference_data.x_dict, inference_data.edge_index_dict)
        # Replace patient embeddings with encoded features
        patient_features = inference_data["patient"].x.to(device)
        z_dict["patient"] = model.encode_patient(patient_features)

        num_meds = z_dict["medication"].shape[0]
        med_ids = torch.arange(num_meds).to(device)
        patients = torch.full((num_meds,), new_patient_id, dtype=torch.long).to(device)
        edge_index = torch.stack([patients, med_ids], dim=0)

        disease_ids_batch   = get_first_associated_node(disease_map, patients)
        procedure_ids_batch = get_first_associated_node(procedure_map, patients)
        lab_ids_batch       = get_first_associated_node(lab_map, patients)

        scores = model.decode(
            z_patient     = z_dict["patient"],
            z_medication  = z_dict["medication"],
            z_disease     = z_dict["disease"],
            z_procedure   = z_dict["procedure"],
            z_lab         = z_dict["lab"],
            edge_index    = edge_index,
            disease_ids   = disease_ids_batch,
            procedure_ids = procedure_ids_batch,
            lab_ids       = lab_ids_batch
        )

        probs = torch.sigmoid(scores)
        top_indices = probs.topk(topk).indices
        recommended_med_ids = med_ids[top_indices].cpu().numpy()
        top_scores = probs[top_indices].cpu().numpy()

        results = []
        for med_id, score in zip(recommended_med_ids, top_scores):
            results.append({
                "medication": med_map[str(med_id)],
                "score": float(score)
            })

        return results

# ---------------------------
# Flask Routes
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Parse input from form; adjust the names to what your form supplies.
        try:
            age = float(request.form.get("age"))
            gender = int(request.form.get("gender"))
            ethnicity = int(request.form.get("ethnicity"))
        except (TypeError, ValueError):
            return render_template("index.html", error="Invalid input. Please enter valid numbers.")

        # Prepare new patient features.
        # Assumption: patient feature layout is [scaled_age, onehot_gender, onehot_ethnicity, ...]
        # Here we use the scaler and encoder you already trained.
        new_patient_age = np.array([[age]])
        new_patient_demo = np.array([[gender, ethnicity]])
        encoded_demo = encoder.transform(new_patient_demo)
        scaled_age = age_scaler.transform(new_patient_age)
        new_patient_features = np.hstack([scaled_age, encoded_demo])
        
        # For the context associations, you may want to collect additional patient-specific data.
        # Here, we simply pick the first associated nodes from existing edges.
        edge_pd = data["patient", "has_disease", "disease"].edge_index
        edge_pp = data["patient", "underwent", "procedure"].edge_index
        edge_pl = data["patient", "has_lab", "lab"].edge_index

        disease_ids = build_first_association_map(edge_pd[0], edge_pd[1])
        procedure_ids = build_first_association_map(edge_pp[0], edge_pp[1])
        lab_ids = build_first_association_map(edge_pl[0], edge_pl[1])

        # Run inference using the predict_with_node_addition function
        results = predict_with_node_addition(
            model=model,
            data=data,
            new_patient_features=new_patient_features,
            disease_ids=list(disease_ids.values()),
            procedure_ids=list(procedure_ids.values()),
            lab_ids=list(lab_ids.values()),
            disease_map_fn=build_first_association_map,
            procedure_map_fn=build_first_association_map,
            lab_map_fn=build_first_association_map,
            med_map=med_map,
            device=device,
            topk=15
        )
        return render_template("index.html", results=results, age=age, gender=gender, ethnicity=ethnicity)
    return render_template("index.html")

if __name__ == "__main__":
    # When deploying on a server, adjust debug and host settings accordingly.
    app.run(debug=True)
