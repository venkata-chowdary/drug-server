# # def get_sentiment(model, text):
# #     prediction = model.predict([text])
# #     return prediction[0]


# import torch
# from torch_geometric.data import Data
# from rdkit import Chem
# from rdkit.Chem import AllChem

# # Dummy functions (replace with actual feature extraction)
# def featurize_drug(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     # For simplicity, we create dummy features
#     x = torch.randn((mol.GetNumAtoms(), 32))  # node features
#     edge_index = torch.randint(0, mol.GetNumAtoms(), (2, 64))  # random edges
#     return x, edge_index

# def featurize_protein(sequence):
#     # Assume you have a protein embedding model. Here’s a dummy embedding
#     return torch.randn((1, 128))  # fake embedding of the protein

# def get_affinity(model, drug_smiles, protein_sequence):
#     drug_x, edge_index = featurize_drug(drug_smiles)
#     protein_embedding = featurize_protein(protein_sequence)

#     data = Data(
#         x=drug_x,
#         edge_index=edge_index,
#         protein_embedding=protein_embedding,
#         batch=torch.zeros(drug_x.size(0), dtype=torch.long)  # single graph batch
#     )

#     with torch.no_grad():
#         affinity = model(data).item()
#     return affinity


import torch
from torch_geometric.data import Data
from rdkit import Chem

def featurize_drug(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    # Create dummy node features with dimension 5 (to match the saved model)
    x = torch.randn((mol.GetNumAtoms(), 5))
    # Create a dummy edge index tensor (random edges for demonstration)
    edge_index = torch.randint(0, mol.GetNumAtoms(), (2, 64))
    return x, edge_index

def featurize_protein(sequence):
    # Return a dummy protein embedding with dimension 128
    return torch.randn((1, 128))

def get_affinity(model, drug_smiles, protein_sequence):
    # Convert drug and protein to features
    drug_x, edge_index = featurize_drug(drug_smiles)
    protein_embedding = featurize_protein(protein_sequence)
    
    # Create a PyTorch Geometric Data object
    data = Data(
        x=drug_x,
        edge_index=edge_index,
        protein_embedding=protein_embedding,
        batch=torch.zeros(drug_x.size(0), dtype=torch.long)  # single graph batch
    )
    
    with torch.no_grad():
        affinity = model(data).item()
    return affinity
