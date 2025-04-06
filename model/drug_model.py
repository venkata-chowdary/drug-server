# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv, global_mean_pool

# class DrugProteinModel(nn.Module):
#     def __init__(self, drug_in_channels, protein_embedding_size, hidden_channels, out_channels):
#         super(DrugProteinModel, self).__init__()

#         # GraphSAGE convolution layer for drug features
#         self.drug_conv = SAGEConv(drug_in_channels, hidden_channels)

#         # Fully connected layer for protein embedding
#         self.protein_lin = nn.Linear(protein_embedding_size, hidden_channels)

#         # Combining drug and protein features
#         self.interaction_lin = nn.Linear(hidden_channels * 2, hidden_channels)

#         # Output prediction layer
#         self.lin = nn.Linear(hidden_channels, out_channels)

#     def forward(self, data):
#         # Drug graph processing
#         drug_x = self.drug_conv(data.x, data.edge_index)
#         drug_x = F.relu(drug_x)
#         drug_x = global_mean_pool(drug_x, data.batch)

#         # Protein embedding processing
#         protein_x = self.protein_lin(data.protein_embedding.squeeze(0))

#         # Combine both
#         combined_x = torch.cat((drug_x, protein_x), dim=1)
#         interaction_x = F.relu(self.interaction_lin(combined_x))

#         # Final prediction
#         return self.lin(interaction_x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class DrugProteinModel(nn.Module):
    def __init__(self, drug_in_channels, protein_embedding_size, hidden_channels, out_channels):
        super(DrugProteinModel, self).__init__()
        # GraphSAGE convolution layer for drug features
        self.drug_conv = SAGEConv(drug_in_channels, hidden_channels)
        
        # Fully connected layer for protein embedding
        self.protein_lin = nn.Linear(protein_embedding_size, hidden_channels)
        
        # Combining drug and protein features
        self.interaction_lin = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Output prediction layer
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        # Process drug graph data
        drug_x = self.drug_conv(data.x, data.edge_index)
        drug_x = F.relu(drug_x)
        drug_x = global_mean_pool(drug_x, data.batch)
        
        # Process protein embedding (do not squeeze to preserve 2D shape)
        protein_x = self.protein_lin(data.protein_embedding)
        
        # Combine drug and protein features
        combined_x = torch.cat((drug_x, protein_x), dim=1)
        interaction_x = F.relu(self.interaction_lin(combined_x))
        
        # Final prediction
        return self.lin(interaction_x)
