import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        # Create learnable weight matrix to trnsform input traffic features
        # nn.parameter tells pytorch to update these numbers during training
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Initialize weights using Xavier initialization
        # This scales the initial random weights based on the input/output size, preventing
        # gradients form explodijng ot infinity or vanishing to zero early in training
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # Transform raw traffic input data using learnable weights
        # This projects the raw traffic features into a mathematical space
        support = torch.matmul(x, self.weight)

        # Multiply by the spatial adjacency matrix
        # The adj matrix makes the actual road network. By multiplying it against the features,
        # this forces data from one sensor to blend with its physically connecte dneighbors,
        # teaching the model about spatial bottlenecks
        output = torch.matmul(adj, support)
        return output

class STGNNModel(nn.Module):
    def __init__(self, num_sensors, in_features=1, hidden_dim=14, horizon_steps=12):
        super(STGNNModel, self).__init__()
        self.num_sensors = num_sensors
        # Initalize the spatial Graph Convolutional layer
        self.gcn = GraphConvLayer(in_features, hidden_dim)
        # Initialize a Gataed Recurrent Unit (GRU) to process the chronological flow of time
        # GRUs use internal gating mechanisms to remember important past traffic patterns
        # and forget irrelevant ones, making htem ideal for time-series forecasting
        # bath_first=True: Tells pytorch the data shape starts with the batch size
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        # Fully connected layer to map the GRU's final summary to the 12 future predictions
        self.fc = nn.Linear(hidden_dim, horizon_steps)

    def forward(self, x, adj):
        # x shape: (batch_size, time_steps, num_sensors)
        batch_size, time_steps, _ = x.shape

        #List to hold the spatially processed data for each time step
        gcn_outputs = []
        # Loop through each historicla time step one by 1 (12 steps total, representing 1 hour)
        for t in range(time_steps):
            # Isolate the current time step and add a feature dimension at the end
            step_x = x[:, t, :].unsqueeze(-1)

            # Pass this time step through the spatial graph and apply a ReLU activation function
            # ReLU (Rectified Linear Unit) turns negative numbers to zero, introducing non linearity
            # so the model can learn complex, non linear traffic patterns instead of just straight lines
            step_out = F.relu(self.gcn(step_x, adj))

            # Append the spatially processed step to the list
            gcn_outputs.append(step_out)

        # Stack the 12 spatially processed time steps back into a single tensor along the tim dimension
        x_spatial = torch.stack(gcn_outputs, dim=1)

        # Reshape the tensor to push the sequence of time through the GRU
        # This flatteens the batch and sensor dimensions together so the GRU processes each
        # individual sensor's chronological timeline independently
        x_temporal = x_spatial.view(batch_size * self.num_sensors, time_steps, -1)

        # Pass the sequentual data through the GRU to learn chronological traffic patterns 
        gru_out, _ = self.gru(x_temporal)

        # Extract only the final hidden state from the GRU (index -1)
        # This final state represents the model's overall learned summary aka the context vector
        # of the entire past hour of traffic for that sensor
        last_hidden_state = gru_out[:, -1, :]

        # Pass the temporal summary through the final linear layer to predict the future 12 time steps
        predictions = self.fc(last_hidden_state)

        # Reshape the 1D predictions back into the proper 3D format
        # batch size, num_sensors, horizon_steps
        return predictions.view(batch_size, self.num_sensors, -1)