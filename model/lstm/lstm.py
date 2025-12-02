import torch
from torch import nn
from config_amass import *


class ArmPoseLSTM(nn.Module):
    """
    LSTM model for sequence-to-sequence arm pose prediction.
    """
    def __init__(self):
        super(ArmPoseLSTM, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE, 
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True, 
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            bidirectional=False
        )
        
        # A fully connected layer to map the LSTM output to our desired pose dimension
        self.fc = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Tensor: Output tensor of shape (batch_size, sequence_length, output_size)
        """
        # LSTM returns: output, (hidden_state, cell_state)
        # We only need the output for each time step
        # Shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take only the output from the LAST time step
        # last_time_step_out shape: (batch_size, hidden_size*2)
        last_time_step_out = lstm_out[:, -1, :]
        
        # To get a prediction for each time step in the sequence,
        # we apply the linear layer to the entire sequence.
        # PyTorch's Linear layer automatically handles this batching.
        # Shape: (batch_size, sequence_length, output_size)
        linear_out = self.fc(last_time_step_out)

        predictions = torch.tanh(linear_out)  # Constrain outputs to [-1, 1]

        return predictions