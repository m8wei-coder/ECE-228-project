import torch
import torch.nn as nn

class CMAPSS_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(CMAPSS_LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_prob if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]

        out = self.fc1(last_time_step_out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        return out

def calculate_metrics(y_pred, y_true):
    mse = nn.functional.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse).item()

    h = y_pred - y_true
    score = torch.where(
        h < 0, 
        torch.exp(-h / 13.0) - 1, 
        torch.exp(h / 10.0) - 1
    )
    total_score = torch.sum(score).item()
    
    return rmse, total_score
