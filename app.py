from flask import Flask, request, Response, json
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import logging

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, dropout):
        super(EnhancedBiLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_layer_size * 2, output_size * 2)  # Adjusted to match saved model

    def forward(self, input_seq):
        h_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c_0 = torch.zeros(self.num_layers * 2, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        lstm_out, _ = self.lstm(input_seq, (h_0, c_0))
        predictions = self.linear(lstm_out[:, -1])
        return predictions


model = EnhancedBiLSTMModel(input_size=2, hidden_layer_size=115, output_size=1, num_layers=2, dropout=0.3)

try:
    state_dict = torch.load("enhanced_bilstm_model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
except Exception as e:
    logger.error(f"Error loading model state_dict: {e}")

model.eval()

def get_binance_url(symbol="ETHUSDT", interval="1m", limit=1000):
    return f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

def compute_macd(df, short_window=12, long_window=26, signal_window=9):
    df['MACD'] = df['price'].ewm(span=short_window, adjust=False).mean() - df['price'].ewm(span=long_window, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

@app.route("/inference/<string:token>")
def get_inference(token):
    if model is None:
        return Response(json.dumps({"error": "Model is not available"}), status=500, mimetype='application/json')

    symbol_map = {
        'ETH': 'ETHUSDT',
        'BTC': 'BTCUSDT',
        'BNB': 'BNBUSDT',
        'SOL': 'SOLUSDT',
        'ARB': 'ARBUSDT'
    }

    token = token.upper()
    if token in symbol_map:
        symbol = symbol_map[token]
    else:
        return Response(json.dumps({"error": "Unsupported token"}), status=400, mimetype='application/json')

    url = get_binance_url(symbol=symbol)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
        df = df[["close_time", "close"]]
        df.columns = ["date", "price"]
        df["price"] = df["price"].astype(float)

        df = compute_macd(df)

        if symbol in ['BTCUSDT', 'SOLUSDT']:
            df = df.tail(10)
        else:
            df = df.tail(20)

        current_price = df.iloc[-1]["price"]
        current_time = df.iloc[-1]["date"]
        logger.info(f"Current Price: {current_price} at {current_time}")

        features = df[['price', 'MACD']].values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(features)

        seq = torch.FloatTensor(scaled_data).view(1, -1, 2).to(torch.device('cpu'))

        with torch.no_grad():
            y_pred = model(seq)

        predicted_prices = scaler.inverse_transform(y_pred.numpy().reshape(-1, 2))

        try:
            if symbol in ['BTCUSDT', 'SOLUSDT']:
                predicted_price = round(float(predicted_prices[0][0]), 2)
            else:
                if len(predicted_prices) > 1:
                    predicted_price = round(float(predicted_prices[1][0]), 2)
                else:
                    predicted_price = round(float(predicted_prices[0][0]), 2)
        except IndexError as e:
            return Response(json.dumps({"error": f"IndexError: {str(e)}"}), status=500, mimetype='application/json')

        logger.info(f"Prediction: {predicted_price}")

        return Response(json.dumps(predicted_price), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"error": "Failed to retrieve data from Binance API", "details": response.text}),
                        status=response.status_code,
                        mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
