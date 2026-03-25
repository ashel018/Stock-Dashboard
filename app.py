from flask import Flask, render_template, jsonify
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# 🏠 Home
@app.route('/')
def home():
    return render_template('index.html')

# 📈 Stock chart data
@app.route('/stock/<symbol>')
def get_stock(symbol):
    try:
        data = yf.download(symbol, period="1mo", progress=False)

        if data.empty:
            return jsonify({"dates": [], "prices": []})

        close_prices = data.iloc[:, 0]

        result = {
            "dates": [str(date.date()) for date in data.index],
            "prices": [float(p) for p in close_prices]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# 💰 Live price
@app.route('/price/<symbol>')
def get_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")

        if data.empty:
            return jsonify({"price": "N/A"})

        price = float(data['Close'].iloc[-1])

        return jsonify({"price": price})

    except Exception as e:
        return jsonify({"error": str(e)})

# 🤖 Prediction (ML)
@app.route('/predict/<symbol>')
def predict(symbol):
    try:
        data = yf.download(symbol, period="1mo", progress=False)

        if data.empty:
            return jsonify({"prediction": "N/A"})

        close_prices = data.iloc[:, 0].values

        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices

        model = LinearRegression()
        model.fit(X, y)

        next_day = np.array([[len(close_prices)]])
        prediction = model.predict(next_day)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)