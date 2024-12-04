# Name - Abhyuday Pratap Singh
# Roll  no - 2201008
# MSE is not explicitly used in the code you shared, but it could be used for model evaluation to measure how well the model's predictions match the actual stock prices.
# Typically, this could be used for comparison between models or for hyperparameter tuning
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Initialize the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Stock Price Prediction App'),
    html.P('''This app uses advanced machine learning techniques to analyze past stock prices and 
           predict the stock price for the next day. It provides insights into key technical 
           indicators and features influencing stock price predictions.'''),
    
    # User inputs
    html.Div([
        html.Label('Enter Stock Symbol (e.g., AAPL, GOOGL):'),
        dcc.Input(id='symbol-input', value='AAPL', type='text', style={'margin-right': '10px'}),
        html.Button('Analyze and Predict', id='analyze-button', n_clicks=0)
    ]),
    html.Br(),
    
    # Explanation Section
    html.Div([
        html.H3('About Stock Price Prediction'),
        html.P('''This application predicts stock prices using historical data and technical 
               indicators. It identifies patterns and trends in the data, which can help investors 
               and analysts make informed decisions. The prediction is for the next day, based on 
               the selected stock's recent performance.''')
    ], style={'margin-bottom': '20px'}),
    
    # Results section
    html.Div(id='results-container')
])

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data using yfinance."""
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

def add_technical_indicators(df):
    """Add technical indicators to the stock data."""
    # Moving Averages
    df['MA5']  =  df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Volatility and Momentum Indicators
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD and Signal Line
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Rate of Change (ROC)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Volume Indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    return df

def prepare_data(df):
    """Prepare data for training the model."""
    df['Target'] = df['Close'].shift(-1)
    features = ['Close', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line', 
                'ROC', 'Volume', 'Volume_MA5', 'Volatility', 'Daily_Return']
    df = df.dropna()
    X = df[features]
    y = df['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selector = SelectKBest(score_func=f_regression, k=8)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, y, selected_features, scaler

def create_ensemble_model():
    """Create an ensemble model for prediction."""
    lr = LinearRegression()
    ridge = Ridge(alpha=1.0)
    dt = DecisionTreeRegressor(max_depth=5)
    model = VotingRegressor([('lr', lr), ('ridge', ridge), ('dt', dt)])
    return model

@app.callback(
    Output('results-container', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('symbol-input', 'value'),
    prevent_initial_call=True
)
def update_results(n_clicks, symbol):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Fetch and process data
        df = fetch_stock_data(symbol, start_date, end_date)
        df = add_technical_indicators(df)
        X, y, selected_features, scaler = prepare_data(df)
        
        # Train the model
        model = create_ensemble_model()
        cv_scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='r2')
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Make future prediction
        last_data = X[-1:]
        future_price = model.predict(last_data)[0]
        current_price = df['Close'].iloc[-1]
        price_change = ((future_price - current_price) / current_price) * 100
        
        # Plot predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-len(y):], y=y, mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=df.index[-len(y_pred):], y=y_pred, mode='lines', name='Predicted Price'))
        fig.update_layout(title=f'{symbol} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price')
        
        # Display results
        return html.Div([
            html.H3('Model Performance'),
            html.P(f'Cross-validation R² scores: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})'),
            html.P(f'Final R² Score: {r2_score(y, y_pred):.3f}'),
            
            html.H3('Prediction for Next Day'),
            html.P(f'Predicted Price: ${future_price:.2f} ({price_change:+.2f}%)'),
            html.P(f'Current Price: ${current_price:.2f}'),
            
            html.H3('Key Features Used for Prediction'),
            html.P(', '.join(selected_features)),
            
            dcc.Graph(figure=fig)
        ])
    except Exception as e:
        return html.Div([html.H3('Error'), html.P(str(e))])

if __name__ == '__main__':
    app.run_server(debug=True)
