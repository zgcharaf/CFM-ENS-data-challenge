# CFM-ENS-data-challenge
## Overview
Capital Fund Management (CFM) is an alternative asset management firm established in 1991, specializing in quantitative trading across global capital markets. CFM leverages a statistically robust analysis of vast datasets to guide its asset allocation and trading strategies. The firm emphasizes a collaborative and innovative work culture.

This repository is dedicated to a challenge where the objective is to identify stocks from sequences of market order book updates. The data, though anonymized and seemingly non-descriptive, contains latent features that can hint at the specific stocks, such as transaction frequencies, order sizes, and price spreads.

## Data Description
The dataset comprises detailed order book updates from multiple trading venues, capturing the dynamics of stock transactions over approximately two years. Each entry in the dataset represents a sequence of 100 consecutive atomic updates to the order books, with multiple such sequences available for each stock per day.
## Data Link   
Access the dataset for the CFM-ENS-data-challenge through the following link:
CFM ENS Challenge Dataset

### Structure
- **X**: Each sequence contains 20 daily sequences for each of the 24 stocks over 504 days, resulting in a total of 24,240,000 lines. The columns include:
  - `obs_id`: Unique identifier for each sequence.
  - `venue`: Encoded integer representing the trading venue (e.g., NASDAQ, BATY).
  - `action`: Type of event in the order book (`A` for add, `D` for delete, `U` for update).
  - `order_id`: Masked identifier for each order to track its changes.
  - `side`: Side of the order book affected (`A` for ask, `B` for bid).
  - `price`: Price of the order.
  - `bid`: Best bid price.
  - `ask`: Best ask price.
  - `bid_size`: Volume at the best bid price.
  - `ask_size`: Volume at the best ask price.
  - `flux`: Change in volume due to the event.
  - `trade`: Boolean indicating whether the event resulted from a trade.

- **Y**: Labels are categorized into integers (0-23), with each integer representing one of the 24 stocks.

## Data Preprocessing
Data is preprocessed to form a feature tensor of shape (100, 30) for each sequence, using:
- Embeddings for categorical items (`venue`, `action`, `trade`).
- Log-transformed sizes (`bid_size`, `ask_size`, `flux`).
- Prices are adjusted by subtracting the best bid price of the first event in each sequence from all price-related columns.

## Model Architecture
The model uses a bidirectional GRU architecture to process the sequences, followed by dense layers to classify each sequence into one of the 24 stock categories:
- Two GRU layers of size 64 process the data in both forward and reverse directions.
- The outputs are concatenated into a 128-dimensional vector.
- This vector passes through two dense layers:
  - The first reduces dimensions to 64 and applies SeLU activation.
  - The second outputs 24 probabilities using softmax.

## Training
- **Loss Function**: Cross-entropy.
- **Optimizer**: Adam with a learning rate of 3e-3.
- Training involves 10,000 batches, each containing 1,000 observations.

## Usage
Instructions for using this repository:
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed (e.g., TensorFlow, NumPy).
3. Load and preprocess the data as outlined.
4. Train the model using the provided scripts.
5. Evaluate the model on the test dataset to assess its performance.

