# High-Frequency Trading Research Project

This repository contains research work on High-Frequency Trading (HFT) under the supervision of professors Mathieu Rosenbaum and Charles-Albert Lehalle. The project focuses on analyzing market microstructure and developing trading strategies using high-frequency data.

## Project Structure

```
HFT/
├── curating/          # Data curation and preprocessing scripts
├── models/           # Trading models and strategies
├── viz/              # Visualization tools and scripts
├── results/          # Output files and analysis results
├── logs/             # Log files for debugging and monitoring
├── tests/            # Unit tests and validation scripts
├── report/           # Documentation and research reports
└── refs/             # Reference materials and papers
```

## Data Format

The project uses market data from Databento, which provides detailed order book information. The data format includes:

- `publisher_id`: Dataset and venue identifier
- `instrument_id`: Unique instrument identifier
- `ts_event`: Matching-engine timestamp (nanoseconds since UNIX epoch)
- `price`: Order price (1 unit = 0.000000001)
- `size`: Order quantity
- `action`: Event type (Add, Cancel, Modify, Trade, Fill)
- `side`: Trading side (Ask/Bid)
- `depth`: Book level information
- Additional fields for bid/ask prices, sizes, and counts at different levels

## Features

1. **Data Curation**
   - Processing raw market data
   - Calculating tick sizes and price statistics
   - Data cleaning and normalization

2. **Analysis Tools**
   - Market microstructure analysis
   - Order book dynamics
   - Price impact studies

3. **Trading Models**
   - High-frequency trading strategies
   - Market making algorithms
   - Risk management systems

## Prerequisites

Before setting up the project, ensure you have the following:

1. Make the launch script executable:
   ```bash
   chmod +x launch.sh
   ```

2. Run the launch script to set up the environment:
   ```bash
   ./launch.sh
   ```

This script will:
- Install `uv` package manager
- Create and activate a virtual environment
- Install project dependencies
- Run environment tests

> **Important**: If you need to work with the project's imports, make sure to run:
> ```bash
> uv pip install -e .
> ```
> This will install the project in development mode, making all imports valid.

> Bien changer dans .env le path de la data
.env : FOLDER_PATH = "/home/janis/HFTP2/HFT/data/DB_MBP_10/"
## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env`
4. Run the data processing pipeline:
   ```bash
   python -m curating.process_data
   ```

## Usage

The project provides various scripts and notebooks primarily within the `data/curating` directory for data preprocessing, analysis, and visualization. Here's how to use some of the key components:

**1. Data Curation and Preprocessing:**

*   **General Curation (`curating.py`, `curating.ipynb`):** Contains core functions for loading, cleaning, and preparing the Databento MBO data. The notebook allows for interactive exploration of these steps.
    ```bash
    # Example: Running the main curation script (if structured as a module)
    python -m data.curating.curating 
    # Or explore interactively in curating.ipynb
    ```
*   **Time Expansion (`timeexpansion.ipynb`):** Illustrates how raw tick data is expanded or sampled into different time frequencies. Primarily for understanding and visualization.
*   **Mid Price Calculation (`mid_price.py`):** Utility script to calculate the mid-price from order book data.
    ```python
    # Example usage within another script
    from data.curating.mid_price import calculate_mid_price
    mid_price_series = calculate_mid_price(order_book_data)
    ```
*   **Sampling (`sample.py`, `sample.ipynb`):** Scripts and notebooks for various data sampling techniques based on time or events.

**2. Statistical Analysis:**

*   **Hill Estimator (`hill.py`, `hill.ipynb`):** Implements the Hill estimator for analyzing the tail index (fat tails) of return distributions. The notebook provides visualizations.
    ```bash
    # Example: Running the Hill analysis script
    python data/curating/hill.py --input_file <path_to_returns> --output_dir results/hill/
    # Or use functions interactively from the notebook
    ```
*   **Stationarity Tests (`stationnarity.py`):** Applies stationarity tests (e.g., ADF) and change point detection methods to time series data.
    ```bash
    python data/curating/stationnarity.py --input_file <path_to_timeseries> --output_dir results/stationarity/
    ```
*   **Volatility Extraction (`extract_vol_process.py`, `extract_vol_process.ipynb`):** Calculates volatility estimates at different time scales, potentially using methods like realized volatility or bipower variation, and analyzes Hurst exponents.
    ```bash
    python data/curating/extract_vol_process.py --input_file <path_to_prices> --output_dir results/volatility/
    ```
*   **Correlation Analysis (`correlation.py`, `correlation.ipynb`):** Computes and visualizes correlation matrices between different assets at various time scales.
    ```bash
    python data/curating/correlation.py --input_dir <path_to_returns_folder> --output_dir results/correlation/
    ```
*   **Copula Analysis (`copulas.py`, `copulas.ipynb`, `copulabis.py`, `copulabis.ipynb`, `copulas_torun.py`):** Fits different copula models (e.g., Gaussian, Student's t, Clayton, Gumbel, Frank) to study the dependence structure between asset returns. Contains scripts for running analysis and notebooks for visualization.
    ```bash
    # Example: Running a specific copula analysis
    python data/curating/copulas_torun.py --asset_pair GOOGL-AAPL --date 2024-07-22 --timescale 1s
    ```
*   **Inter-arrival Times (`quick_inter_arrival_google.py`, `quick_inter_arrival_google.ipynb`):** Analyzes the distribution of time intervals between market events (e.g., trades or price changes) for specific assets like GOOGL.
*   **Gaussian Tests (`gaussiantest.py`, `gaussiantest.ipynb`, etc.):** Scripts likely used to test the normality of return distributions at different scales.

**3. Hawkes Process Analysis:**

*   **Hawkes Investigation (`hawkes_investigation.py`, `hawkes_investigation.ipynb`):** Explores the fitting and analysis of Hawkes processes to model the self-exciting nature of market events, including jump detection and classification (endogenous/exogenous).
    ```bash
    # Example execution
    python data/curating/hawkes_investigation.py --input_file <path_to_events> --output_dir results/hawkes/
    ```
*   **Quick Hawkes (`quick_hawkes_google.py`, `quick_hawkes_google.ipynb`, `quick_hawks_google.ipynb`):** Focused notebooks and scripts for preliminary Hawkes process analysis, possibly specifically on GOOGL data.

**4. Visualization:**

*   **General Visualization (`viz.py`, `viz.ipynb`):** Contains utility functions and notebooks for creating various plots used throughout the analysis (e.g., price series, return distributions, correlation heatmaps).
    ```python
    # Example usage within another script
    from data.curating.viz import plot_time_series
    plot_time_series(data['price'], title='Asset Price')
    ```

*Note: The exact command-line arguments or function calls might differ. Refer to the individual `.py` scripts or `.ipynb` notebooks for precise usage instructions.*

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Supervisors: Mathieu Rosenbaum and Charles-Albert Lehalle
- Data provider: Databento
- Contributors and collaborators



Useful tools : 
git-filter-repo : https://github.com/newren/git-filter-repo
nbstripout : https://github.com/kynan/nbstripout
nbconvert : https://nbconvert.readthedocs.io/en/latest/
