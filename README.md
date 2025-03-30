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

The project is structured to support various research workflows in the future:

1. Data Processing
   ```python
   from curating.processor import MarketDataProcessor
   processor = MarketDataProcessor()
   processor.process_data()
   ```

2. Analysis
   ```python
   from models.analyzer import MarketAnalyzer
   analyzer = MarketAnalyzer()
   results = analyzer.analyze_market_impact()
   ```

3. Visualization
   ```python
   from viz.plotter import MarketPlotter
   plotter = MarketPlotter()
   plotter.plot_order_book()
   ```

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

essayer le microprice aussi,
on peut plus faire l'hypothese modele bachelier à partir d'un certain temps pour la vol

sudo apt update
sudo apt install python3-dev python3-numpy
added swig, & tick

faire des commits et dig la librairie tick x datainitiative

CPPFLAGS="-I /home/janis/HFTP2/HFT/.venv/lib/python3.12/site-packages/numpy/_core/include" uv add tick ---> be careful with that




added pyhawkes
added pybasicbayes