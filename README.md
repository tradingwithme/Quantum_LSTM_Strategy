<h1>Quantum_LSTM_Strategy</h1>
<p>Quantum-Inspired LSTM Backtesting Strategy</p>

<p>This project implements a stock trading strategy workflow incorporating a quantum-inspired approach (optional integration), an LSTM-based predictive model, trading simulation, and reporting. It also includes optional components for classical portfolio optimization and risk analysis.</p>

<h2>Project Structure</h2>
<p>The project is organized into several Python scripts and directories:</p>
<ul>
    <li><code>main.py</code>: The main entry point of the application. Orchestrates the entire workflow based on command-line arguments.</li>
    <li><code>preprocess_data.py</code>: Contains functions for fetching historical stock data, adding technical indicators, creating sequential data for the model, and scaling features. Saves raw and processed data to the <code>data/</code> directory.</li>
    <li><code>modeling.py</code>: Defines the LSTM model architecture and includes functions for training the model (with optional fine-tuning) and saving/loading model weights and configuration. Saves models to the <code>models/</code> directory.</li>
    <li><code>evaluate_model.py</code>: Provides a standalone function for evaluating the trained model on a dataset, saving core metrics to the <code>core/</code> directory, and optionally logging details to the <code>output/</code> directory.</li>
    <li><code>simulation.py</code>: Implements the trading simulation logic based on model predictions and technical indicators. Saves simulation trades and the equity curve to the <code>output/</code> directory.</li>
    <li><code>report.py</code>: Generates a report summarizing the simulation results, including performance metrics (Sharpe Ratio, Max Drawdown) and an equity curve plot. Saves the report, metrics (in JSON format), and plot to the <code>output/</code> directory.</li>
    <li><code>config/</code>: Directory for configuration files.
        <ul>
            <li><code>config/optimizer.py</code>: Contains functions for classical Mean-Variance Portfolio Optimization and generating the Efficient Frontier using SciPy and CVXPY. Also includes basic risk metric calculations (imported from <code>metrics.py</code>) and saves optimization results to the <code>config/optimizer_results/</code> directory.</li>
        </ul>
    </li>
    <li><code>risk_control.py</code>: Contains functions for implementing risk control measures like stop-losses, cash limits, and portfolio rebalancing (though these are not directly integrated into the current <code>simulation.py</code> loop but could be used for analysis or in a more advanced simulation).</li>
    <li><code>metrics.py</code>: Contains core functions for calculating performance and risk metrics (Sharpe Ratio, Max Drawdown, CVaR, Portfolio Performance).</li>
    <li><code>requirements.txt</code>: Lists the required Python libraries and their versions.</li>
    <li><code>data/</code>: Directory where raw and processed data files will be saved.</li>
    <li><code>models/</code>: Directory where trained model weights and configuration will be saved.
    <li><code>results/</code>: Directory where simulation results, reports, and plots will be saved.</li>
    <li><code>core/</code>: Directory where core metrics (e.g., evaluation accuracy) will be saved.</li>
</ul>

<h2>Setup and Installation</h2>
<ol>
    <li>
        <p><strong>Clone the repository (if applicable):</strong></p>
        <pre><code>git clone &lt;repository_url&gt;
cd <repository_directory>

Create and activate a virtual environment (recommended):

python -m venv .venv

On Windows:

.venv\Scripts\activate


On macOS/Linux:

source .venv/bin/activate

Install required packages:

pip install -r requirements.txt
<h2>How to Run `main.py`</h2>
<p>The `main.py` script is the entry point and uses command-line arguments to control the workflow. You can run it from your terminal within the activated virtual environment.</p>

<h3>Basic Execution</h3>
<p>Run the default workflow (data preprocessing, model training/loading, simulation, and basic reporting) with default tickers (AAPL, MSFT) and hyperparameters:</p>
<pre><code>python main.py</code></pre>

<h3>Command-Line Arguments</h3>
<p>You can customize the execution using the following arguments:</p>
<ul>
    <li><code>--tickers &lt;TICKER1&gt; [&lt;TICKER2&gt; ...]</code>: List of tickers to process (e.g., <code>--tickers TSLA AMZN GOOG</code>). Defaults to `['AAPL', 'MSFT']`.</li>
    <li><code>--sequence_length &lt;INT&gt;</code>: Number of time steps in each sequence for the LSTM model. Defaults to `20`.</li>
    <li><code>--initial_cash &lt;FLOAT&gt;</code>: Initial cash for the trading simulation. Defaults to `10000`.</li>
    <li><code>--trading_ticker &lt;STR&gt;</code>: Ticker for which the trading simulation is performed. Defaults to `'AAPL'`.</li>
    <li><code>--hidden_dim &lt;INT&gt;</code>: Number of units in LSTM hidden layers. Defaults to `64`.</li>
    <li><code>--dropout_prob &lt;FLOAT&gt;</code>: Dropout rate for regularization in the LSTM model. Defaults to `0.3`.</li>
    <li><code>--lr &lt;FLOAT&gt;</code>: Learning rate for the optimizer during model training. Defaults to `0.001`.</li>
    <li><code>--epochs &lt;INT&gt;</code>: Number of training epochs. Defaults to `50`.</li>
    <li><code>--batch_size &lt;INT&gt;</code>: Batch size for training. Defaults to `32`.</li>
    <li><code>--model_path &lt;STR&gt;</code>: Path to save/load the best model state dictionary. Defaults to `'models/best_model.pth'`.</li>
    <li><code>--config_path &lt;STR&gt;</code>: Path to save/load the model configuration JSON file. Defaults to `'models/model_config.json'`.</li>
    <li><code>--skip_training</code>: If set, skips model training and attempts to load an existing model from <code>--model_path</code> and <code>--config_path</code>.</li>
    <li><code>--output_dir &lt;STR&gt;</code>: Directory to save output files (simulation results, reports, plots). Defaults to `'results'`.</li>
    <li><code>--core_dir &lt;STR&gt;</code>: Directory to save core metrics (e.g., evaluation accuracy). Defaults to `'core'`.</li>
    <li><code>--log_evaluation_details</code>: If set, enables logging of detailed evaluation results to the output directory.</li>
    <li><code>--run_portfolio_optimization</code>: If set, runs classical portfolio optimization after the simulation.</li>
    <li><code>--enable_risk_control_analysis</code>: If set, performs risk control analysis (e.g., calculates drawdown and CVaR) on the simulation results.</li>
    <li><code>--quantum_enabled</code>: If set, enables quantum-inspired/quantum features if they are implemented and available. **Note:** The current implementation includes this argument as a flag but does not contain the actual quantum optimization logic. This is a placeholder for future integration.</li>
</ul>

<h3>Examples</h3>
<p>Run with custom tickers and more epochs:</p>
<pre><code>python main.py --tickers TSLA AMZN GOOG --epochs 100</code></pre>

<p>Run with custom tickers, more epochs, and enable classical portfolio optimization:</p>
<pre><code>python main.py --tickers TSLA AMZN GOOG --epochs 100 --run_portfolio_optimization</code></pre>

<p>Skip training and load a pre-trained model from custom paths:</p>
<pre><code>python main.py --skip_training --model_path my_models/my_lstm.pth --config_path my_models/my_config.json</code></pre>

<p>Run with default settings and enable risk control analysis:</p>
<pre><code>python main.py --enable_risk_control_analysis</code></pre>

<h2>Considerations When Running the Algorithm</h2>

<ul>
    <li>
        <p><strong>Data Availability:</strong> The script relies on fetching historical data using `yfinance` and `webull`. Data availability and quality can vary depending on the ticker and time period. Rate limits from data providers can also impact data fetching.</p>
    </li>
    <li>
        <p><strong>Model Training Time:</strong> Training the LSTM model can be time-consuming, especially with large datasets or many epochs. Consider using a GPU runtime if available.</p>
    </li>
    <li>
        <p><strong>Hyperparameter Tuning:</strong> The default hyperparameters for the LSTM model are starting points. Optimal performance may require tuning `hidden_dim`, `dropout_prob`, `lr`, `epochs`, and `batch_size` based on your specific data and goals.</p>
    </li>
    <li>
        <p><strong>Model Performance:</strong> The LSTM model's predictive accuracy is crucial for the trading strategy. Evaluate the model's performance on the evaluation set before relying on simulation results.</p>
    </li>
    <li>
        <p><strong>Trading Strategy Logic:</strong> The simulation uses a specific trading strategy based on model predictions and technical indicators (MACD and RSI). This strategy is a simplified example and may not be profitable or suitable for all market conditions. Consider modifying or developing a more sophisticated strategy.</p>
    </li>
    <li>
        <p><strong>Portfolio Optimization:</strong> The classical portfolio optimization is performed *after* the simulation and is based on historical returns. It provides insights into potential portfolio allocations but is not dynamically integrated into the simulation's trading decisions in the current setup.</p>
    </li>
    <li>
        <p><strong>Risk Control:</strong> The `risk_control.py` script provides functions for risk management, but these are not actively used within the main simulation loop. Implementing features like dynamic stop-losses or rebalancing during the simulation would require modifying the `simulation.py` script.</p>
    </li>
     <li>
        <p><strong>Quantum Integration:</strong> The `--quantum_enabled` flag is included as a placeholder for integrating quantum-inspired or quantum optimization algorithms. The actual implementation of such algorithms and their integration into the workflow (e.g., using quantum optimization for portfolio allocation or feature selection) would need to be added separately.</p>
    </li>
    <li>
        <p><strong>Backtesting Limitations:</strong> The simulation is a backtest based on historical data. Past performance is not indicative of future results. Factors like transaction costs, slippage, and market impact are not explicitly modeled in the current simulation.</p>
    </li>
    <li>
        <p><strong>Dependencies:</strong> Ensure all required libraries listed in `requirements.txt` are installed. Some libraries like `cvxpy` might require additional solver installations depending on your system.</p>
    </li>
</ul>
