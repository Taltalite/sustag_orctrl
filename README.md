# sustag_orctrl

We introduce SUSTag-ORCtrL that uses raw ionic current signatures of DNA tags for demultiplexing into 96 or 384 classes. Our system, involving a Bhattacharyya distance and incremental clustering enhanced molecular tag design (SUSTag), an optional-reject CNN-LSTM deep learning model, and a transfer-learning pipeline (ORCtrL), minimizes crosstalk for random access.

## Preparation

### Clone the repository

``` bash
git clone https://github.com/Taltalite/sustag_orctrl.git
```

### Requirements

Before you begin, please ensure you have a compatible Python environment set up. Follow these steps carefully:

1.  **Create the Conda Environment**
    This command creates a new Conda environment named `sustag_orctrl` with the base dependencies listed in the `env.yml` file.

    ```bash
    conda env create -f env.yml -n <env_name>
    conda activate <env_name>
    ```

2.  **Install the MinKNOW API**
    The Oxford Nanopore MinKNOW API is required for real-time communication with the sequencing device. Install it using `pip`:

    ```bash
    pip install minknow-api
    ```
3.  **Install PyTorch**
    To ensure you have the correct PyTorch version for your hardware (e.g., CPU-only or a specific CUDA version for GPU acceleration), we recommend installing it by following the official guide.

      * Visit the PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
      * Select your preferences (OS, Package, Compute Platform).
      * Run the generated command.

    For example, a typical command for installing PyTorch with CUDA 11.8 support via pip would look like this (**do not run this command without verifying it on the PyTorch website**):

## Quick Start

This guide provides instructions to quickly set up and run a simulated real-time adaptive sampling experiment using the `sustag_orctrl` project.

### Running the Simulation

Once your environment is ready, you can run a test that simulates a real-time nanopore sequencing experiment. This script uses existing `.fast5` files to mimic a live data stream.

Follow these steps to start the simulation:

```bash
cd sustag_orctrl/readuntil

python fast5_test.py --toml ./config.toml --num-reads 3000 --stats-file fast5_test.csv --log-file fast5_test.log 
```

You can change the fast5 file path for simulation input by modifying the ```fast5_directory``` parameter in the ```.toml``` configuration file.

### Expected Output

In the terminal interface or log file, you will see output similar to this, indicating that the simulation program is running normally.

```bash
2025-08-20 17:46:02,573 - INFO - ✅ CH:512 ID:d1178732-bf2f-4000-9829-b487dc7b931f - Target - pred=40, conf=0.99
2025-08-20 17:46:02,573 - INFO - Fast5Client: Received STOP_RECEIVING for CH:512, Read:d1178732-bf2f-4000-9829-b487dc7b931f
2025-08-20 17:46:02,578 - INFO - ❌ CH:87 ID:ccc3e731-bb5e-46c0-9f3b-1e4e479555c5 - Non-target - pred=23, conf=0.99
2025-08-20 17:46:02,578 - INFO - Fast5Client: Received UNBLOCK for CH:87, Read:ccc3e731-bb5e-46c0-9f3b-1e4e479555c5. Simulating removal.
2025-08-20 17:46:02,578 - INFO - ❌ CH:87 ID:ccc3e731-bb5e-46c0-9f3b-1e4e479555c5 - Non-target - pred=23, conf=0.99
2025-08-20 17:46:02,578 - INFO - Fast5Client: Received UNBLOCK for CH:87, Read:ccc3e731-bb5e-46c0-9f3b-1e4e479555c5. Simulating removal.
2025-08-20 17:46:02,583 - INFO - Data reporting interval: 0.0114 sec. (This batch contains 11 reads)
```


-----

## Usage

This section details how to use the `sustag_orctrl` project, from acquiring data to performing real-time analysis.

### 1\. Dataset Download

We have prepared a sample dataset for demonstration and testing purposes. You can download it from the following link:

[https://mirrors.sustech.edu.cn/site/datasets-share/sustag\_orctrl/](https://mirrors.sustech.edu.cn/site/datasets-share/sustag_orctrl/)

In addition, you can see the file tree structure of the dataset in ```sustag_data.md```

### 2\. Generating a Custom Dataset

If you want to use your own sequencing data, follow this pipeline to generate a dataset compatible with our training scripts.

1.  **Basecall with Guppy**

    Use Guppy (version 6.0.0 is recommended) to basecall your raw data. It is crucial to enable the `--fast5_out` option to ensure that the output `.fast5` files contain the basecalling information required for later steps.

2.  **Map Reads to a Reference**

    Align the basecalled reads from the previous step to your reference sequence using `minimap2`. You can use the provided helper script `dataset_gen/mapping.py` to streamline this process.

3.  **Generate Read-to-Reference Dictionary**

    Run the `dataset_gen/readsdic_gen.py` script. This will process the alignment results from `minimap2` and create a dictionary that maps each read ID to its corresponding reference sequence.

4.  **Extract Raw Signal Data**

    Use the `dataset_gen/data_extraction.py` script to extract the raw signal data from the `.fast5` files. This script uses the dictionary created in the previous step to selectively extract only the reads that successfully mapped to your reference.

5.  **Create Training Dataset**

    Finally, run `dataset_gen/pickleset_gen_1d.py` on the extracted data. This will package the raw signals and corresponding labels into a single `.pickle` file, which is the final format required for training the models.

### 3\. Model Training

The `train/` directory contains scripts for training the different neural network architectures used in this project (e.g., `train_CNN.py`, `train_CNNLSTM.py`, `train_ORCtrL.py`). You can execute the appropriate script to train a model on the dataset you have prepared.

### 4\. Real-time Adaptive Sampling

Our real-time adaptive sampling capability is built upon the `readuntil_api`, with modifications inspired by the [WarpDemuX](https://github.com/KleistLab/WarpDemuX) project. We have further updated the code to ensure compatibility with the version 6.x of ```minknow_api``` (The version applied for this project is v6.4.3).

To perform a real-time "Read Until" experiment, follow these steps:

1.  Configure the `readuntil/config.toml` file with your specific settings, such as the path to your trained model checkpoint and decision-making thresholds.
2.  Start a new sequencing experiment in the Oxford Nanopore MinKNOW software.
3.  Once the sequencing run is active and generating data, execute the `run_adaptive_sampling.py` script:
    ```bash
    python readuntil/run_adaptive_sampling.py
    ```
    The script will connect to the MinKNOW API and begin classifying reads in real-time.

> **IMPORTANT TIP FOR FASTER RESPONSE:**
> To achieve faster decision-making, we strongly recommend modifying a MinKNOW configuration file to shorten the data reporting interval.
>
>   * **File Location (Windows):** `C:\Program Files\OxfordNanopore\MinKNOW\conf\package\sequencing\`
>   * **File to Modify:** `sequencing_MIN106_DNA.toml` (or the equivalent for your flow cell)
>   * **Change:** In the `[analysis_configuration.read_detection]` section, set the `break_reads_after_seconds` parameter to `0.1`.
>     ```toml
>     [analysis_configuration.read_detection]
>     break_reads_after_seconds = 0.1 
>     ```
>
> **Disclaimer:** Some research suggests that modifying this parameter should be done in conjunction with other settings (e.g., `rules_in_execution_order`). We have not deeply investigated these interactions. Therefore, our recommended change is primarily suited for demonstration purposes and may not be optimal for all experimental conditions.


## License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.

You can find the full text of the license in the [LICENSE](https://www.google.com/search?q=LICENSE) file at the root of this repository.

### A Note on Project Components

The `readuntil/readuntil_api` directory contains code that is a modification of a pre-existing project which is also licensed under MPL-2.0. To maintain compliance and simplify the overall licensing for `sustag_orctrl`, we have chosen to license the entire project under the MPL-2.0.
