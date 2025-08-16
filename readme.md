
## Project Structure


```
project/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ CITATION.cff
├─ requirements.txt
├─ analysis.iypnb
├─ data/
│  └─ input.csv    
├─ figures/
├─ modules/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ deduplication.py
│  ├─ similirity.py
│  ├─ embeddings.py
│  ├─ network.py
│  ├─ evolution.py
│  ├─ clustering.py
│  ├─ building_taxonomy.py
|  └─ tracking_subclasses.py
```

## Setup Python Environment



1. Installing all the libraries and dependencies used by python to run the jupyter notebook. For this purpose, python should be already installed in the machine. The 

```bash
chmod +x setup.sh
./setup.sh
```

```bash
source .venv/bin/activate
```

## Reproduce Results: Running Jupyter Notebook

analysis.ipynb


## Embeddings

This project uses the open-source Qwen3-Embedding-8B model for generating text embeddings, chosen for its efficiency and accuracy. The file `data/items_df.pkl` already contains precomputed embeddings for both the original item texts and their tokenized versions.

### Reproducing Embeddings

To reproduce or update embeddings, you need to set up the Qwen3-Embedding-8B model locally. We recommend using [Ollama](https://ollama.com/) for easy deployment:

1. **Install Ollama**  
    Follow the instructions at [Ollama Installation Guide](https://ollama.com/download).

2. **Pull the Qwen3-Embedding-8B Model**  
    ```bash
    ollama pull qwen3-embedding:8b
    ```

3. **Run the Model Locally**  
    ```bash
    ollama run qwen3-embedding:8b
    ```

This will start a local API server for the model.

#### Generating Embeddings

Use the provided script to generate embeddings:

```bash
python modules/embeddings.py --input data/input.csv --output data/items_df.pkl
```

This script will call the local API and save the embeddings to `data/items_df.pkl`.

**Note:**  
- Ensure Ollama and the model server are running before executing the script.
- Adjust input/output paths as needed.


### Text Classification - Llama 3-8B

