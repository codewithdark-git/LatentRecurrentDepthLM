# Latent Recurrent Depth Language Model

[![Model](https://img.shields.io/badge/transformer-Model-orange?logo=pytorch)](https://huggingface.co/models/codewithdark/latent-recurrent-depth-lm)  
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Space-yellow?logo=huggingface)](https://huggingface.co/spaces/codewithdark/LatentRecurrentDepthLM)  
[![arXiv](https://img.shields.io/badge/arXiv-2502.05171-b31b1b.svg)](https://arxiv.org/abs/2502.05171)  
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  

Welcome to the **Latent Recurrent Depth Language Model** repository! This project provides an implementation of a deep language model that combines latent recurrent architectures with modern attention mechanisms. The model is designed for efficient sequence modeling and language understanding tasks.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Push to Hub](#push-to-hub)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository implements a novel language modeling architecture that leverages:
- **Latent Recurrent Blocks**: To capture long-term dependencies.
- **Multi-Head Attention**: For modeling complex interactions between tokens.
- **Deep Stacking of Model Blocks**: To achieve depth and expressivity in the network.

The project is modularized to separate concerns such as data handling, tokenization, model definition, training pipelines, and inference utilities. This makes it easy to experiment with different configurations and extend the model.

---

## Features

- **Custom Dataset Processing**: Easily preprocess and load your text data using `dataset.py`.
- **Flexible Training Pipeline**: Train the model with configurable options using `trainer.py` and `pipeline.py`.
- **Inference Utilities**: Generate sequences or test model predictions with scripts in the `Inference/` directory.
- **Model Hub Integration**: Push trained models to popular hubs using `push_to_hub.py`.
- **Modular Model Design**: Separate model components in the `Model/` directory including:
  - `latent_Recurrent.py`
  - `recurrent_Block.py`
  - `prelude_Block.py`
  - `codaBlock.py`
  - `multi_head_Attention.py`

---

## Directory Structure

```plaintext
codewithdark-git-latentrecurrentdepthlm/
├── README.md
├── LICENSE
├── dataset.py
├── pipeline.py
├── push_to_hub.py
├── tokenizer.py
├── trainer.py
├── Inference/
│   ├── One_word.py
│   ├── Squence_Generator.py
│   └── locally.py
└── Model/
    ├── codaBlock.py
    ├── latent_Recurrent.py
    ├── model.py
    ├── multi_head_Attention.py
    ├── prelude_Block.py
    └── recurrent_Block.py
```

- **Root Files**: Core utilities for data processing, training, tokenization, and hub integration.
- **Inference/**: Contains scripts for various inference scenarios:
  - `One_word.py`: Likely for single-word prediction or testing.
  - `Squence_Generator.py`: For generating sequences.
  - `locally.py`: For running inference locally.
- **Model/**: Contains model definitions and components that build the architecture.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/codewithdark/latent-recurrent-depth-lm.git
   cd latent-recurrent-depth-lm
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   Install the required Python packages. For example, if using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** If a `requirements.txt` is not provided, ensure you have the following installed:
   > - Python 3.7+
   > - PyTorch 
   > - NumPy
   > - (Any other library required by your specific implementation)

---

## Usage

### Data Preparation

Use `dataset.py` to preprocess your text data. 

### Training

Start training the model by running the pipeline. You can adjust hyperparameters and training configurations within `pipeline.py` 

---

## Model Architecture

The model architecture is composed of several custom blocks:

- **latent_Recurrent.py & recurrent_Block.py**: Implements the recurrent components for sequence modeling.
- **prelude_Block.py & codaBlock.py**: Serve as the input and output blocks, respectively, to preprocess input tokens and postprocess model outputs.
- **multi_head_Attention.py**: Implements multi-head attention mechanisms that allow the model to focus on different parts of the input simultaneously.
- **model.py**: Combines all these components into a cohesive model that can be trained and evaluated.

The modular design allows for easy experimentation with different configurations and architectures.

---

## Push to Hub

To share your trained model with the community or deploy it on a model hub, use the `push_to_hub.py` script.
---

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).

---

Happy Modeling!

