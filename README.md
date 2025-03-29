# Foosha

A collection of autoregressive model implementations and experiments, including various papers and models in the field of language modeling.

## Features

- Implementation of various autoregressive models and papers
- Support for training and fine-tuning language models
- Integration with Lightning Fabric for efficient training
- Example implementations and experiments
- Support for various model architectures and training techniques

## Installation

```bash
git clone https://github.com/joey00072/fuchsia.git
cd fuchsia
pip install -e .
```

## Quick Start

### Download and Pre-tokenize Dataset
```bash
python examples/prepare-dataset.py
```

### Training
You can use either the standard PyTorch version or the Lightning Fabric version (recommended):

```bash
# Standard PyTorch version
python examples/train_llama.py

# Lightning Fabric version (recommended)
python examples/train_llama_fabric.py
```

### Inference
For inference on models like Phi-2:
```bash
python examples/phi_inference.py
```

## Implemented Papers/Models

- TokenFormer
- MLA
- Griffin & Hawk
- Galore
- Qsparse
- Bitnet
- Renet
- Alibi Embeddings
- Rotary Embeddings
- LoRA
- LLAMA
- XPOS
- Mamba
- GPT

## Project Structure

- `fuchsia/`: Main package directory
- `examples/`: Example implementations and scripts
- `docs/`: Documentation and notes
- `experiments/`: Experimental code and results

## Development Status

This project is under active development. Some features that are planned or in progress:

- Improved inference class
- Enhanced training loop using Lightning Fabric
- Structured fine-tuning implementation
- DPO (Direct Preference Optimization) support
- PyModel integration for experiment management

## Contributing

Contributions are welcome! Please follow these guidelines:

- Be nice and respectful
- Include code explanations and documentation
- Memes in PRs are appreciated

## License

MIT License

## Support

If you find this project helpful, consider supporting the development:
[Ko-fi](https://ko-fi.com/yourusername)

