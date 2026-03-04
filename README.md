# Gas Splatting

## Installation

**1. Clone repository**
```bash
git clone https://github.com/aleevr04/gas_splatting.git
cd gas_splatting
```

**2. Create and activate a virtual environment**

The most convenient way to install dependencies is by creating a virtual environment. There are several tools for this purpose like `venv` or `conda`. A `venv` example is shown below:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

**3. Install PyTorch**

Since PyTorch is highly hardware-dependent (CPU or GPU), please install it by following the instructions from their [official website](https://pytorch.org/get-started/locally/). CPU-only example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**4. Install project dependencies**

```bash
pip install -r requirements.txt
```

## Acknowledgements and License

This project introduces custom implementations and novel features developed specifically for **Gas Splatting**. However, the core architecture and several foundational utilities are deeply inspired by and built upon the open-source repository [r2_gaussian](https://github.com/ruyi-zha/r2_gaussian), which in turn is a derivative of the original 3D Gaussian Splatting implementation by Inria and MPII.

Because this repository contains derivative work, it is distributed under the identical **Gaussian-Splatting License**. 

* **The software is provided strictly for non-commercial, research, and evaluation purposes.**
* Any commercial use requires prior and explicit consent from the original licensors (Inria/MPII).

For more details, please refer to the `LICENSE` file included in this repository.