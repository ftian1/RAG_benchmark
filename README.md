Benchmark on RAG performance with Intel SPR
===========================

## Installation on Ubuntu

Please ensure a `conda` env has been ready and follow below steps to create bechmarking env.

```shell
conda install astunparse ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install jemalloc gperftools -c conda-forge -y
conda install git-lfs -y

pip install -U langchain
pip install -U langchain-community


python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

pip install sentence_transformers

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

pip install intel-extension-for-transformers
pip install accelerate
pip install datasets

```

## Performance calcuation

Execute below cmd in shell, you will see logs in the terminal.

```shell
source run.sh
```


