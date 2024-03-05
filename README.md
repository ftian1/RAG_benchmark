Performance Benchmark on RAG embedding, indexing and search
===========================

## Env prepare

### Intel SPR

```shell
#install intel oneAPI
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh
sudo sh ./l_BaseKit_p_2024.0.1.46_offline.sh

#install conda/pip dependencies
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

### Nvidia A100/H100

```shell

conda install -c rapidsai -c conda-forge -c nvidia rmm cuda-version=12.1  #or 11.4
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0
```

## RAG indexing and search performance benchmark on Intel SPR


### Benchmark command

```shell

KMP_BLOCKTIME=1 KMP_SETTINGS=1 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=48 LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so numactl -l -C 0-47 python benchmark.py --device cpu 
```

## RAG indexing and search performance benchmark on Nvidia A100/H100

```shell

# benchmark cmd
python benchmark.py --device gpu
```

## Embedding Generation performance benchmark on Intel SPR

### Env prepare

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

### Benchmark command

```shell
source run.sh
```

