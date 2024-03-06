Performance Benchmark on RAG embedding, indexing and search
===========================

## RAG indexing and search performance benchmark on Intel SPR

### Env prepare

The RAG indexing and search time is measured based on `faiss` package. The `faiss` default binary can't bring good performance on Intel Xeon platforms. It requires us to manual build `faiss` cpu version from source.

```shell

#install intel oneAPI
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46_offline.sh
sudo sh ./l_BaseKit_p_2024.0.1.46_offline.sh

#source mkl related env variables
source /opt/intel/oneapi/mkl/latest/env/vars.sh

#install dependency
conda install -c conda-forge swig=4.1.1

#build faiss-cpu w/ avx512 support
git clone https://github.com/facebookresearch/faiss.git
mkdir build && cd build
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_RAFT=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_OPT_LEVEL=avx512 -DBLA_VENDOR=Intel10_64_dyn ..
make -j faiss && make -j swigfaiss
cd faiss/python/ && pip install -e .
make install
```

### Benchmark command

```shell

KMP_BLOCKTIME=1 KMP_SETTINGS=1 KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=48 LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so numactl -l -C 0-47 python benchmark.py --device cpu 
```

## RAG indexing and search performance benchmark on Nvidia A100/H100

### Env prepare

```shell

conda install -c rapidsai -c conda-forge -c nvidia rmm cuda-version=12.1  #or 11.4
conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0
```

### Benchmark command

```shell

python benchmark.py --device gpu
```

## Embedding Generation performance benchmark on Intel SPR

### Env prepare

The `embedding generation` time is measured based on langchain RAG API with Intel SPR 1s48c

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

