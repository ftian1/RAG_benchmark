#!/bin/bash
set -x

# KMP
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# OMP
export OMP_NUM_THREADS=144
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so

# tc malloc
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2" #"BAAI/bge-small-en-v1.5"
  count=1000000 #85000000
  batch_size=5000
  vector_database="chroma"
  search_type="mmr"
  max_length=512
  output_path="./db"

  for var in "$@"
  do
    case $var in
      --embedding_model=*)
          embedding_model=$(echo $var |cut -f2 -d=)
      ;;
      --count=*)
          count=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --vector_database=*)
          vector_database=$(echo $var |cut -f2 -d=)
      ;;
      --search_type=*)
          search_type=$(echo $var |cut -f2 -d=)
      ;;
      --max_length=*)
          max_length=$(echo $var |cut -f2 -d=)
      ;;
      --output_path=*)
          output_path=$(echo $var |cut -f2 -d=)
      ;;
    esac
  done

}

# run_tuning
function run_benchmark {

    numactl -l -C 0-143 python -m vectordb_benchmark  \
            --embedding_model ${embedding_model} \
            --count ${count} \
            --batch_size ${batch_size} \
            --vector_database ${vector_database} \
            --search_type ${search_type} \
            --max_length ${max_length} \
            --output_path ${output_path}

    rm -rf ./db/
}

main "$@"


