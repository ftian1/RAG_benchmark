# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import faiss
import time
import argparse
import sys
import random

######################################################
# Command-line parsing
######################################################

rs = np.random.RandomState(123)
parser = argparse.ArgumentParser()

# from test_dataset import load_sift1M, evaluate
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    xb = fvecs_read("sift1M/sift_base.fvecs")
    xq = fvecs_read("sift1M/sift_query.fvecs")
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()
    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10
    return (t1 - t0) * 1000.0 / nq, recalls

def aa(*args, **kwargs):
    group.add_argument(*args, **kwargs)

group = parser.add_argument_group('benchmarking options')
aa('--raft_only', default=False, action='store_true',
   help='whether to only produce RAFT enabled benchmarks')
aa('--device', default='cpu', type=str,
   help='whether to do benchmarks on cpu or gpu device')
aa('--dummy', default=True, type=bool,
   help='whether to do benchmarks on dummy or real dataset')

group = parser.add_argument_group('IVF options')
aa('--bits_per_code', default=8, type=int, help='bits per code. Note that < 8 is only supported when RAFT is enabled')
aa('--pq_len', default=32, type=int, help='number of vector elements represented by one PQ code')
aa('--use_precomputed', default=True, type=bool, help='use precomputed codes (not with RAFT enabled)')

group = parser.add_argument_group('searching')
aa('--k', default=5, type=int, help='nb of nearest neighbors')
aa('--nprobe', default=200, type=int, help='nb of IVF lists to probe')
aa('--bm_search', default=True, type=bool)

args = parser.parse_args()

print("Loading data")
if args.dummy:
    print("Generate dummy 85M data...")
    time11 = time.time()
    xb = rs.rand(1000000, 768).astype('float32')
    xb = np.tile(xb, (85, 1))
    #xb = np.tile(xb, (85, 1))
    time22 = time.time()
    print("Generating dataset cost %.3f seconds" % (time22-time11))
    xq = rs.rand(10000, 768).astype('float32')
else:
    import numpy as np

    def read_fbin(fname):
        shape = np.fromfile(fname, dtype=np.uint32, count=2)
        if float(shape[0]) * shape[1] * 4 > 2000000000:
            data = np.memmap(fname, dtype=np.float32, offset=8, mode="r").reshape(
                shape
            )
        else:
            data = np.fromfile(fname, dtype=np.float32, offset=8).reshape(shape)
        return data
    
    printf("Generate real 88M data...")
    time11 = time.time()
    xb = read_fbin('./wiki_all_88M/base.88M.fbin')
    xq = read_fbin('./wiki_all_88M/queries.fbin')
    time22 = time.time()
    printf("Generating dataset cost %.3f seconds" % (time22-time11))

if args.device == 'gpu':
    import rmm
    res = faiss.StandardGpuResources()
    # Use an RMM pool memory resource for device allocations
    mr = rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
    rmm.mr.set_current_device_resource(mr)

def bench_train_milliseconds(index, trainVecs, use_raft):
    if args.device == 'gpu':
        co = faiss.GpuMultipleClonerOptions()
        # use float 16 lookup tables to save space
        co.useFloat16LookupTables = True
        co.use_raft = use_raft
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
        t0 = time.time()
        index_gpu.train(trainVecs)
        return 1000*(time.time() - t0)
    else:
        t0 = time.time()
        index.train(trainVecs)
        return 1000*(time.time() - t0)

def bench_add_milliseconds(index, addVecs, use_raft):
    if args.device == 'gpu':
        co = faiss.GpuMultipleClonerOptions()
        # use float 16 lookup tables to save space
        co.useFloat16LookupTables = True
        co.use_raft = use_raft
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
        index_gpu.copyFrom(index)
        t0 = time.time()
        index_gpu.add(addVecs)
        return 1000*(time.time() - t0)
    else:
        t0 = time.time()
        index.add(addVecs)
        return 1000*(time.time() - t0)

def bench_search_milliseconds(index, addVecs, queryVecs, nprobe, k, use_raft):
    if args.device == 'gpu':
        co = faiss.GpuMultipleClonerOptions()
        co.use_raft = use_raft
        co.useFloat16LookupTables = True
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index, co)
        index_gpu.copyFrom(index)
        index_gpu.add(addVecs)
        index_gpu.nprobe = nprobe
        t0 = time.time()
        index_gpu.search(queryVecs, k)
        return 1000*(time.time() - t0)
    else:
        index.nprobe = nprobe
        t0 = time.time()
        index.search(queryVecs, k)
        return 1000*(time.time() - t0)

trainset_sizes = [500000]
pq_lens = [16, 32, 64]
nprobes = [50, 100, 200]
nlists =  [4096, 8192]

for train_row in trainset_sizes:
    for pqlen in pq_lens:
        for probe in nprobes:
            for nlist in nlists:
                try:
                    idx = np.random.randint(xb.shape[0], size=train_row)
                    xt = xb[idx, :]
                    n_rows, n_cols = xb.shape
                    M = n_cols // pqlen

                    index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(nlist, M, args.bits_per_code))
                    print("The current group params is trainVec size: %d, pq_len: %d, n_centroids: %d, numSubQuantizers: %d, bitsPerCode: %d, nprobe: %d" % (
                        train_row, pqlen, nlist, M, args.bits_per_code, probe))
                    print("=" * 40)
                    print("{} Train Benchmarks".format(args.device))
                    print("=" * 40)
                    raft_gpu_train_time = bench_train_milliseconds(index, xt, True)
                    if args.raft_only or args.device == 'cpu':
                        print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, RAFT enabled GPU train time: %.3f milliseconds" % (
                            n_cols, nlist, M, args.bits_per_code, train_row, raft_gpu_train_time))
                    else:
                        classical_gpu_train_time = bench_train_milliseconds(
                            index, xt, False)
                        print("Method: IVFPQ, Operation: TRAIN, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numTrain: %d, classical GPU train time: %.3f milliseconds, RAFT enabled GPU train time: %.3f milliseconds" % (
                            n_cols, nlist, M, args.bits_per_code, train_row, classical_gpu_train_time, raft_gpu_train_time))

                    print("=" * 40)
                    print("{} Add Benchmarks".format(args.device))
                    print("=" * 40)
                    if args.device == 'gpu':
                        index.train(xt)
                    raft_gpu_add_time = bench_add_milliseconds(index, xb, True)
                    if args.raft_only or args.device == 'cpu':
                        print("Method: IVFPQ, Operation: ADD, dim: %d, n_centroids %d numSubQuantizers %d, bitsPerCode %d, numAdd %d, RAFT enabled GPU add time: %.3f milliseconds" % (
                            n_cols, nlist, M, args.bits_per_code, n_rows, raft_gpu_add_time))
                    else:

                        classical_gpu_add_time = bench_add_milliseconds(
                            index, xb, False)
                        print("Method: IVFFPQ, Operation: ADD, dim: %d, n_centroids %d, numSubQuantizers %d, bitsPerCode %d, numAdd %d, classical GPU add time: %.3f milliseconds, RAFT enabled GPU add time: %.3f milliseconds" % (
                            n_cols, nlist, M, args.bits_per_code, n_rows, classical_gpu_add_time, raft_gpu_add_time))

                    if args.bm_search:
                        print("=" * 40)
                        print("{} Search Benchmarks".format(args.device))
                        print("=" * 40)
                        queryset_sizes = [10000]
                        n_train, n_cols = xt.shape
                        n_add, _ = xb.shape
                        print(xq.shape)
                        M = n_cols // pqlen
                        if args.device == 'gpu':
                            index = faiss.index_factory(n_cols, "IVF{},PQ{}x{}np".format(nlist, M, args.bits_per_code))
                            index.train(xt)
                        for n_rows in queryset_sizes:
                            queryVecs = xq[np.random.choice(xq.shape[0], n_rows, replace=False)]
                            raft_gpu_search_time = bench_search_milliseconds(
                                index, xb, queryVecs, probe, args.k, True)
                            if args.raft_only or args.device == 'cpu':
                                print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, RAFT enabled GPU search time: %.3f milliseconds" % (
                                    n_cols, nlist, M, args.bits_per_code, n_add, n_rows, probe, args.k, raft_gpu_search_time))
                            else:
                                classical_gpu_search_time = bench_search_milliseconds(
                                    index, xb, queryVecs, probe, args.k, False)
                                print("Method: IVFPQ, Operation: SEARCH, dim: %d, n_centroids: %d, numSubQuantizers %d, bitsPerCode %d, numVecs: %d, numQuery: %d, nprobe: %d, k: %d, classical GPU search time: %.3f milliseconds, RAFT enabled GPU search time: %.3f milliseconds" % (
                                    n_cols, nlist, M, args.bits_per_code, n_add, n_rows, probe, args.k, classical_gpu_search_time, raft_gpu_search_time))
                except:
                    print("The current group params is Train size: %d, pq_len: %d, n_centroids: %d,  bitsPerCode: %d, nprobe: %d will have OOM issue" % (
                        train_row, pqlen, nlist, args.bits_per_code, probe))
                print("**"*40)
