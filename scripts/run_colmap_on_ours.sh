#!/bin/sh

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_path> <sparse_only_bool>"
    exit 1
fi
cwd=$(pwd)
root=$1
sparse_only=$2

# for i in {5..300..40}
# for i in {0..299}
# do
i=45
    echo "Processing $root/colmap_$i/ ..."
    cd "$root/colmap_$i/" || exit
    if [ -d "true_colmap" ]; then
        echo "true_colmap already exists, deleting..."
        rm -rf true_colmap
    fi
    if [ -d "tmp" ]; then
        echo "tmp already exists, deleting..."
        rm -rf tmp
    fi
    cd "$cwd" || exit
    rm -rf "$root/colmap_$i/sparse/0/points3D_colmap.ply" || exit
    rm -rf "$root/colmap_$i/sparse/0/points3D_colmap_dense.ply" || exit
    if [ "$sparse_only" -eq "1" ]; then
        echo "Running COLMAP with sparse-only option..."
        python scripts/colmap_extraction.py "$root/colmap_$i" --sparse-only
    else
        echo "Running COLMAP with dense option..."
        python scripts/colmap_extraction.py "$root/colmap_$i"
    fi
# done

