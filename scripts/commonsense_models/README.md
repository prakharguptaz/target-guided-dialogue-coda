# Code for sampling commonsense paths from conceptnet

Code is based on the repo https://github.com/wangpf3/Commonsense-Path-Generator/
Please read their Readme first. Use the conda environment setup from their repo.

There are two modifications: 1) We sample longer paths upto length of 6. 2) We sample paths of two types for kpg-wc and kpg-ht models a) with head, tail and intermediate entities specified, b) only head tail entities specified. We then train a model on both types of paths
We have provided samplepaths folder,please delete it if retraining the model and sampling the paths again.

## For sampling paths of the path generator
```bash
cd data
unzip conceptnet.zip
cd ..
python sample_path_rw.py
```

After path sampling, shuffle the resulting data './data/sample_path/sample_path.txt'
and then split them into train.txt, dev.txt and test.txt by ratio of 0.9:0.05:0.05 under './data/sample_path/'

Then you can start to train the path generator by running
```bash
# the first arg is for specifying which gpu to use
./run.sh $gpu_device
```

