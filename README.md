# README

To generate noisy hanzi character dataset used in [Noise2Self](https://arxiv.org/abs/1901.11365), run the script `generate.py`.

It will generate training, test, and validation tiles, each stored as a `.npy` file. Each data point is 9 64x64 tiles; the ground truth, then four copies of the image at increasing noise levels, then another four at the same noise levels (for noise2noise training).

To generate a small dataset, cap the number of characters.

`python generate.py --max_chars 128`

By default, it will produce 6 noisy training data points for each character. To
modify that multiplicity, use the `--multiplicity` flag.
