
# Skribbl.ai CNN Training

This repository trains a CNN for the [Skribbl.ai](https://github.com/William-Gao1/skribbl-ai) web app.

The data used comes from the Quick, Draw! [dataset](https://quickdraw.withgoogle.com/data) provided by Google.

The training was done on a server using Slurm.
## Download Data

The notebook `download_data/download_data.ipynb` downloads the full raw data from Google Cloud Storage (180G).

Since we only need to train on a subset of the images, we extract 1000 images from each category. This is done in the `download_data/generate_subset.ipynb` notebook.
## Preprocessing

The format of the raw data (both during training and inference) is:

```
[
  [  // First stroke
    [x0, x1, x2, x3, ...],
    [y0, y1, y2, y3, ...],
    [t0, t1, t2, t3, ...]
  ],
  [  // Second stroke
    [x0, x1, x2, x3, ...],
    [y0, y1, y2, y3, ...],
    [t0, t1, t2, t3, ...]
  ],
  ... // Additional strokes
]
```

We want something more like a bitmap for our CNN. The `get_image_from_strokes()` function in the `preprocess/get_img.py` file does this conversion using PIL.

We use this `get_image_from_strokes()` function in the `preprocess/preprocess.ipynb` notebook to generate numpy bitmaps for each image that we extracted in the [Download Data](#download-data) step. We save these bitmaps in `.npy` files in preparation for training.
## Training

Training is done with the `train/run.py` file. It simply reads in the `.npy` images generated during the [Preprocessing](#preprocessing) stage, builds a CNN, and trains for 30 epochs.

After training has completed, we convert the model into a production-ready [TF-Lite](https://www.tensorflow.org/lite) model so that it can be used in the Skribbl.ai backend.
## Misc

The `predict.ipynb` file is to test that the model works and the `view_processed_bitmap.ipynb` file is to check that the bitmaps have been generated properly.