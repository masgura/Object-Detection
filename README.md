# Object detection in the SPEED+ dataset

This repository contains all the files for detecting the Tango spacecraft in the SPEED+ dataset.

# Dataset creation

Once SPEED+ is downloaded it needs to be rearrenged in a correct format.

Original format:

| SYNTHETIC
  | IMAGES
      Synthetic images used for training and validation
| SUNLAMP
  | IMAGES
      Real images containing sunlamps used for testing
| LIGHTBOX
  | IMAGES
      Real images containing lightboxes used for testing

New format:
| SYNTHETIC
  | IMAGES
    | TRAIN
        Synthetic images used for training
    | VAL
        Synthetic images used for validation
  | LABELS
    | TRAIN
        .txt files with training labels
    | VAL
        .txt files with validation labels
| SUNLAMP
  | IMAGES
      Real images containing sunlamps used for testing
  | LABELS
      .txt files with sunlamp testing labels
| LIGHTBOX
  | IMAGES
      Real images containing lightboxes used for testing
  | LABELS
      .txt files with lightbox testing labels
