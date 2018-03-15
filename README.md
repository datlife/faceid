# Face ID
A minimal implementation of [Face Recognition System](https://en.wikipedia.org/wiki/Facial_recognition_system) using Deep Learning and [Batch Hard Triplet Loss](https://arxiv.org/pdf/1703.07737.pdf)


Project status: **Under development!**


## Disclaimer
* This project is NOT an official method Apple uses, rather an attempt to reproduce their results. 
* Apple may trained  their model on private, non open-sourced, dataset. In this project, we use open-sourced dataset available on the Internet.



## Usage

* Define a new `DataProvider`, depending on source of dataset
* Use `trainer.py` to pre-train, which may requires to update new `parse_fn` and `preprocess_fn`.

