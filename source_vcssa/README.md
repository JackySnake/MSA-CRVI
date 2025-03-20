**NeurIPS 2024** 

PyTorch implementation of the paper **[Infer Induced Sentiment of Comment Response to Video: A New Task, Dataset and Baseline](https://proceedings.neurips.cc/paper_files/paper/2024/file/bbf090d264b94d29260f5303efea868c-Paper-Datasets_and_Benchmarks_Track.pdf)**


### Environement

Create a Python 3.8 environment with:
```
torch              1.13.1      
numpy              1.22.0
scikit-learn       1.2.1
transformers       4.26.1
easydict           1.10
```

### Training

We provide two training scripts in `script` folder:
- Single GPU: `main.py`
- Multi-GPU: `main_multigpu.py`


**Example usage (Multi-GPU on CSMV dataset):**
```bash
cd script
sh train_multigpu.sh
```

**Important Notes:**
- Training parameters are configured in the shell files
- Checkpoints are saved after each epoch in `ckpt/mymodel/`
- Automatic validation is performed after each epoch to track best performance

### Evaluation 

To evaluate the top-performing checkpoint (configure top-k in `main_eval.py`), detecting the checkpoints in `ckpt/mymodel/`:
```bash
cd script
sh eval.sh
```


### License

The source code for the site is licensed under the MIT license, which you can find in the LICENSE file.

### Contact

For questions or feedback, please contact [Qi Jia](https://github.com/JackySnake) on github.

### Acknowledgement

This implementation references [MOSEI_UMONS](https://github.com/jbdel/MOSEI_UMONS).

## Paper Citation

```bibtex
@inproceedings{DBLP:conf/nips/0004FXLJD0Z0L24,
  author       = {Qi Jia and
                  Baoyu Fan and
                  Cong Xu and
                  Lu Liu and
                  Liang Jin and
                  Guoguang Du and
                  Zhenhua Guo and
                  Yaqian Zhao and
                  Xuanjing Huang and
                  Rengang Li},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {Infer Induced Sentiment of Comment Response to Video: {A} New Task,
                  Dataset and Baseline},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/bbf090d264b94d29260f5303efea868c-Abstract-Datasets\_and\_Benchmarks\_Track.html},
  timestamp    = {Thu, 13 Feb 2025 16:56:44 +0100},
  biburl       = {https://dblp.org/rec/conf/nips/0004FXLJD0Z0L24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
