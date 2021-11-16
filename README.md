# ItaNeMoASR
Scripts and models for setting up an Italian ASR system based on NVIDIA NeMo.

This scripts have been tested on:
- Python 3.6.8
- NumPy 1.19.5
- PyTorch 1.9.0
- NVIDIA NeMo 1.1.0

For reproducing our results:
- Clone the repository.
- Download our models from http://corpora.ficlit.unibo.it/UploadDIR/ItaNeMoASR_models.tar.gz
- Extract model files.
- Download the dataset listed in the paper from their websites.
- Test our models or transcribe new speech:
  - With *Greedy Decoding*
```
    python3 
```


In case of problems contact me at <fabio.tamburini@unibo.it>.

## Citation

If you use my work, please cite:
```tex
@InProceedings{Tamburini2021,
  author = {Tamburini, Fabio},
  title = {{Playing with NeMo for building anAutomatic Speech Recogniser for Italian}},
  booktitle = {{Proceedings of the 7th Italian Conference on Computational Linguistics - CLIC-it 2021}},
  year = 	{2021},
  publisher = {CEUR-WS XXXX},
  location = {Milan, Italy},
  url = 	{http://}
}
```
