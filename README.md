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
- Test our models or transcribe new speech...
  - ...by using **Greedy Decoding**
  ```
    python3 transcribe_speech.py model_path=models/stt_itUniBO_quartznet15x5.nemo dataset_manifest=TCorpora/cv-corpus-7.0-2021-07-21_test.json 
  ```
  - ...by applying **Beam-Search Decoding & N-gram Rescoring**
  ```
    python3 eval_beamsearch_ngram.py --nemo_model_file models/stt_itUniBO_quartznet15x5.nemo --input_manifest TCorpora/cv-corpus-7.0-2021-07-21_test.json --kenlm_model_file models/6gramLM_CORIS165C.kenlm --decoding_mode beamsearch_ngram --beam_width 1024 --beam_alpha 1.0 --beam_beta 0.5 
  ```
  - ...by applying **Beam-Search Decoding & Neural Rescoring**
  ```
    python3 eval_neural_rescorer.py --lm_model=models/TransformerLM_CORIS165C_e36.nemo --beams_file=BEAM_1024_1_0.5/preds_out_width512_alpha1.0_beta0.5.tsv --beam_size=1024 --eval_manifest=TCorpora/cv-corpus-7.0-2021-07-21_test.json 
  ```

### Re-Training the entire model
To re-train the Italian ASR model from scratch, you have to download the standard **stt_en_quartznet15x5.nemo v1.0.0rc1 published the 30th June 2021**, adjust the parameters into the train_QuartzNet.py script and then execute
```
python3 train_QuartzNet.py
```
In case of problems contact me at <fabio.tamburini@unibo.it>.

### Acknowledgements
All the scripts are based on those released by NVIDIA or on some tutorial from NVIDIA scholars.

### Citation

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
