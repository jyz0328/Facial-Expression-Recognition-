# Facial Expression Recognition Project (May Version)

**Authors:**  
Jingyi Zhang, Haoxian Ruan, Andrew Nguyen, and Adam Mhal  
**Emails:** jyz0328@bu.edu, rhx2000@bu.edu, aynguyen@bu.edu, adammhal@bu.edu
**some files are too huge to upload to github, so plz go to google drive to take a look: https://drive.google.com/drive/u/1/folders/1MdR9WIQ8Kbv_Aq7Bg2KntPREfKzp0Utv  ：** 


---

## Objective

This project aims to build accurate and generalizable models for facial expression recognition using the AffectNet dataset. We implemented and compared CNN and ResNet architectures, including improved variants. In addition, we explored a GAN-based approach to help the model better learn identity-invariant emotional features through expression synthesis.

---

## Structure

### I. `data/` Folder
**some data are too huge so that we cannot upload them via github. plz check GOOGLE DRIVE LINK ABOVE**
Contains all datasets used for training and evaluation:

- **affectnet/**
  - `X_train.npy` — (201347, 48, 48, 1)
  - `y_train.npy` — (201347, 8)
  - `X_val.npy` — (43155, 48, 48, 1)
  - `y_val.npy` — (43155, 8)
  - `X_test.npy` — (43148, 48, 48, 1)
  - `y_test.npy` — (43148, 8)
  - `preprocessed_data.zip`: archive of the above files

Each image is 48×48 grayscale. Labels are one-hot vectors for:
0 – Neutral, 1 – Happiness, 2 – Sadness, 3 – Surprise, 4 – Fear, 5 – Disgust, 6 – Anger, 7 – Contempt.

- `Expression_image.png` and `Reference_image.png`: input pairs for GAN model
- `previous_FER2013/`, `previous_data_for_GAN/`: legacy data (not used)

---

### II. `data_preprocess_code/` Folder

- `affectnetCSV.py`: creates structured CSV from raw dataset
- `process_AffectNet.py`: converts images to grayscale 48×48 `.npy` files
- `make_bbox_csv.py`: crops/resizes 224×224 RGB images for ResNet Improvement

---

### III. `MODEL_CODE/` Folder

Contains all implemented models:

- `check_emotion_amount.ipynb`: visualizes emotion distribution
- `CNN_baseline_Affectnet.ipynb`: CNN baseline
- `CNN_improvement_Affectnet.ipynb`: improved CNN
- `ResNet_baseline_Affectnet.ipynb`: ResNet baseline
- `ResNet_improvement_Affectnet.py`: improved ResNet with bbox
- `GANtransformEmotionsSCCAffectNet.py`: conditional GAN for expression transfer
- `previous_MODEL_CODE/`: legacy scripts (not used)

**Note:**
- All models except ResNet Improvement use `.npy` files from `data/affectnet/`.
- The GAN also uses `Expression_image.png` and `Reference_image.png`.
- ResNet Improvement requires running `make_bbox_csv.py` to generate cropped CSV input.

---

### IV. `FIGURE_and_TABLE_RESULT/` Folder

Contains all final results:

#### Accuracy & Loss Plots
- `CNN_Baseline_Affectnet_loss_and_accuracy_plot.png`
- `CNN_Improvement_Affectnet_loss_and_accuracy_plot.png`
- `ResNet_Baseline_Affectnet_loss_and_accuracy_plot.png`
- `ResNet_Improvement_Affectnet_loss_and_accuracy_plot.png`

#### Confusion Matrices (Test Set)
- `CNN_Baseline_Affectnet_confusion_matrix.png`
- `CNN_Improvement_Affectnet_confusion_matrix.png`
- `ResNet_Baseline_Affectnet_confusion_matrix.png`
- `ResNet_Improvement_Affectnet_confusion_matrix.png`

#### Excel Table
- `All_Result_Tables.xlsx`:  
  (a) Label distribution  
  (b) Loss/Accuracy comparison  
  (c) Per-class accuracy (diagonals)

#### GAN Results (`GAN_result/`)
- `GAN_Combined_Loss.png`  
- `GAN_Discriminator_Accuracy.png`  
- `GAN_Generator_Loss.png`  
- `Inference_Image.png`

#### `previous_figure_result/`: legacy results (not used)

---

### V. `SAVED_MODEL/` Folder
- **some data are too huge so that we cannot upload them via github. plz check GOOGLE DRIVE LINK ABOVE**
Saved model checkpoints:

- `AffectNet_CNN_Baseline_best_model.h5`
- `AffectNet_CNN_Improvement_model.h5`
- `AffectNet_ResNet_Baseline_best_model.h5`
- `AffectNet_ResNet_Improvement_best_model.h5`
- `best_GAN_generator.pth`

Folder also contains:
- `previous_models_result/`: legacy models (not used)

---

## How to Run the Code

### CNN / Improved CNN / ResNet Baseline

- Open corresponding `.ipynb` file in `MODEL_CODE/`
- Run all cells  
- Uses `.npy` input from `data/affectnet/`  
- Outputs saved in `FIGURE_and_TABLE_RESULT/` and `SAVED_MODEL/`

---

### ResNet Improvement

1. Run `affectnetCSV.py` to generate CSV  
2. Run `make_bbox_csv.py` for cropped/resized data  
3. Update input path in `ResNet_improvement_Affectnet.py`  
4. Run script  
5. Outputs saved to `FIGURE_and_TABLE_RESULT/` and `SAVED_MODEL/`

---

### GAN Expression Transfer

- Run `GANtransformEmotionsSCCAffectNet.py`  
- Requires:
  - `.npy` data from `data/affectnet/`
  - `Expression_image.png` and `Reference_image.png`
- Outputs include synthesized image and loss plots in `GAN_result/`

---

For questions or reproducibility, please refer to the code or contact the authors.
