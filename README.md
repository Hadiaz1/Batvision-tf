# Batvision-tf

## ✨ Unofficial Tensorflow Implementation for the BatVision dataset

### 📝 Paper title: [The Audio-Visual BatVision Dataset for Research on Sight and Sound](https://arxiv.org/abs/2303.07257)
### 🗃️ Datasets: [BatvisionV1 and BatvisionV2](https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M) 
*Note: Currently Only V2 is supported in this repository*
### 🥅 The objective is to ingest the dataset and to benchmark a model zoo, with variable complexities, against the Cycle GAN UNet used by the authors as a high performing baseline

## Usage
1. Clone the repository:
```cmd 
git clone https://github.com/Hadiaz1/Batvision-tf.git
```
2. Navigate to the project directory:
```cmd 
cd Batvision-tf
```
3. Install the required dependencies:
```cmd 
pip install -r requirements.txt
```
4. Train and test a model from the tuned params.yaml:
```cmd 
python main.py
```


## Results
| Model Name        | Test Mean Absolute Error |
|-------------------|-------------------------|
| simple_UNet       |                         |
| MobileNetv2_UNet  | 0.1055                  |
| ResAttention_UNet | 0.1047                  |
| Transformer_UNet  |                         |