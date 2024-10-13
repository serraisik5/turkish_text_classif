# Hepsiburada Data Scientist Case Study

This repository contains machine learning (ML) and large language model (LLM) solutions for a text classification task. The project focuses on the preprocessing, training, evaluation, and inference of SVM and fine-tuned BERT model.

## Project Structure

data/: script for data preparation and splitting.
ML_model/: scripts for preprocessing, training, and evaluation of the SVM model.
LLM_model/: scripts for preprocessing, fine-tuning, and evaluating a BERT model (BertTurk).
models/: Contains saved models, including one pre-trained model trained on full dataset for 3 epochs in the cloud.

## Building and Running the Docker Image 

docker build -t case_image .
docker run -it case_image

## Data Preparation

The split_data.py divides the dataset into training, validation, and test CSV files. You can optionally set small_sample=True to use a smaller dataset for experimentation.

python data/split_data.py --small_sample

## Machine Learning Model (SVM)
The ML model uses an SVM for text classification. To preprocess the data, train the model, and evaluate its performance, run:

python ML_model/main_SVM.py

For inference:
python ML_model/inference.py

## Large Language Model (BERT)
The LLM model fine-tunes a Turkish BERT model with a specialized tokenizer that includes added tokens. To preprocess, train, and evaluate the model, run:

python LLM_model/main_LLM.py

For inference:
python LLM_model/inference.py "sentence to classify"
example: python LLM_model/inference.py "Türkçe sondan eklemeli bir dildir."

### Models
The models/ directory contains the trained models. You can find the final model of fine-tuned BertTurk model, trained on the full dataset for 3 epochs in the Collab, which can be used for inference and evaluation.



#### Additional Content 

Ollama Model for Llama3 Classification

In the `Ollama_model` directory, you can find the code that implements prompt engineering for the Llama3 model for a classification task.

To create the model:
ollama create lamaturk -f ./Ollama_model/Modelfile

To run the task: 
python Ollama_model/Ollama_main.py