# Translation Model using Transformers (From Arabic - Italy).

## Overview
This project aims to build a translation model using Hugging Face's Transformers library. The model translates text from Arabic to Italian by fine-tuning the MarianMTModel. The project includes data preprocessing, model training, evaluation using BLEU score, and deployment using a Streamlit application for real-time translation.

## Project Structure

- **Data Preparation:** The dataset used for this project is `Helsinki-NLP/news_commentary`, which contains translation pairs for multiple language pairs. The data was split into training and testing sets.
  
- **Model and Tokenizer:** The MarianMTModel and MarianTokenizer were loaded from the Hugging Face model hub using the `Helsinki-NLP/opus-mt-ar-it` checkpoint.

- **Text Preprocessing:** Text data was tokenized and prepared for training using the `DataCollatorForSeq2Seq`.

- **Model Training:** The model was trained for 3 epochs with a batch size of 8. The training was optimized using a learning rate of `2e-5` and a weight decay of `0.01`.

- **Evaluation:** The model was evaluated using the BLEU score, a common metric for assessing the quality of machine-translated text.

- **Prediction:** A prediction function was created to translate text using the trained model.

- **Deployment:** The model was deployed using Streamlit, allowing users to input text and receive translations in real-time.

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install transformers datasets evaluate sacrebleu numpy streamlit
```

## Usage

### Training the Model

1. **Load the Dataset:**
   ```python
   ds = load_dataset("Helsinki-NLP/news_commentary", "ar-it")
   ```

2. **Preprocess the Data:**
   ```python
   def preprocessing(batch):
       # Add your preprocessing steps here
       return model_inputs

   ds = ds.map(preprocessing, batched=True)
   ```

3. **Train the Model:**
   ```python
   trainer = Seq2SeqTrainer(
       model=model,
       args=model_args,
       tokenizer=tokenizer,
       data_collator=data_collector,
       compute_metrics=compute_metrics,
       train_dataset=ds['train'],
       eval_dataset=ds['test']
   )
   trainer.train()
   ```

4. **Save the Model:**
   ```python
   trainer.save_model('/path/to/save/model')
   ```

### Making Predictions

Use the trained model to translate text:

```python
translated_text = translation_pipeline("مرحبا")
print(translated_text)
```

### Deploying with Streamlit

1. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

2. **Input Text:** Enter the text you want to translate in the Streamlit interface.

3. **Get Translation:** Click "Translate" to see the translated text.

## Evaluation

The model's performance is evaluated using the BLEU score. The `compute_metrics` function in the code computes this score by comparing the model's translations to the reference translations.

## Conclusion

This project showcases the power of Transformers for sequence-to-sequence tasks like translation. By fine-tuning a pre-trained MarianMTModel, we were able to create a high-quality translation model that performs well on the Arabic to Italian translation task.

## Future Work
Potential improvements could include experimenting with other language pairs, adjusting hyperparameters for better performance, or integrating the model into a larger application.
