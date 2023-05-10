import json
import pandas as pd
from pandas import json_normalize
import os
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import pathlib

def generate_jsonl(data, prediction_path, model_name, data_type):

    # input file type
    file_extension=pathlib.Path(data).suffix
    if file_extension =='.csv':
        # reading csv file
          raw_df = pd.read_csv(data)

    if file_extension =='.json':
          # reading json file
          raw_df = open(data)
          raw_df = json.load(raw_df)
          raw_df=json_normalize(raw_df['data'])

    if 'answers.text' in raw_df.columns:
        raw_df=raw_df[["context","question","answers.text"]]    
        raw_df['answer'] = raw_df['answers.text'].apply(lambda x:" ".join(x))
        raw_df = raw_df.drop(columns = ['answers.text'])
    else:
        raw_df=raw_df[["context","question"]] 
        
    # reading predicted file
    if data_type == "train" or data_type == "test":
      f= open(os.path.join(prediction_path, "predict_predictions.json"))
      data = json.load(f)
      predicted_answer=[]
      for x in data:
        predicted_answer.append(data[x])
    else:
      f = open(os.path.join(prediction_path, "eval_predictions.json"))
      data = json.load(f)
      predicted_answer=[]
      for x in data:
        predicted_answer.append(data[x])
        
    # creating dataframe
    final_df = raw_df.copy()
    final_df['predicted_answer'] = predicted_answer
    final_df['model_name'] = model_name
    final_df['data_type'] = data_type
    final_df.to_json(
      os.path.join(prediction_path, f"{data_type}_data_predictions_processed.jsonl"),orient="records",)

    if 'answer' in final_df.columns:
    # Benchmarking models
      bleu_score, rouge1, rougeL, semantic_similarity, _df_ = get_evaluation_metrics(final_df['answer'].tolist(), final_df['predicted_answer'].tolist())
      final_df['bleu_score'] = _df_['bleu_score'].tolist()
      final_df['rouge1'] = _df_['rouge1'].tolist()
      final_df['rougeL'] = _df_['rougeL'].tolist()
      final_df['sentence_similarity'] = _df_['sentence_similarity'].tolist()

      benchmark_df = pd.DataFrame({'model_name':[model_name],'type':[data_type], 'bleu_score':[bleu_score], 'rouge1':[rouge1], 
                                  'rougeL':[rougeL], 'semantic_similarity':[semantic_similarity]})
      
      benchmark_df.to_csv(os.path.join(prediction_path, f"{data_type}_data_benchmarks.csv"),index=False)


def get_sentence_similarity(reference = '', generated = '',  model_name = None):
  '''
  Generate cosine similarity score based on embeddings of two strings
  Parameters:
    reference (str) : Reference string to check similarity
    generated (str) : Generated/Target string to check similarity
    model_name (str) : Sentence tranformer model names
  Returns:
    Similarity score (float) : Cosine similarity score based on embeddings of the two strings
  '''
  if model_name == None:
    model = SentenceTransformer('all-minilm-l6-v2')
  else:
    model = SentenceTransformer(model_name)

  # convert to embeddings
  embedding1 = model.encode(reference, convert_to_tensor=True)
  embedding2 = model.encode(generated, convert_to_tensor=True)

  # compute similarity scores of two embeddings
  cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

  return cosine_scores.item() 

def get_bleu_score(reference, candidate):
  '''
  Function to get BLEU scores for two strings

  Parameters:
    reference (str) : Reference String
    candidate (str) : Candidate String

  Returns: 
    (float) : BLEU score
  '''
  candidate_ = candidate.split() 
  reference_ = []
  reference_.append(reference.split())
  return sentence_bleu(reference_, candidate_, weights=(1, 0, 0, 0))

def get_evaluation_metrics(actuals, predicted):
  '''
  Generate benchamrking scores on different metrics for generated text

  Parameters:
    actuals (str | list) : Actual text or reference
    predicted (str | list) : Generated text or predictions
  Returns:
    blue_score (float) : Mean BLUE score
    rouge1 (float): Mean ROUGE1 score
    rougeL (float): Mean ROUGEL score
    sentence similarity (float): Mean Cosine Similariy score on embeddedings
  '''
  if isinstance(actuals, list) and isinstance(predicted, list):
    df = pd.DataFrame({'actuals':actuals,'predicted':predicted})
  elif isinstance(actuals, str) and isinstance(predicted, str):
    df = pd.DataFrame({'actuals':[actuals],'predicted':[predicted]})

  scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

  df['bleu_score'] = df.apply(lambda x: get_bleu_score(x['actuals'], x['predicted']), axis = 1)
  df['rouge1'] = df.apply(lambda x: scorer.score(x['actuals'], x['predicted'])['rouge1'].fmeasure, axis = 1)
  df['rougeL'] = df.apply(lambda x: scorer.score(x['actuals'], x['predicted'])['rougeL'].fmeasure, axis = 1)
  df['sentence_similarity'] = df.apply(lambda x: get_sentence_similarity(reference = x['actuals'], generated = x['predicted']), axis = 1)

  return df['bleu_score'].mean(), df['rouge1'].mean(), df['rougeL'].mean(), df['sentence_similarity'].mean(), df

