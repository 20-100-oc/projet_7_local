# To start the app, type in cmd (while being in the same dir as this file):
# uvicorn <filename_without_extention>:<app_variable_name> --reload
#
# Exemple:
# uvicorn api_test:my_app --reload
#
# Then use the streamlit app


import os
import torch
from torch import nn
from fastapi import FastAPI

from google.cloud import storage
from google.oauth2 import service_account
from tempfile import TemporaryFile

from transformers import DistilBertTokenizer
from transformers import DistilBertModel


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print('Downloaded to file: {destination_file_name}')



def main():
    device = 'cpu'
    class_names = ['negative', 'positive']
    model_name = 'distilbert-base-uncased'



    tokenizer = DistilBertTokenizer.from_pretrained(model_name)



    class Classifier(nn.Module):

        def __init__(self, nb_classes, max_nb_tokens):
            super(Classifier, self).__init__()
            self.distilbert = DistilBertModel.from_pretrained(model_name)
            self.flatten = nn.Flatten()
            fc_input_size = self.distilbert.config.hidden_size * max_nb_tokens
            self.fully_connected = nn.Linear(fc_input_size, nb_classes)
        
        def forward(self, input_ids, attention_mask):
            bert_output = self.distilbert(input_ids=input_ids, 
                                        attention_mask=attention_mask
                                        ).last_hidden_state
            bert_output_flattened = self.flatten(bert_output)
            classification_layer = self.fully_connected(bert_output_flattened)

            return classification_layer


    nb_classes = len(class_names)
    max_nb_tokens = 230
    model = Classifier(nb_classes=nb_classes, max_nb_tokens=max_nb_tokens)


    '''
    # get model localy
    root_dir = 'D:/OpenClassrooms/projet_7'
    model_weights_path = root_dir + '/data/models/' + model_name + '_weights_' + '3' + '.pt'
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))
    '''



    print('\n'*2, 'cwd:', os.getcwd(), '\n'*2)
    print('Loading model...')



    # get model from google cloud storage
    bucket_name = 'bucket_distilbert_1'
    model_bucket = 'distilbert-base-uncased_weights_3.pt'
    #json_key_path = 'omega-baton-363815-c529b1061a3a.json'

    credentials = service_account.Credentials.from_service_account_file(json_key_path)
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(model_bucket)

    with TemporaryFile() as temp_file:
        blob.download_to_file(temp_file)
        temp_file.seek(0)
        model.load_state_dict(torch.load(temp_file, map_location=torch.device(device)))
    print('Model ready.', '\n')


    # actual api
    sentiment_app = FastAPI()

    @sentiment_app.get('/predict_tweet')
    async def predict_sentiment(tweet: str):

        encoded_tweet = tokenizer.encode_plus(tweet, 
                                            max_length=max_nb_tokens, 
                                            add_special_tokens=True, 
                                            padding='max_length', 
                                            return_attention_mask=True, 
                                            return_token_type_ids=False, 
                                            return_tensors='pt'
                                            )

        input_ids = encoded_tweet['input_ids'].to(device)
        attention_mask = encoded_tweet['attention_mask'].to(device)

        output = model(input_ids, attention_mask)
        _, pred = torch.max(output, dim=1)
        proba = nn.functional.softmax(output, dim=1)

        sentiment = class_names[pred]
        proba_sentiment = round(float(proba[0][pred]), 3)

        res = {
            'tweet': tweet, 
            'sentiment': sentiment, 
            'probability': proba_sentiment, 
            }
        return res


if __name__ == '__main__':
    main()