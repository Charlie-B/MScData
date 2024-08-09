#RoBERTa

import torch
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained('google-bert/bert-base-uncased')


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Load the dataset into a pandas dataframe.
wd = "mnt/parscratch/users/$USER/anaconda/.envs/test/"
read_location = wd + "SentencesShort.csv"

input_doc = pd.read_csv(read_location, header=0)

year_list = [2024, 2014, 2004, 1994, 1984]

for year in year_list:

    input_doc_filtered = input_doc[input_doc["year"] == year]
    sentences = input_doc_filtered.text.values

    # Getting the max token length for every sentence
    max_len = 0

    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print(max_len)

    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_len,
            padding='max_length',  # Pad all sentences to the max length
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Run the model!
    with torch.no_grad():
        output = model(input_ids, attention_masks, output_hidden_states=True)

    hs = output[-1]
    pen_hs = hs[-2]

    hs_loc = wd + "hsShort" + str(year) + ".pt"
    torch.save(pen_hs,hs_loc)

    decoded_tokens_list = []
    for i in range(len(sentences)):
        # Extract token IDs for the current sentence
        encoded_sent = input_ids[i]

        # Convert tensor to list
        encoded_sent_list = encoded_sent.squeeze().tolist()

        # Convert token IDs to tokens
        tokens = tokenizer.convert_ids_to_tokens(encoded_sent_list)
        decoded_tokens_list.append(tokens)

    data = []
    for i, tokens in enumerate(decoded_tokens_list):
        for j, token in enumerate(tokens):
            embedding = pen_hs[i, j, :]
            data.append([token, embedding])

    df_embeddings = pd.DataFrame(data, columns=['token', 'embedding'])

    df_filtered = df_embeddings.loc[df_embeddings.token != '<pad>']
    df_filtered = df_filtered.loc[df_filtered.token != ',']

    df_filtered['token'] = df_filtered['token'].copy().str.replace('Ġ', '', regex=False)

    # Find average embedding per token
    averaged_df = df_filtered.groupby('token').agg(
        count=('token', lambda x: len(x)),
        mean=('embedding', lambda x: np.vstack(x).mean(axis=0).tolist()),
        token=('token', 'first')
    )

    av_loc = wd + "avShort" + str(year) + ".csv"
    averaged_df.to_csv(av_loc)
