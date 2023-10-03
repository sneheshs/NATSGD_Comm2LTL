import pandas as pd
import numpy as np
import torch
from transformers import BartTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW

from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, train_test_split
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
import os

class HiddenLSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, device):
        super(HiddenLSTMEncoderDecoder, self).__init__()

        self.num_layers = 2

        self.device = device
        # Encoder
        self.encoder = nn.LSTM(input_dim, latent_dim, num_layers=self.num_layers, batch_first=True)

        # Decoder
        self.decoder = nn.LSTM(input_dim, latent_dim, num_layers=self.num_layers, batch_first=True)

        self.hidden_to_embed = nn.Linear(latent_dim, input_dim)

    def _unflatten(self, x):
        return x.view(self.batch_size, self.num_layers, latent_dim).transpose(0, 1).contiguous()

    def _flatten(self, h):
        return h.transpose(0, 1).contiguous().view(self.batch_size, -1)

    def _unflatten_hidden(self, x):
        x_split = torch.split(x, int(x.shape[1] / 2), dim=1)
        h = (self._unflatten(x_split[0]), self._unflatten(x_split[1]))
        return h

    def _init_hidden_state(self, encoder_hidden):
        return tuple([self._concat_directions(h) for h in encoder_hidden])

    def _concat_directions(self, hidden):
        return hidden

    def _step(self, input, hidden):
        output, hidden = self.decoder(input, hidden)

        output = self.hidden_to_embed(output.squeeze())
        return output, hidden

    def forward(self, x):
        x = x.to(self.device)
        self.batch_size, seq_len, features = x.size()
        _, (hidden, cell) = self.encoder(x)

        z = torch.cat([self._flatten(hidden), self._flatten(cell)], 1)

        # initialize the hidden state of the decoder
        hidden = self._unflatten_hidden(z)
        hidden = self._init_hidden_state(hidden)

        outputs = torch.zeros((self.batch_size, seq_len, features)).to(self.device)

        input = x[:, -1:, :]
        for i in range(seq_len):
            output, hidden = self._step(input, hidden)
            outputs[:, i:i + 1, :] = output.unsqueeze(1)
            input = x[:, i:i + 1, :]

        return outputs

class tensorTokenizer:
    def __init__(self, tokenizer, max_seq_length=512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def tokenize_tensor(self, tensor):
        # Convert flattened tensor elements to a list of values
        values = tensor.tolist()

        # Convert values to tokens as strings
        tokens = [str(val) for sublist in values for val in sublist]

        return tokens

def preprocess_data(data, tokenizer, motion_model, max_seq_length=512):
    tensor_tokenizer = tensorTokenizer(tokenizer, max_seq_length = 512)
    input_ids = []
    attention_masks = []
    target_ids = []
    references = []
    source_ges = []
    target_texts = []
    for index, row in data.iterrows():

        # Convert gestures to a tensor and move it to CUDA
        ges = row["Gesture"]
        source_ges.append(list(ges))  # Append gestures as a list

        ges_length = torch.tensor(len(ges))
        ges = ges.float().unsqueeze(0).to('cuda:0')  # Ensure it's 4D (B, F, J, C)

        ges = ges.view(1,ges.shape[1], -1)

        packed_input = nn.utils.rnn.pack_padded_sequence(ges, [ges_length], batch_first=True,
                                                         enforce_sorted=False).to(device)

        ges, _ = motion_model.encoder(packed_input)
        tokens = tensor_tokenizer.tokenize_tensor(ges.data)

        # Encode gestures and speech tokens separately
        encoded_input_ges = tokenizer.encode_plus(
            tokens,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt")


        input_ids.append(encoded_input_ges["input_ids"])
        attention_masks.append(encoded_input_ges["attention_mask"])

        target_text = row["LTL"]
        target_texts.append(target_text)

        encoded_target = tokenizer.encode_plus(
                target_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
        target_ids.append(encoded_target["input_ids"])
        references.append([target_text])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    target_ids = torch.cat(target_ids, dim=0)


    return input_ids, attention_masks, target_ids, source_ges, target_texts

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0



def calculate_cosine_similarity(predictions, references):

    # Ensure predictions and references have the same length
    if len(predictions) != len(references):
        raise ValueError("Input lists must have the same length")

    # Join the lists of strings into single strings
    predictions = [" ".join(prediction) for prediction in predictions]
    references = [" ".join(reference) for reference in references]

    # Initialize a TF-IDF vectorizer with text preprocessing
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, tokenizer=str.split)

    # Fit and transform the predictions and references
    prediction_vectors = tfidf_vectorizer.fit_transform(predictions)
    reference_vectors = tfidf_vectorizer.transform(references)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(prediction_vectors, reference_vectors)

    return similarity_scores


dataset = np.load('data/NatSGD_v1.0.npz', allow_pickle=True)
data = dataset['data']
cols = dataset['fields']
data = pd.DataFrame(data[:, [3,4,6]], columns=['Speech', "Gesture", "LTL"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

motion_model_path =os.getcwd()+'/data/lstmEncoder_22.pth'
motion_model = torch.load(motion_model_path).to(device)
motion_model.eval()

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model and data to GPU if available
torch.cuda.empty_cache()

model.to(device)

#Scores file path and write headers
modality = 'gestures'
model_ver = 'T5'
train_scores_file =os.getcwd()+ '/data/results/scores/gesturesT5/train/train_score_' + modality +'-'+ model_ver + '.txt'
test_scores_file =os.getcwd()+ '/data/results/scores/gesturesT5/test/test_score_' + modality +'-'+ model_ver + '.txt'
with open(train_scores_file, 'a') as f:
    f.write("train_jaq_similiarity_score, train_cosine_similarity\n")
with open(test_scores_file, 'a') as f:
    f.write("test_jaq_similiarity_score, test_cosine_similarity\n")

# Lists to store test losses for each random state
test_loss_lis = []
i  =  0
latent_dim = 512  # List of hidden dimensions for each layer
input_dim = 51 ###17x3


# for random_state in random_states:
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

batch_size = 4

input_ids, attention_masks, target_ids, train_ges, train_target = preprocess_data(train_data, tokenizer, motion_model, max_seq_length=512)
train_data_tensor = TensorDataset(input_ids, attention_masks, target_ids)
train_loader = DataLoader(train_data_tensor, batch_size=batch_size, shuffle=False)

test_input_ids, test_attention_masks, test_target_ids, test_ges, test_target = preprocess_data(test_data, tokenizer, motion_model, max_seq_length=512)
test_data_tensor = TensorDataset(test_input_ids, test_attention_masks, test_target_ids)
test_loader = DataLoader(test_data_tensor, batch_size=batch_size, shuffle=False)

num_epochs = 100
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
train_loss_lis = []
test_loss_lis = []
train_data_by_epoch = []
test_data_by_epoch = []
train_score_lis = []
test_score_lis = []

for epoch in range(num_epochs):
    train_data = []
    test_data = []

    total_loss = 0
    train_predictions = []
    train_references = []
    model.train()
    for batch in train_loader:
        input_ids_batch, attn_mask_batch, target_ids_batch = batch
        input_ids_batch, attn_mask_batch, target_ids_batch = \
            input_ids_batch.to(device), attn_mask_batch.to(device), target_ids_batch.to(device)

        outputs = model(input_ids=input_ids_batch, attention_mask=attn_mask_batch, labels=target_ids_batch)
        loss = outputs.loss
        loss = loss.mean()
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # Convert the predicted token IDs to text and add them to the predictions list
        generated_ids = torch.argmax(outputs.logits, dim=-1)  # Get the token IDs with the highest probability
        batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        train_predictions.extend(batch_predictions)

        # Convert the ground-truth token IDs to text and add them to the references list
        target_texts = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
        train_references.extend([[text] for text in target_texts]) 
    average_loss = total_loss / len(train_loader)
    train_loss_lis.append(average_loss)

    train_jac_scores = [jaccard_similarity(set(" ".join(pred).split()), set(" ".join(ref).split())) for pred, ref in zip(train_predictions, train_references)]

    train_mean_jaq_similiarity_score = np.mean(train_jac_scores)
    train_mean_cosine_similarity = calculate_cosine_similarity(train_predictions, train_references).mean()

    train_score_lis.append((train_mean_jaq_similiarity_score,train_mean_cosine_similarity))

    for i in range(len(train_ges)):
        train_data.append((train_ges[i], train_target[i], train_predictions[i], train_references[i]))
    train_data_by_epoch.append(train_data)




    test_loss = 0
    test_predictions = []
    test_references = []
    model.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            test_input_ids_batch, test_attn_mask_batch, test_target_ids_batch = test_batch
            test_input_ids_batch, test_attn_mask_batch, test_target_ids_batch = \
                test_input_ids_batch.to(device), test_attn_mask_batch.to(device), test_target_ids_batch.to(device)

            test_outputs = model(input_ids=test_input_ids_batch, attention_mask=test_attn_mask_batch, labels=test_target_ids_batch)
            test_loss = test_outputs.loss
            test_loss = test_loss.mean()
            total_test_loss = test_loss.item()

            generated_ids_test = torch.argmax(test_outputs.logits, dim=-1)
            test_batch_predictions = tokenizer.batch_decode(generated_ids_test, skip_special_tokens=True)
            test_predictions.extend(test_batch_predictions)

            target_texts_test = tokenizer.batch_decode(test_target_ids_batch, skip_special_tokens=True)
            test_references.extend([text for text in target_texts_test])

    average_test_loss = total_test_loss / len(test_loader)
    test_loss_lis.append(average_test_loss)


    test_jac_scores = [
        jaccard_similarity(set(" ".join(pred).split()), set(" ".join(ref).split()))
        for pred, ref in zip(test_predictions, test_references)
    ]

    test_mean_jaq_similiarity_score = np.mean(test_jac_scores)
    test_mean_cosine_similarity = calculate_cosine_similarity(test_predictions, test_references).mean()
    test_score_lis.append((test_mean_jaq_similiarity_score,test_mean_cosine_similarity))

    for i in range(len(test_ges)):
        test_data.append((test_ges[i], test_target[i],  test_predictions[i], test_references[i]))

    test_data_by_epoch.append(test_data)


    print(f'Epoch {epoch+1} Train Loss: {average_loss} Test Loss {average_test_loss} Test Jaccard Score {test_mean_jaq_similiarity_score} Test Cosine Similarity: {test_mean_cosine_similarity}')

train_data_directory =os.getcwd()+ '/data/results/predictions/gesturesT5/train/'
test_data_directory =os.getcwd()+ '/data/results/predictions/gesturesT5/test/'

train_data_paths = [f'{train_data_directory}train_data_epoch_{epoch + 1}.txt' for epoch in range(num_epochs)]
test_data_paths = [f'{test_data_directory}test_data_epoch_{epoch + 1}.txt' for epoch in range(num_epochs)]

for epoch, epoch_train_data in enumerate(train_data_by_epoch):
    with open(train_data_paths[epoch], 'w') as train_file:
        for data_point in epoch_train_data:
            train_file.write(f'Gesture: {data_point[0]}\n')
            train_file.write(f'LTL: {data_point[1]}\n')
            train_file.write(f'Predictions: {data_point[2]}\n')
            train_file.write(f'References: {data_point[3]}\n\n')

for epoch, epoch_test_data in enumerate(test_data_by_epoch):
    with open(test_data_paths[epoch], 'w') as test_file:
        for data_point in epoch_test_data:
            test_file.write(f'Gesture: {data_point[0]}\n')
            test_file.write(f'LTL: {data_point[1]}\n')
            test_file.write(f'Predictions: {data_point[2]}\n')
            test_file.write(f'References: {data_point[3]}\n\n')

model_file =os.getcwd()+ '/data/results/gestures_t5.pth'
torch.save(model, model_file)

with open(train_scores_file, 'w') as file:
    # Write each element of the list to the file
    for item in train_score_lis:
        file.write(str(item) + '\n')

with open(test_scores_file, 'w') as file:
    # Write each element of the list to the file
    for item in test_score_lis:
        file.write(str(item) + '\n')
