import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

model.to(device)

# Load and preprocess CSV data

def preprocess_csv_data(data, max_seq_length):

    # Initialize BART tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # Lists to store tokenized input and target sequences
    input_ids = []
    attention_masks = []
    target_ids = []
    references = []
    source_texts =  []
    target_texts = []

    # Process each row in the DataFrame
    for index, row in data.iterrows():
        # Tokenize source (input) text
        source_text = row["Speech"]
        source_texts.append(source_text)
        encoded_input = tokenizer.encode_plus(source_text, max_length=max_seq_length,
                                              padding="max_length", truncation=True,
                                              return_tensors="pt")
        input_ids.append(encoded_input["input_ids"])
        attention_masks.append(encoded_input["attention_mask"])

        # Tokenize target (translation) text
        target_text = row["LTL"]
        target_texts.append(target_text)
        encoded_target = tokenizer.encode_plus(target_text, max_length=max_seq_length,
                                               padding="max_length", truncation=True,
                                               return_tensors="pt")
        target_ids.append(encoded_target["input_ids"])
        references.append([target_text])

    # Convert lists to PyTorch tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    target_ids = torch.cat(target_ids, dim=0)

    return input_ids, attention_masks, target_ids, source_texts, target_texts



dataset = np.load('data/NatSGD_v1.0.npz', allow_pickle=True)
data = dataset['data']
cols = dataset['fields']
data = pd.DataFrame(data[3,6], columns=["Speech", "LTL"])

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42) # Split data into training and validation sets

batch_size = 8

input_ids, attention_masks, target_ids, train_source, train_target = preprocess_csv_data(train_data, max_seq_length = 512)
train_data_tensor = TensorDataset(input_ids, attention_masks, target_ids)
train_loader = DataLoader(train_data_tensor, batch_size=batch_size, shuffle=False)

test_input_ids, test_attention_masks, test_target_ids, test_source, test_target = preprocess_csv_data(test_data, max_seq_length = 512)
test_data_tensor = TensorDataset(test_input_ids, test_attention_masks, test_target_ids)
test_loader = DataLoader(test_data_tensor, batch_size=batch_size, shuffle=False)

num_epochs = 100
learning_rate = 1e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

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

#Scores file path and write headers
modality = 'speech'
model_ver = 'bart'
train_scores_file = os.getcwd() + '/results/scores/speechBart/train/train_score_' + modality +'-'+ model_ver + '.txt'
test_scores_file = os.getcwd() + '/results/scores/speechBart/test/test_score_' + modality +'-'+ model_ver + '.txt'
with open(train_scores_file, 'a') as f:
    f.write("train_jaq_similiarity_score, train_cosine_similarity\n")
with open(test_scores_file, 'a') as f:
    f.write("test_jaq_similiarity_score, test_cosine_similarity\n")

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
        train_references.extend([[text] for text in target_texts])  # Adding each target summary as a list for BLEU calculation

    average_loss = total_loss / len(train_loader)
    train_loss_lis.append(average_loss)

    train_jac_scores = [jaccard_similarity(set(" ".join(pred).split()), set(" ".join(ref).split())) for pred, ref in zip(train_predictions, train_references)]

    train_mean_jaq_similiarity_score = np.mean(train_jac_scores)
    train_mean_cosine_similarity = calculate_cosine_similarity(train_predictions, train_references).mean()

    train_score_lis.append((train_mean_jaq_similiarity_score,train_mean_cosine_similarity))

    for i in range(len(train_source)):
        train_data.append((train_source[i], train_target[i], train_predictions[i], train_references[i]))
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
            test_loss += test_outputs.loss.item()

            generated_ids_test = torch.argmax(test_outputs.logits, dim=-1)
            test_batch_predictions = tokenizer.batch_decode(generated_ids_test, skip_special_tokens=True)
            test_predictions.extend(test_batch_predictions)

            target_texts_test = tokenizer.batch_decode(test_target_ids_batch, skip_special_tokens=True)
            test_references.extend([text for text in target_texts_test])

    average_test_loss = test_loss / len(test_loader)
    test_loss_lis.append(average_test_loss)
    test_loss_stdev = np.std(test_loss)


    test_jac_scores = [
        jaccard_similarity(set(" ".join(pred).split()), set(" ".join(ref).split()))
        for pred, ref in zip(test_predictions, test_references)
    ]

    test_mean_jaq_similiarity_score = np.mean(test_jac_scores)
    test_mean_cosine_similarity = calculate_cosine_similarity(test_predictions, test_references).mean()
    test_score_lis.append((test_mean_jaq_similiarity_score,test_mean_cosine_similarity))

    for i in range(len(test_source)):
        test_data.append((test_source[i], test_target[i], test_predictions[i], test_references[i]))

    test_data_by_epoch.append(test_data)


    print(f'Epoch {epoch+1} Train Loss: {average_loss} Test Loss {average_test_loss} Test Jaccard Score {test_mean_jaq_similiarity_score} Test Cosine Similarity: {test_mean_cosine_similarity}')

train_data_directory = os.getcwd() + '/results/predictions/speechBart/train/'
test_data_directory = os.getcwd() + '/results/predictions/speechBart/test/'

train_data_paths = [f'{train_data_directory}train_data_epoch_{epoch + 1}.txt' for epoch in range(num_epochs)]
test_data_paths = [f'{test_data_directory}test_data_epoch_{epoch + 1}.txt' for epoch in range(num_epochs)]

for epoch, epoch_train_data in enumerate(train_data_by_epoch):
    with open(train_data_paths[epoch], 'w') as train_file:
        for data_point in epoch_train_data:
            train_file.write(f'Speech: {data_point[0]}\n')
            train_file.write(f'LTL: {data_point[1]}\n')
            train_file.write(f'Predictions: {data_point[2]}\n')
            train_file.write(f'References: {data_point[3]}\n\n')

for epoch, epoch_test_data in enumerate(test_data_by_epoch):
    with open(test_data_paths[epoch], 'w') as test_file:
        for data_point in epoch_test_data:
            test_file.write(f'Speech: {data_point[0]}\n')
            test_file.write(f'LTL: {data_point[1]}\n')
            test_file.write(f'Predictions: {data_point[2]}\n')
            test_file.write(f'References: {data_point[3]}\n\n')

with open(train_scores_file, 'w') as file:
    # Write each element of the list to the file
    for item in train_score_lis:
        file.write(str(item) + '\n')

with open(test_scores_file, 'w') as file:
    # Write each element of the list to the file
    for item in test_score_lis:
        file.write(str(item) + '\n')

model_file = os.getcwd() + '/results/speech_bart.pth'
torch.save(model, model_file)





