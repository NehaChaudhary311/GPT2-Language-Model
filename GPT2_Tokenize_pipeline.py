import torch
import torch.cuda
from transformers import GPT2Tokenizer, GPT2Model


model = GPT2Model.from_pretrained('gpt2')
model.eval()
def gpt2Tokenizer(text) :
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize our sentence with the GPT2 tokenizer.
    tokenized_text = tokenizer.tokenize(text)

    tokenized_text_rem = tokenized_text.copy()

    j = 0
    # Removal of splitted tokens
    for i in range(1, len(tokenized_text)):
        if tokenized_text[i] in ",:;(){}" :
            i += 1
        elif tokenized_text[i].startswith('Ġ') :
            i += 1
        else:
            break
    if i != len(tokenized_text):
        for j in range(i, len(tokenized_text)):
            if tokenized_text[j][0] in "Ġ.!?, ":
                break
            else:
                j += 1

    else:
        pass

    tokenized_text_rem[i - 1:j] = [''.join(tokenized_text_rem[i - 1:j])]


    # Encoding : maps words to key IDs in vocabulary ditionary
    # Encoding our already tokenized text with the GPT2 encode()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text_index = tokenizer.encode(tokenized_text, add_prefix_space=True)


    # Decoding our token IDs using decode()
    # decoded_text = tokenizer.decode(text_index, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    # print(decoded_text)

    # Taking average of to-be-removed-tokens only when 'i' is not equal to len(tokenized_text)
    # That means 'i' has gone over the list and didn't find any splitted word, which further implies
    # that we do no have to take average of the text_index
    if i != len(tokenized_text):
        sum = 0
        count = 0
        # Calculating sum
        for x in range(i - 1, j):
            sum += text_index[x]
            count += 1

        avg = int(sum / count)

        # Inserting the average in place of the other tokens
        text_index = text_index[:i - 1] + [avg] + text_index[j:]
    else:
        pass

    #Converting text_index to tensor
    text_index_tensor = torch.tensor(text_index)
    return text_index_tensor


