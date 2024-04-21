from evaluation import *

class_names = ['Mainland China', 'Hong Kong/Macau', 'Taiwan', 'Military', 'Society']
y_news_texts, y_pred, y_pred_probs, y_test = getClassification(model, test_dataloader)
idx = 2
news_text = y_news_texts[idx]
true_category = y_test[idx]
prediction = y_pred[idx]

pred_df = pd.DataFrame (
    {
        'class_names': class_names,
        'values': y_pred_probs[idx]
    }
)
print('\n'.join(wrap(news_text)))
print(f'Predicted Category: {class_names[prediction]}' )
print(f'True Category: {class_names[true_category]}')

def confidence_level(pred_df):
    sns.barplot(x = 'values', y = 'class_names', data = pred_df, orient = 'h')
    plt.xlabel('probability')
    plt.ylabel('category')
    plt.xlim([0,1])
    plt.show()
# confidence_level(pred_df)

def raw_text_pred():
    test_text = '他媽的'
    encoded_review = tokenizer.encode_plus(
        test_text,
        max_length = MAX_LEN,
        add_special_tokens = True,
        truncation = True,
        return_token_type_ids = False,
        padding = 'max_length',
        return_attention_mask = True,
        return_tensors = 'pt')
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = model(input_ids, attention_mask) #creates predictions from the sample text
    _, prediction = torch.max(output, dim = 1) #outputs the prediction with the highest probability
    print(f'sample text: {test_text}')
    print(f'classification: {class_names[prediction]}')

raw_text_pred()

