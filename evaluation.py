from main import *

model = NewsClassifier(5) #5 classes to predict
model.load_state_dict(torch.load('best_model_state.bin'))
model = model.to(device)

def getClassification(model, data_loader):
    model = model.eval()
    
    news_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d['content']
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

            _, preds = torch.max(outputs, dim = 1)
            
        
            news_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
    
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    real_values = torch.stack(real_values)
            

    return news_texts, predictions, prediction_probs, real_values

#test_acc, test_loss = eval_model(model, test_dataloader, loss_fn, device, len(test_dataset))
#print(test_acc)
#Accuracy: 0.8090
def main():
    class_names = ['Mainland China', 'Hong Kong/Macau', 'Taiwan', 'Military', 'Society']
    y_news_texts, y_pred, y_pred_probs, y_test = getClassification(model, test_dataloader)
    print(classification_report(y_test, y_pred, target_names = class_names))

    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot = True, fmt = "d", cmap = "Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right')
        hmap.yaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation = 30, ha = 'right')
        plt.ylabel('True Category')
        plt.xlabel('Predicted Category')
        plt.show()

    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = class_names, columns = class_names)
    show_confusion_matrix(df_cm)

# Comment out when you just want to test stuff (in testing.py)
if __name__ == "__main__":
    main()