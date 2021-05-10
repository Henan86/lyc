from sklearn.metrics import accuracy_score

metrics_computing={
    'accuracy': accuracy_score,
}

def pred_forward(model, eval_dl):
    all_preds = []
    all_true = []
    for batch in eval_dl:
        label = batch.pop('label')
        outputs=model(**batch)
        all_preds.extend(outputs.logits)
        all_true.extend(label)
    
    return all_true, all_preds

def GeneralEval(model, eval_dl, writer, metrics, global_step):

    all_true, all_preds = pred_forward(model, eval_dl)
    
    results = {}
    for metric in metrics:
        results[metric] = metrics_computing[metric](all_true, all_preds)
    
    for k,v in results.items():
        writer.add_scalar(k, v, global_step)
    
    return results
