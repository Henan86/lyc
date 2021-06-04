from sklearn.metrics import accuracy_score
from transformers.trainer_callback import TrainerCallback
from lyc.utils import vector_l2_normlize
import numpy as np
from sklearn.metrics import accuracy_score

metrics_computing={
    'accuracy': accuracy_score,
}

def pred_forward(model, eval_dl):
    all_preds = []
    all_true = []
    for batch in eval_dl:
        label = batch.pop('labels')
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

def SimCSEEvalAccComputing(preds, threshold=0.4):
    prediction=preds.prediction
    labels=pred.label_ids

    prediction = vector_l2_normlize(prediction)
    embs_a, embs_b = np.split(prediction, 2)
    sims = np.dot(embs_a, embs_b.T)
    sims = np.diag(sims)
    acc=accuracy_score(labels, sims>threshold)
    print('ACC: ', acc)
    return {'ACC' : acc}
