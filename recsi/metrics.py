"""Count-based metrics in percent; labels occupy rows of the confusion matrix."""
import numpy as np


def classification_metrics(labels, predictions, classes):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("Expected matching nonempty label/prediction vectors")
    if min(y.min(), p.min()) < 0 or max(y.max(), p.max()) >= classes:
        raise ValueError("Class index outside the fixed label vocabulary")
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    support, guessed, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    precision = np.divide(tp, guessed, out=np.zeros(classes), where=guessed > 0)
    f1 = np.divide(2*tp, support+guessed, out=np.zeros(classes), where=support+guessed > 0)
    return {"W-F1": 100*float((f1*support).sum()/len(y)),
            "WA": 100*float(tp.sum()/len(y)), "UA": 100*float(recall.mean()),
            "Macro-F1": 100*float(f1.mean()), "confusion": cm.tolist(),
            "per_class_precision": (100*precision).tolist(),
            "per_class_recall": (100*recall).tolist(), "per_class_f1": (100*f1).tolist(),
            "n": len(y), "correct": int(tp.sum())}
