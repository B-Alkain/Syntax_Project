import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix as sk_confusion_matrix
import numpy as np
import pandas as pd

def tag_whole_test(model, sentences, tags):
    gold_all = []
    pred_all = []

    for sent, gold in zip(sentences, tags):
        pred = model.viterbi(sent)
        gold_all.extend(gold)
        pred_all.extend(pred)
    return gold_all, pred_all

def confusion_matrix(gold_all, pred_all, tags, title='Confusion matrix'):

    cm = sk_confusion_matrix(gold_all, pred_all, labels=list(sorted(tags)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12))
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt=".2f", # Format to 2 decimal places for percentages
        cmap="Blues",
        cbar=True,
        xticklabels=tags,
        yticklabels=tags
    )

    plt.title(title)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()
    return cm_norm

def report_to_dataframe(report):
    tag_data = {}
    for tag, metrics in report.items():
        if tag not in ['accuracy', 'macro avg', 'weighted avg']:
            tag_data[tag] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            }
    df = pd.DataFrame(tag_data).T[['precision', 'recall', 'f1-score']]
    df = df.sort_values(by='f1-score', ascending=False)

    return df

def per_tag_metrics(report, title = 'Per-Tag Performance: Precision, Recall, and F1-Score'):
    df = report_to_dataframe(report)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('POS Tag', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Metric')
    plt.tight_layout()
    print(f"Grouped bar chart")
    plt.show()


def compare_reports(report1, report2, metric='f1-score', 
                           model_name1='Our HMM', model_name2='Sklearn HMM',
                           language=None):
    
    df1 = report_to_dataframe(report1)
    df2 = report_to_dataframe(report2)
    
    tags = df1.index.tolist()

    bar_width = 0.35
    x1 = np.arange(len(tags))
    x2 = x1 + bar_width
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax1 = ax.bar(x1, df1['f1-score'], width=bar_width, label=model_name1)
    ax2 = ax.bar(x2, df2['f1-score'], width=bar_width, label=model_name2)
    ax.set_xticks(x1 + bar_width / 2)
    ax.set_xticklabels(tags, rotation=45, ha='right')
    ax.set_xlabel('Part-of-Speech Tag', fontweight='bold')
    ax.set_ylabel(f'{metric.upper()}', fontweight='bold')
    ax.set_title(f'Comparison of {metric.upper()} by Tag: {model_name1} vs {model_name2} [{language}]')
    ax.legend()
    plt.tight_layout()
    plt.show()