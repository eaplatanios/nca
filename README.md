# Neural Cognitive Architectures (NCA)

Grammar:

```
<concept>:
  - grammar      {CoLA}
  - implication  {MNLI, QNLI, RTE, SNLI, WNLI}
  - equivalence  {MRPC, QQP}
  - sentiment    {SST}
  - apply[<modifier>, <concept>]

<modifier>:
  - positive
  - negative
  - neutral

<task>:
  - classify[<modifier>[<concept>]...]
  - label[<modifier>[<concept>]...]
```

Examples:

  - *CoLA:* `classify[positive[grammar], negative[grammar]]`
  - *MNLI:* `classify[positive[implication], negative[implication], neutral[implication]]`

### Hugging Face BERT Base Results

  - CoLA: Matthew's Correlation Coefficient = 48.87
  - SST-2: Accuracy = 91.74
  - MRPC: F1/Accuracy = 90.70/86.27
  - STS-B: Person/Spearman Correlation Coefficients = 91.39/91.04
  - QQP: Accuracy/F1 = 90.79/87.66
  - MNLI: Matched/Mismatched Accuracy = 83.70/84.83
  - QNLI: Accuracy = 89.31
  - RTE: Accuracy = 71.43
  - WNLI: Accuracy = 43.66

### RoBERTa Results

      MNLI QNLI  QQP RTE   SST MRPC CoLA  STS
Base  87.6 92.8 91.9 78.7 94.8 90.2 63.6 91.2
Large 90.2 94.7 92.2 86.6 96.4 90.9 68.0 92.4
