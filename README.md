**Question 3.** **- Transliteration**
   
`Training Details`
The model was trained on a representative subset of the complete dataset to ensure computational stability and avoid runtime interruptions during training. This setup was designed to demonstrate the model’s architecture and performance within the available compute limits.
 - Number of Epochs: Training was intentionally limited to a minimal number of epochs (≤30) due to GPU and time constraints. While longer training would likely improve accuracy, the focus here was to present a stable and efficient implementation for evaluation purposes.
 - Early Stopping: An early stopping mechanism was implemented with `restore_best_weights=True` to automatically retain the most optimal model weights based on validation performance.
 - Model Saving Strategy: After each epoch, only the best-performing model (lowest validation loss) is saved as `conditioned_seq2seq_models/best_conditioned_seq2seq.h5` Older checkpoints are automatically removed to maintain a clean directory and conserve storage space.

**Note:** Although I initially intended to train the model for more epochs to achieve higher accuracy, GPU runtime limitations restricted extended training.
