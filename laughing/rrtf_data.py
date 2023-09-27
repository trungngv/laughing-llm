# Create the data in RRTF format for training (should we do this on training data only or both training + testing?)
# what is test data used for? the purpose of this exercise is to do well on human-eval ...
# 1. Generate code program with phi_1.5 base model
# 2. Generate unit test cases fined-tuned model (using output from step 1)
# 3. Score test cases by different models

from laughing import phi15

model, _, tokenizer = phi15.load_model_and_tokenizer()
