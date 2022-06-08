import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_gpt_cls_keras as gpt_cls
from sklearn.metrics import classification_report

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_encode, x_output, 
    optimizer, reg_cls=0.0, reg_lm=0.0, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_encode.shape[0]
    if batch_size <= sub_batch_sz:
        sub_batch = 1
    elif batch_size % sub_batch_sz == 0:
        sub_batch = int(batch_size / sub_batch_sz)
    else:
        sub_batch = int(batch_size / sub_batch_sz) + 1
    
    model_params  = model.trainable_variables
    acc_gradients = [
        tf.zeros_like(var) for var in model_params]
    
    tot_losses = 0.0
    for n_sub in range(sub_batch):
        id_st = n_sub*sub_batch_sz
        if n_sub != (sub_batch-1):
            id_en = (n_sub+1)*sub_batch_sz
        else:
            id_en = batch_size
        
        tmp_output = x_output[id_st:id_en]
        tmp_decode = x_encode[id_st:id_en, 1:]
        tmp_encode = x_encode[id_st:id_en, :-1]
        
        with tf.GradientTape() as grad_tape:
            model_outputs = model(
                tmp_encode, training=True)
            cls_logits = model_outputs[0]
            gpt_logits = model_outputs[1]
            
            cls_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=cls_logits))
            
            gpt_losses = tf.reduce_sum(tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_decode, logits=gpt_logits), axis=1))
            tmp_losses = tf.add(
                reg_cls*cls_losses, reg_lm*gpt_losses)
        
        # Accumulate the gradients. #
        tot_losses += tmp_losses
        tmp_gradients = grad_tape.gradient(
            tmp_losses, model_params)
        acc_gradients = [tf.add(
            acc_grad, grad) for acc_grad, grad \
                in zip(acc_gradients, tmp_gradients)]
    
    # Update using the optimizer. #
    avg_losses = tot_losses / batch_size
    acc_gradients = [tf.math.divide_no_nan(
        acc_grad, batch_size) for acc_grad in acc_gradients]
    
    clip_tuple = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clip_tuple[0], model_params))
    return avg_losses

# Model Parameters. #
batch_size = 256
batch_test = 128
sub_batch  = 64
seq_length = 256
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 1250
restore_flag  = True
save_step     = 50
warmup_steps  = 500
finetune_step = 750
final_tuning  = 1000
display_step  = 25
anneal_step   = 2500
anneal_rate   = 0.75

grad_clip  = 1.0
prob_noise = 0.0
prob_keep  = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 50

model_ckpt_dir  = "TF_Models/thai_reviews_gpt_cls_v3"
train_loss_file = "train_loss_thai_reviews_gpt_cls_v3.csv"

# Load the data. #
tmp_pkl_file = "../../Data/wongnai_reviews/"
tmp_pkl_file += "thai_reviews_word.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    train_data = pkl.load(tmp_load_file)
    valid_data = pkl.load(tmp_load_file)
    test_sw_data = pkl.load(tmp_load_file)

    word_vocab = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)
    word_2_idx = pkl.load(tmp_load_file)

# Filter out the data. #
train_data = [
    (x, y) for x, y in train_data if len(y) != 0]

# Convert the rating to zero-based index. #
train_data = [(x-1, y) for x, y in train_data]
valid_data = [(x-1, y) for x, y in valid_data]

# Define the mapping for the labels. #
vocab_size = len(word_vocab)
print("Vocabulary Size:", str(vocab_size) + ".")

# Number of samples to train and validate the model. #
num_test  = len(valid_data)
num_data  = len(train_data)
num_class = 5

# Define the special tokens. #
CLS_token = vocab_size
EOS_token = vocab_size + 1
PAD_token = vocab_size + 2
UNK_token = vocab_size + 3
print("Total of", str(len(train_data)), "rows loaded.")

if num_test <= batch_test:
    n_val_batches = 1
elif num_test % batch_test == 0:
    n_val_batches = int(num_test / batch_test)
else:
    n_val_batches = int(num_test / batch_test) + 1

# Extract the validation labels. #
test_labels = np.array([x for x, y in valid_data])

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(6)

# Build the GPT Model. #
print("Building the GPT Model.")
start_time = time.time()

cls_model = gpt_cls.GPTClassifierDecoder(
    num_class, num_layers, num_heads, 
    hidden_size, ffwd_size, vocab_size+4, 
    seq_length, rate1=0.0, rate2=1.0-prob_keep)
cls_optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9, beta_2=0.98, epsilon=1.0e-9)

elapsed_time = (time.time() - start_time) / 60
print("GPT Classifier Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Print the model summary. #
tmp_zero = np.zeros(
    [sub_batch, seq_length], dtype=np.int32)
tmp_pred = cls_model(tmp_zero, training=True)[0]

print(cls_model.summary())
del tmp_zero, tmp_pred

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    cls_model=cls_model, 
    cls_optimizer=cls_optimizer)

manager = tf.train.CheckpointManager(
    ckpt, model_ckpt_dir, max_to_keep=1)

if restore_flag:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Model restored from {}".format(
            manager.latest_checkpoint))
    else:
        print("Error: No latest checkpoint found.")
    
    train_loss_df = pd.read_csv(train_loss_file)
    train_loss_list = [tuple(
        train_loss_df.iloc[x].values) \
            for x in range(len(train_loss_df))]
else:
    print("Training a new model.")
    train_loss_list = []

# Train the GPT Classifier model. #
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+1], dtype=np.int32)
tmp_out_lab = np.zeros([batch_size], dtype=np.int32)

# Warmup learning schedule. #
n_iter = ckpt.step.numpy().astype(np.int32)
if warmup_flag:
    step_min = float(max(n_iter, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_min
else:
    initial_lr = 0.001
    anneal_pow = int(n_iter / anneal_step)
    learning_rate = max(np.power(
        anneal_rate, anneal_pow)*initial_lr, 1.0e-5)

print("-" * 50)
print("Training the GPT Classifier Network", 
      "(" + str(n_iter) + " iterations).")
print(str(num_data), "training samples.")
print(str(num_test), "test data samples.")
print("-" * 50)

# Update the neural network's weights. #
tot_loss = 0.0
start_tm = time.time()
while n_iter < maximum_iter:
    if warmup_flag:
        step_min = float(max(n_iter, warmup_steps))**(-0.5)
        learning_rate = float(hidden_size)**(-0.5) * step_min
    else:
        if n_iter % anneal_step == 0:
            anneal_pow = int(n_iter / anneal_step)
            learning_rate = max(np.power(
                anneal_rate, anneal_pow)*initial_lr, 1.0e-4)
    
    if n_iter >= warmup_steps:
        # Finetuning stage. #
        reg_lm  = 0.001
        reg_cls = 1.0
        if n_iter <= final_tuning:
            learning_rate = 1.0e-4
        else:
            learning_rate = 1.0e-5
    else:
        # Language Model. #
        reg_cls = 0.0
        reg_lm  = 1.0
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    tmp_out_lab[:] = 0
    tmp_in_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = train_data[tmp_index][0]
        tmp_i_tok = train_data[tmp_index][1]
        n_tokens  = len(tmp_i_tok)
        
        # Convert into integer indices. #
        tmp_i_idx = [word_2_idx.get(
            x, UNK_token) for x in tmp_i_tok]
        
        n_input = len(tmp_i_idx)
        if n_input > seq_length:
            if n_iter >= finetune_step:
                # Truncate the review. #
                tmp_p_idx = tmp_i_idx[:(seq_length+1)]
            else:
                # Randomly sample a segment. #
                if n_input == (seq_length+1):
                    id_st = 0
                else:
                    id_st = np.random.randint(
                        0, n_input-seq_length-1)
                id_en = id_st + seq_length
                tmp_p_idx = tmp_i_idx[id_st:id_en]
        else:
            # Take the input sequence as it is. #
            tmp_p_idx = tmp_i_idx + [EOS_token]
        n_sw_toks = len(tmp_p_idx)

        tmp_out_lab[n_index] = tmp_label
        tmp_in_seq[n_index, :n_sw_toks] = tmp_p_idx

    # Set the training data. #
    tmp_input  = tmp_in_seq
    tmp_output = tmp_out_lab
    
    tmp_loss = sub_batch_train_step(
        cls_model, sub_batch, 
        tmp_input, tmp_output, 
        cls_optimizer, reg_cls=reg_cls, reg_lm=reg_lm, 
        learning_rate=learning_rate, grad_clip=grad_clip)
    
    n_iter += 1
    ckpt.step.assign_add(1)

    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        if n_iter >= warmup_steps:
            # Get the validation accuracy. #
            pred_labels = []
            for n_val_batch in range(n_val_batches):
                id_st = n_val_batch * batch_test
                id_en = (n_val_batch+1) * batch_test
                
                if n_val_batch == (n_val_batches-1):
                    curr_batch = num_test - id_st
                else:
                    curr_batch = batch_test
                
                tmp_test_tokens = np.zeros(
                    [curr_batch, seq_length], dtype=np.int32)
                
                tmp_test_tokens[:, :] = PAD_token
                for tmp_n in range(curr_batch):
                    tmp_input = [
                        word_2_idx.get(x, UNK_token) for \
                            x in valid_data[id_st+tmp_n][1]]
                    
                    # Truncate if the length is longer. #
                    n_input  = len(tmp_input)
                    if n_input >= seq_length:
                        # Truncate the review. #
                        tmp_toks  = tmp_input[:seq_length]
                        n_sw_toks = seq_length
                    else:
                        tmp_toks  = tmp_input + [EOS_token]
                        n_sw_toks = len(tmp_toks)
                    
                    tmp_test_tokens[tmp_n, :n_sw_toks] = tmp_toks
                    del tmp_toks, tmp_input, n_input, n_sw_toks

                # Perform inference. #
                tmp_pred_labels = cls_model.infer(
                    tmp_test_tokens).numpy()
                pred_labels.append(tmp_pred_labels)
                del tmp_test_tokens
            
            # Concatenate the predicted labels. #
            pred_labels = np.concatenate(
                tuple(pred_labels), axis=0)
            del tmp_pred_labels
            
            # Compute the accuracy. #
            accuracy = np.sum(np.where(
                pred_labels == test_labels, 1, 0)) / num_test
            eval_labels = pred_labels
            del pred_labels
        else:
            accuracy = 0.0
        
        end_tm = time.time()
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (end_tm - start_tm) / 60
        
        print("Iteration", str(n_iter) + ".")
        print("Elapsed Time:", str(elapsed_tm), "mins.")
        print("Gradient Clip:", str(gradient_clip) + ".")
        print("Learning Rate:", str(learning_rate) + ".")
        print("Average Loss:", str(avg_loss) + ".")
        print("Test Accuracy:", str(round(accuracy*100, 2)) + "%.")
        
        train_loss_list.append((n_iter, avg_loss, accuracy))
        start_tm = time.time()
        print("-" * 50)
    
    # Model checkpoint. #
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        tmp_df_column = ["n_iter", "xent_loss", "test_acc"]
        tmp_df_losses = pd.DataFrame(
            train_loss_list, columns=tmp_df_column)
        tmp_df_losses.to_csv(train_loss_file, index=False)
        del tmp_df_losses
    
    # Cool the GPU. #
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("Resume Training.")
        print("-" * 50)

# Print the evaluation report. #
print("Generating evaluation report.")

# Get the validation accuracy. #
pred_labels = []
for n_val_batch in range(n_val_batches):
    id_st = n_val_batch * batch_test
    id_en = (n_val_batch+1) * batch_test
    
    if n_val_batch == (n_val_batches-1):
        curr_batch = num_test - id_st
    else:
        curr_batch = batch_test
    
    tmp_test_tokens = np.zeros(
        [curr_batch, seq_length], dtype=np.int32)
    
    tmp_test_tokens[:, :] = PAD_token
    for tmp_n in range(curr_batch):
        tmp_input = [
            word_2_idx.get(x, UNK_token) for \
                x in valid_data[id_st+tmp_n][1]]
        
        # Truncate if the length is longer. #
        n_input  = len(tmp_input)
        if n_input >= seq_length:
            # Truncate the review. #
            tmp_toks  = tmp_input[:seq_length]
            n_sw_toks = seq_length
        else:
            tmp_toks  = tmp_input + [EOS_token]
            n_sw_toks = len(tmp_toks)
        
        tmp_test_tokens[tmp_n, :n_sw_toks] = tmp_toks
        del tmp_toks, tmp_input, n_input, n_sw_toks

    # Perform inference. #
    tmp_pred_labels = cls_model.infer(
        tmp_test_tokens).numpy()
    pred_labels.append(tmp_pred_labels)
    del tmp_test_tokens

# Concatenate the predicted labels. #
pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)
del tmp_pred_labels

# Compute the accuracy. #
accuracy = np.sum(np.where(
    pred_labels == test_labels, 1, 0)) / num_test
eval_labels = pred_labels
del pred_labels

target_name = [
    "rating_" + str(x+1) for x in range(num_class)]
pred_report = classification_report(
    test_labels, eval_labels, target_names=target_name)

tmp_result_file = "thai_review_gpt_"
tmp_result_file += "validation_report_v1.txt"
with open(tmp_result_file, "w") as tmp_write:
    tmp_write.write(pred_report)
print(pred_report)
