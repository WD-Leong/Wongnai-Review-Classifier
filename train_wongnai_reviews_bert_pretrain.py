import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_keras as bert
from sklearn.metrics import classification_report

# Define the weight update step for multiple sub-batches. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    x_mask, x_input, x_sequence, x_output, 
    optimizer, reg_cls=0.0, reg_emb=0.01, 
    learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_input.shape[0]
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
        
        tmp_mask = x_mask[id_st:id_en, :]
        tmp_input  = x_input[id_st:id_en, :]
        tmp_label  = x_output[id_st:id_en]
        tmp_output = x_sequence[id_st:id_en, :]
        
        with tf.GradientTape() as grad_tape:
            model_outputs = model(
                tmp_input, training=True)
            
            class_logits = model_outputs[0]
            vocab_logits = model_outputs[1]
            bert_outputs = model_outputs[2]
            
            # Masked Language Model Loss. #
            msk_xent = tf.multiply(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=vocab_logits), tmp_mask)
            num_mask = tf.cast(
                tf.reduce_sum(tmp_mask, axis=1), tf.float32)
            msk_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_xent, axis=1), num_mask))
            
            # CLS token is the average embeddings since there is #
            # no Next Sentence Prediction in this pre-training.  #
            cls_embed  = bert_outputs[:, 0, :]
            avg_embed  = tf.reduce_mean(
                bert_outputs[:, 1:, :], axis=1)
            emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(cls_embed - avg_embed), axis=1))
            
            # Supervised Loss. #
            cls_losses = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tmp_label, logits=class_logits))
            
            # Full Loss Function. #
            pre_losses = msk_losses + reg_emb*emb_losses
            tmp_losses = tf.add(
                reg_cls * cls_losses, (1.0-reg_cls) * pre_losses)
        
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
    
    clipped_gradients, _ = tf.clip_by_global_norm(
        acc_gradients, grad_clip)
    optimizer.apply_gradients(
        zip(clipped_gradients, model_params))
    return avg_losses

# Model Parameters. #
batch_size = 256
batch_test = 64
sub_batch  = 64
seq_length = 128
num_heads  = 4
num_layers = 3

gradient_clip = 1.00
maximum_iter  = 1250
restore_flag  = True
save_step     = 50
warmup_steps  = 500
finetune_step = 500
final_tunings = 1000
display_step  = 25
anneal_step   = 2500
anneal_rate   = 0.75

p_keep = 0.90
p_mask = 0.15

grad_clip  = 1.0
prob_noise = 0.0
prob_keep  = 0.9
hidden_size = 256
ffwd_size   = 4*hidden_size
warmup_flag = True
cooling_step = 50

model_ckpt_dir  = "TF_Models/thai_reviews_bert_v1"
train_loss_file = "train_loss_thai_reviews_bert_v1.csv"

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
MSK_token = vocab_size + 4
TRU_token = vocab_size + 5
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

# Build the BERT Model. #
print("Building the BERT Model.")
start_time = time.time()

bert_model = bert.BERTClassifier(
    num_class, num_layers, 
    num_heads, hidden_size, ffwd_size, 
    vocab_size+6, seq_length+2, rate=1.0-p_keep)
bert_optimizer = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

elapsed_time = (time.time() - start_time) / 60
print("BERT Model Built", 
      "(" + str(elapsed_time) + " mins).")

# Print the model summary. #
tmp_zero = np.zeros(
    [sub_batch, seq_length+2], dtype=np.int32)
tmp_pred = bert_model(tmp_zero, training=True)[0]

print(bert_model.summary())
del tmp_zero, tmp_pred

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optimizer=bert_optimizer)

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

# Train the BERT model. #
tmp_in_seq  = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_in_mask = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_out_seq = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
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
print("Training the BERT Network", 
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

    if n_iter >= finetune_step:
        reg_cls = 1.0
        #batch_size = 32
        #sub_batch  = 32
        
        if n_iter < final_tunings:
            learning_rate = 1.0e-4
        else:
            learning_rate = 1.0e-5
    else:
        reg_cls = 0.0
        batch_size = 256
        sub_batch  = 64
    
    # Select a sample from the data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    # Set the mask to be ones to let it learn EOS #
    # and PAD token embeddings.                   # 
    tmp_out_lab[:] = 0
    tmp_in_mask[:, :] = 1.0
    tmp_in_seq[:, :]  = PAD_token
    tmp_out_seq[:, :] = PAD_token
    for n_index in range(batch_size):
        tmp_index = batch_sample[n_index]
        tmp_label = train_data[tmp_index][0]
        tmp_i_tok = [
            word_2_idx.get(x, UNK_token) \
                for x in train_data[tmp_index][1]]
        num_token = len(tmp_i_tok)

        # Truncate the sequence if it exceeds the maximum #
        # sequence length. Randomly select the review's   #
        # start and end index to be the positive example. #
        if num_token > seq_length:
            # For the anchor. #
            id_st = np.random.randint(
                0, num_token-seq_length)
            id_en = id_st + seq_length
            
            tmp_i_idx = [CLS_token]
            tmp_i_idx += tmp_i_tok[id_st:id_en]
            
            if id_en < num_token:
                # Add TRUNCATE token. #
                tmp_i_idx += [TRU_token]
            else:
                tmp_i_idx += [EOS_token]
            del id_st, id_en
        else:
            tmp_i_idx = tmp_i_tok
            tmp_i_idx += [EOS_token]
        n_input = len(tmp_i_idx)

        # Generate the masked sequence. #
        mask_seq  = [MSK_token] * n_input
        tmp_mask  = np.random.binomial(
            1, p_mask, size=n_input)
        
        tmp_noise = [CLS_token]
        tmp_noise += list(np.random.choice(
            vocab_size, size=n_input-2))
        tmp_noise += [tmp_i_idx[-1]]
        
        tmp_unif = np.random.uniform()
        if tmp_unif <= 0.8:
            # Replace with MASK token. #
            tmp_i_msk = [
                tmp_i_idx[x] if tmp_mask[x] == 0 else \
                    mask_seq[x] for x in range(n_input)]
        elif tmp_unif <= 0.9:
            # Replace with random word. #
            tmp_i_msk = [
                tmp_i_idx[x] if tmp_mask[x] == 0 else \
                    tmp_noise[x] for x in range(n_input)]
        else:
            # No replacement. #
            tmp_i_msk = tmp_i_idx
        
        tmp_out_lab[n_index] = tmp_label
        tmp_in_mask[n_index, :n_input] = tmp_mask
        tmp_in_seq[n_index, :n_input]  = tmp_i_msk
        tmp_out_seq[n_index, :n_input] = tmp_i_idx
    
    # Set the training data. #
    tmp_in  = tmp_in_seq
    tmp_out = tmp_out_lab
    tmp_seq = tmp_out_seq
    tmp_msk = tmp_in_mask
    
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, 
        tmp_msk, tmp_in, tmp_seq, tmp_out, 
        bert_optimizer, reg_cls=reg_cls, 
        learning_rate=learning_rate, grad_clip=grad_clip)
    
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        # For simplicity, get the test accuracy #
        # instead of validation accuracy.       #
        pred_labels = []
        for n_val_batch in range(n_val_batches):
            id_st = n_val_batch * batch_test
            id_en = (n_val_batch+1) * batch_test
            
            if n_val_batch == (n_val_batches-1):
                curr_batch = num_test - id_st
            else:
                curr_batch = batch_test
            
            tmp_test_tokens = np.zeros(
                [curr_batch, seq_length+2], dtype=np.int32)
            
            tmp_test_tokens[:, :] = PAD_token
            for tmp_n in range(curr_batch):
                tmp_input = [
                    word_2_idx.get(x, UNK_token) \
                        for x in valid_data[id_st+tmp_n][1]]
                
                # Truncate if the length is longer. #
                n_input  = len(tmp_input)
                tmp_toks = [CLS_token]
                if n_input > seq_length:
                    tmp_toks += tmp_input[:seq_length]
                    tmp_toks += [TRU_token]
                else:
                    tmp_toks += tmp_input + [EOS_token]
                n_decode = len(tmp_toks)
                
                tmp_test_tokens[tmp_n, :n_decode] = tmp_toks
                del tmp_toks, tmp_input, n_input, n_decode
            
            # Perform inference. #
            tmp_pred_labels = bert_model.infer(
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
        del pred_labels
        
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

pred_labels = []
for n_val_batch in range(n_val_batches):
    id_st = n_val_batch * batch_test
    id_en = (n_val_batch+1) * batch_test
    
    if n_val_batch == (n_val_batches-1):
        curr_batch = num_test - id_st
    else:
        curr_batch = batch_test
    
    tmp_test_tokens = np.zeros(
        [curr_batch, seq_length+2], dtype=np.int32)
    
    tmp_test_tokens[:, :] = PAD_token
    for tmp_n in range(curr_batch):
        tmp_input = [
            word_2_idx.get(x, UNK_token) \
                for x in valid_data[id_st+tmp_n][1]]
        
        # Truncate if the length is longer. #
        n_input  = len(tmp_input)
        tmp_toks = [CLS_token]
        if n_input > seq_length:
            tmp_toks += tmp_input[:seq_length]
            tmp_toks += [TRU_token]
        else:
            tmp_toks += tmp_input + [EOS_token]
        n_decode = len(tmp_toks)
        
        tmp_test_tokens[tmp_n, :n_decode] = tmp_toks
        del tmp_toks, tmp_input, n_input, n_decode
    
    # Perform inference. #
    tmp_pred_labels = bert_model.infer(
        tmp_test_tokens).numpy()
    pred_labels.append(tmp_pred_labels)
    del tmp_test_tokens

# Concatenate the predicted labels. #
pred_labels = np.concatenate(
    tuple(pred_labels), axis=0)
del tmp_pred_labels

target_name = [
    "rating_" + str(x+1) for x in range(num_class)]
pred_report = classification_report(
    test_labels, pred_labels, target_names=target_name)
del pred_labels

tmp_result_file = "thai_review_bert_"
tmp_result_file += "validation_report.txt"
with open(tmp_result_file, "w") as tmp_write:
    tmp_write.write(pred_report)
print(pred_report)
