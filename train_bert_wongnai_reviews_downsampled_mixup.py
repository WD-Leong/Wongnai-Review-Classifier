import time
import numpy as np
import pandas as pd
import pickle as pkl

import tensorflow as tf
import tensorflow_addons as tfa
import tf_ver2_bert_downsampled_mixup as bert
from sklearn.metrics import classification_report

# Function to generate inputs. #
def generate_input(
    token_input, word_2_idx, vocab_size, 
    ker_sz, seq_length, out_length, 
    CLS_token, EOS_token, MSK_token, TRU_token):
    tmp_i_tok = [word_2_idx.get(
        x, UNK_token) for x in token_input]
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
    mask_seq = [MSK_token] * n_input
    tmp_mask = [1 for x in range(n_input)]
    #tmp_mask = np.random.binomial(
    #    1, p_mask, size=n_input)
    
    tmp_replace = [
        x for x in np.random.choice(
            vocab_size, size=n_input-2)]

    tmp_noise = [CLS_token]
    tmp_noise += tmp_replace
    tmp_noise += [tmp_i_idx[-1]]
    del tmp_replace
    
    # Sample within the average pooling kernel. #
    o_sample = np.random.randint(
        0, ker_sz, size=out_length+1)
    o_index  = [min(
        n_input-1, x+o_sample[int(x/ker_sz)]) \
            for x in range(0, n_input, ker_sz)]
    tmp_o_idx = [
        tmp_i_idx[x] for x in o_index[:out_length]]
    n_output  = len(tmp_o_idx)
    
    # Get the downsampling mask. #
    tmp_o_msk = [
        tmp_mask[x:(x*ker_sz)] for x in range(n_output)]
    tmp_o_msk = [
        0 if len(x) == 0 else max(x) for x in tmp_o_msk]
    
    # Set the input mask mechanism. #
    tmp_unif = np.random.uniform()
    if tmp_unif <= 0.8:
        # Replace with MASK token. #
        tmp_i_msk = [
            tmp_i_idx[x] if x not in o_index else \
                mask_seq[x] for x in range(n_input)]
    elif tmp_unif <= 0.9:
        # Replace with random word. #
        tmp_i_msk = [
            tmp_i_idx[x] if x not in o_index else \
                tmp_noise[x] for x in range(n_input)]
    else:
        # No replacement. #
        tmp_i_msk = tmp_i_idx
    return tmp_i_msk, tmp_o_idx, tmp_o_msk

# Sub-batch weight updates. #
def sub_batch_train_step(
    model, sub_batch_sz, 
    vocab_sz, n_class, x_mask1, x_mask2, 
    x_input1, x_input2, x_output1, x_output2, 
    x_label1, x_label2, optimizer, reg_cls=0.0, 
    reg_emb=0.01, learning_rate=1.0e-3, grad_clip=1.0):
    optimizer.lr.assign(learning_rate)
    
    batch_size = x_input1.shape[0]
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
        
        tmp_mask1 = x_mask1[id_st:id_en, :]
        tmp_mask2 = x_mask2[id_st:id_en, :]

        tmp_input1 = x_input1[id_st:id_en, :]
        tmp_label1 = x_label1[id_st:id_en]
        tmp_input2 = x_input2[id_st:id_en, :]
        tmp_label2 = x_label2[id_st:id_en]
        tmp_output1 = x_output1[id_st:id_en, :]
        tmp_output2 = x_output2[id_st:id_en, :]
        
        # Randomly set the mix-up proportion. #
        alpha = np.random.uniform(size=id_en-id_st)
        alpha = np.expand_dims(alpha, axis=1)
        
        tmp_mask = tf.add(
            alpha * tmp_mask1, (1.0-alpha) * tmp_mask2)
        tmp_mask = tf.cast(tmp_mask, tf.float32)
        tmp_label = tf.add(
            alpha * tf.one_hot(tmp_label1, n_class), 
            (1.0-alpha) * tf.one_hot(tmp_label2, n_class))
        
        alpha = np.expand_dims(alpha, axis=2)
        tmp_output = tf.add(
            alpha * tf.one_hot(tmp_output1, vocab_sz), 
            (1.0-alpha) * tf.one_hot(tmp_output2, vocab_sz))
        
        with tf.GradientTape() as grad_tape:
            model_outputs = model.mixup_output(
                tmp_input1, tmp_input2, alpha, training=True)
            
            class_logits = model_outputs[0]
            vocab_logits = model_outputs[1]
            bert_outputs = model_outputs[2]
            
            # Masked Language Model Loss. #
            msk_xent = tf.multiply(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tmp_output, logits=vocab_logits), tmp_mask)
            num_mask = tf.cast(
                tf.reduce_sum(tmp_mask, axis=1), tf.float32)
            msk_losses = tf.reduce_sum(tf.math.divide_no_nan(
                tf.reduce_sum(msk_xent, axis=1), num_mask))

            # CLS token embeddings. #
            cls_embed = bert_outputs[:, 0, :]
            avg_embed = tf.reduce_mean(
                bert_outputs[:, 1:, :], axis=1)
            emb_losses = tf.reduce_sum(tf.reduce_mean(
                tf.square(cls_embed - avg_embed), axis=1))

            # Supervised Loss. #
            cls_losses = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=tmp_label, logits=class_logits))
            
            # Full Loss Function. #
            pre_losses = msk_losses + reg_emb * emb_losses
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

# Load the data. #
print("Loading the data.")

tmp_pkl_file = "../../Data/wongnai_reviews/"
tmp_pkl_file += "thai_reviews_word.pkl"
with open(tmp_pkl_file, "rb") as tmp_load_file:
    train_data = pkl.load(tmp_load_file)
    valid_data = pkl.load(tmp_load_file)
    test_sw_data = pkl.load(tmp_load_file)

    word_vocab = pkl.load(tmp_load_file)
    idx_2_word = pkl.load(tmp_load_file)
    word_2_idx = pkl.load(tmp_load_file)

vocab_size = len(word_vocab)
print("Total of", vocab_size, "tokens.")

# Filter out the data. #
train_data = [
    (x, y) for x, y in train_data if len(y) != 0]

# Convert the rating to zero-based index. #
train_data = [(x-1, y) for x, y in train_data]
valid_data = [(x-1, y) for x, y in valid_data]

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

# Parameters. #
steps_max  = 1250
n_layers   = 3
n_heads    = 4
seq_length = 300

batch_size = 256
sub_batch  = 64
batch_test = 64

ker_sz = 3
p_mask = 0.15
p_keep = 0.90
hidden_size = 256
ffwd_size   = 4 * hidden_size
out_length  = int((seq_length+2) / ker_sz)

gradient_clip = 1.00
warmup_steps  = 500
finetune_step = 500
final_tunings = 1000
cooling_step  = 100
save_step     = 50
display_step  = 25
restore_flag  = False

model_ckpt_dir  = "TF_Models/bert_downsampled_thai_reviews_mixup"
train_loss_file = "bert_downsampled_thai_reviews_losses_mixup.csv"

# Set the number of threads to use. #
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(5)

# Define the classifier model. #
bert_model = bert.BERTClassifier(
    num_class, n_layers, n_heads, 
    hidden_size, ffwd_size, vocab_size+6, 
    seq_length+2, ker_sz, rate=1.0-p_keep)
bert_optim = tfa.optimizers.AdamW(
    weight_decay=1.0e-4)

# Extract the labels. #
target_labels = [
    "rating_" + str(x+1) for x in range(num_class)]

# Create the model checkpoint. #
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0), 
    bert_model=bert_model, 
    bert_optim=bert_optim)

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
    del train_loss_df
else:
    print("Training a new model.")
    train_loss_list = []

# Format the data before training. #
if num_test <= batch_size:
    n_val_batches = 1
elif num_test % batch_size == 0:
    n_val_batches = int(num_test / batch_size)
else:
    n_val_batches = int(num_test / batch_size) + 1

# Train the model. #
print("Training the BERT Down-Sampled Model.")
print(num_data, "training data.")
print(num_test, "validation data.")
print("-" * 50)

n_iter = ckpt.step.numpy().astype(np.int32)
tmp_in_seq1 = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_in_seq2 = np.zeros(
    [batch_size, seq_length+2], dtype=np.int32)
tmp_in_mask1 = np.zeros(
    [batch_size, out_length], dtype=np.float32)
tmp_in_mask2 = np.zeros(
    [batch_size, out_length], dtype=np.float32)

tmp_out_seq1 = np.zeros(
    [batch_size, out_length], dtype=np.int32)
tmp_out_seq2 = np.zeros(
    [batch_size, out_length], dtype=np.int32)
tmp_out_lab1 = np.zeros([batch_size], dtype=np.int32)
tmp_out_lab2 = np.zeros([batch_size], dtype=np.int32)

tot_loss = 0.0
start_tm = time.time()
while n_iter < steps_max:
    # Constant warmup rate. #
    step_val = float(max(n_iter+1, warmup_steps))**(-0.5)
    learning_rate = float(hidden_size)**(-0.5) * step_val

    if n_iter >= finetune_step:
        reg_cls = 1.0
        if n_iter < final_tunings:
            learning_rate = 1.0e-4
        else:
            learning_rate = 1.0e-5
    else:
        reg_cls = 0.0
    
    # Randomly sample the training data. #
    batch_sample = np.random.choice(
        num_data, size=batch_size, replace=False)
    batch_mixaug = np.random.choice(
        num_data, size=batch_size, replace=False)
    
    # Reset the values of the inputs and outputs. #
    tmp_out_lab1[:] = 0
    tmp_out_lab2[:] = 0
    tmp_in_seq1[:, :] = PAD_token
    tmp_in_seq2[:, :] = PAD_token

    tmp_in_mask1[:, :] = 0.0
    tmp_in_mask2[:, :] = 0.0
    tmp_out_seq1[:, :] = PAD_token
    tmp_out_seq2[:, :] = PAD_token

    for n_index in range(batch_size):
        tmp_index1 = batch_sample[n_index]
        tmp_label1 = train_data[tmp_index1][0]
        tmp_token1 = train_data[tmp_index1][1]
        
        tmp_index2 = batch_mixaug[n_index]
        tmp_label2 = train_data[tmp_index2][0]
        tmp_token2 = train_data[tmp_index2][1]

        input_tuple1 = generate_input(
            tmp_token1, word_2_idx, vocab_size, 
            ker_sz, seq_length, out_length, 
            CLS_token, EOS_token, MSK_token, TRU_token)
        
        input_tuple2 = generate_input(
            tmp_token2, word_2_idx, vocab_size, 
            ker_sz, seq_length, out_length, 
            CLS_token, EOS_token, MSK_token, TRU_token)
        
        n_input1  = len(input_tuple1[0])
        n_input2  = len(input_tuple2[0])
        n_output1 = len(input_tuple1[1])
        n_output2 = len(input_tuple2[1])
        
        tmp_out_lab1[n_index] = tmp_label1
        tmp_out_lab2[n_index] = tmp_label2

        tmp_in_seq1[n_index, :n_input1] = input_tuple1[0]
        tmp_in_mask1[n_index, :n_output1] = input_tuple1[2]
        tmp_out_seq1[n_index, :n_output1] = input_tuple1[1]

        tmp_in_seq2[n_index, :n_input2] = input_tuple2[0]
        tmp_in_mask2[n_index, :n_output2] = input_tuple2[2]
        tmp_out_seq2[n_index, :n_output2] = input_tuple2[1]
    
    # Update the weights. #
    tmp_loss = sub_batch_train_step(
        bert_model, sub_batch, vocab_size+6, 
        num_class, tmp_in_mask1, tmp_in_mask2, 
        tmp_in_seq1, tmp_in_seq2, tmp_out_seq1, tmp_out_seq2, 
        tmp_out_lab1, tmp_out_lab2, bert_optim, reg_cls=reg_cls, 
        learning_rate=learning_rate, grad_clip=gradient_clip)
    
    # Increment the step. #
    n_iter += 1
    ckpt.step.assign_add(1)
    
    tot_loss += tmp_loss.numpy()
    if n_iter % display_step == 0:
        if n_iter >= finetune_step:
            # Get the validation accuracy. #
            pred_labels = []
            for n_val_batch in range(n_val_batches):
                id_st = n_val_batch * batch_size
                if n_val_batch == (n_val_batches-1):
                    id_en = num_test
                else:
                    id_en = (n_val_batch+1) * batch_size
                curr_batch = id_en - id_st

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
            
            # Compute the accuracy. #
            y_valid  = [x for x, y in valid_data]
            accuracy = np.sum(np.where(
                pred_labels == y_valid, 1, 0)) / num_test
            
            # Generate the classification report. #
            eval_report = "bert_thai_reviews_"
            eval_report += "classification_report_mixup.txt"
            eval_header = "BERT Word Token Classification Report"
            eval_header += " at iteration " + str(n_iter) + " \n"

            pred_report = classification_report(
                y_valid, pred_labels, 
                zero_division=0, target_names=target_labels)
            with open(eval_report, "w") as tmp_write:
                tmp_write.write(eval_header)
                tmp_write.write(pred_report)
            del pred_labels
        else:
            accuracy = 0.0
        
        avg_loss = tot_loss / display_step
        tot_loss = 0.0
        elapsed_tm = (time.time() - start_tm) / 60
        
        print("Iteration:", str(n_iter) + ".")
        print("Elapsed Time:", 
              str(round(elapsed_tm, 2)), "mins.")
        print("Learn Rate:", str(learning_rate) + ".")
        print("Average Train Loss:", str(avg_loss) + ".")
        print("Validation Accuracy:", 
              str(round(accuracy * 100, 2)) + "%.")
        
        start_tm = time.time()
        train_loss_list.append((n_iter, avg_loss, accuracy))
        
        if n_iter % cooling_step != 0:
            print("-" * 50)
    
    if n_iter % save_step == 0:
        # Save the model. #
        save_path = manager.save()
        print("Saved model to {}".format(save_path))
        
        df_col_names  = ["iter", "xent_loss", "val_acc"]
        train_loss_df = pd.DataFrame(
            train_loss_list, columns=df_col_names)
        train_loss_df.to_csv(train_loss_file, index=False)
        del train_loss_df
    
    if n_iter % cooling_step == 0:
        print("Cooling GPU for 2 minutes.")
        time.sleep(120)
        print("-" * 50)
