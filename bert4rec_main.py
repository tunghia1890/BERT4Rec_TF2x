# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --train_input_file=./data/${dataset_name}${signature}.train.tfrecord \
#     --test_input_file=./data/${dataset_name}${signature}.test.tfrecord \
#     --vocab_filename=./data/${dataset_name}${signature}.vocab \
#     --user_history_filename=./data/${dataset_name}${signature}.his \
#     --checkpointDir=${CKPT_DIR}/${dataset_name} \
#     --signature=${signature}-${dim} \
#     --do_train=True \
#     --do_eval=True \
#     --bert_config_file=./bert_train/bert_config_${dataset_name}_${dim}.json \
#     --batch_size=${batch_size} \
#     --max_seq_length=${max_seq_length} \
#     --max_predictions_per_seq=${max_predictions_per_seq} \
#     --num_train_steps=${num_train_steps} \
#     --num_warmup_steps=100 \
#     --learning_rate=1e-4

import os
import pickle
import sys

import numpy as np
import tensorflow as tf

import modeling
import optimization

from data_processing import FreqVocab

bert_config_file = 'data_dir/bert_config_ml-1m_64.json'
checkpoint_dir = 'models/ml-1m'
train_input_file = 'data_dir/ml-1m.train.tfrecord'
test_input_file = 'data_dir/ml-1m.train.tfrecord'
vocab_filename = 'data_dir/ml-1m.vocab'
user_history_filename = 'data_dir/ml-1m.his'
save_checkpoints_steps = 1000
init_checkpoint = None
learning_rate = 1e-4
num_train_steps = 100000
num_warmup_steps = 100
use_tpu = False
batch_size = 32
max_seq_length = 128
max_predictions_per_seq = 20
use_pop_random = True

is_training = True


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.compat.v1.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.compat.v1.variable_scope("transform"):
            dense_input_tensor = tf.keras.layers.Dense(units=bert_config.hidden_size,
                                                       activation=modeling.get_activation(bert_config.hidden_act),
                                                       kernel_initializer=modeling.create_initializer(
                                                           bert_config.initializer_range
                                                       ))
            # input_tensor = tf.layers.dense(
            #     input_tensor,
            #     units=bert_config.hidden_size,
            #     activation=modeling.get_activation(bert_config.hidden_act),
            #     kernel_initializer=modeling.create_initializer(
            #         bert_config.initializer_range))
            input_tensor = dense_input_tensor(input_tensor)
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.compat.v1.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        print("*** Features ***")
        for name in sorted(features.keys()):
            print("  name = %s, shape = %s" % (name, features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        #         all_user_and_item = model.get_embedding_table()
        #         item_ids = [i for i in range(0, item_size + 1)]
        #         softmax_output_embedding = tf.nn.embedding_lookup(all_user_and_item, item_ids)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                                    [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            tf.compat.v1.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.compat.v1.add_to_collection('eval_sp', input_ids)
            tf.compat.v1.add_to_collection('eval_sp', masked_lm_ids)
            tf.compat.v1.add_to_collection('eval_sp', info)

            eval_metrics = metric_fn(masked_lm_example_loss,
                                     masked_lm_log_probs, masked_lm_ids,
                                     masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.compat.v1.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.compat.v1.to_int32(t)
        example[name] = t

    return example


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "info":
                tf.compat.v1.FixedLenFeature([1], tf.int64),  # [user]
            "input_ids":
                tf.compat.v1.FixedLenFeature([max_seq_length], tf.int64, default_value=[0] * max_seq_length),
            "input_mask":
                tf.compat.v1.FixedLenFeature([max_seq_length], tf.int64, default_value=[0] * max_seq_length),
            "masked_lm_positions":
                tf.compat.v1.FixedLenFeature([max_predictions_per_seq], tf.int64,
                                                     default_value=[0] * max_predictions_per_seq),
            "masked_lm_ids":
                tf.compat.v1.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.compat.v1.FixedLenFeature([max_predictions_per_seq], tf.float32,
                                             default_value=[0.0] * max_predictions_per_seq)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            # cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            # d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            # d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d

    return input_fn


class EvalHooks(tf.compat.v1.train.SessionRunHook):
    def __init__(self):
        print('run init')

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ap = 0.0

        np.random.seed(12345)

        self.vocab = None

        if user_history_filename is not None:
            print('load user history from :' + user_history_filename)
            with open(user_history_filename, 'rb') as input_file:
                self.user_history = pickle.load(input_file)

        if vocab_filename is not None:
            print('load vocab from :' + vocab_filename)
            with open(vocab_filename, 'rb') as input_file:
                self.vocab = pickle.load(input_file)

            keys = self.vocab.counter.keys()
            values = self.vocab.counter.values()
            self.ids = self.vocab.convert_tokens_to_ids(keys)
            # normalize
            # print(values)
            sum_value = np.sum([x for x in values])
            # print(sum_value)
            self.probability = [value / sum_value for value in values]

    def end(self, session):
        print(
            "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
            format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
                   self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
                   self.ndcg_10 / self.valid_user,
                   self.hit_10 / self.valid_user, self.ap / self.valid_user,
                   self.valid_user))

    def before_run(self, run_context):
        variables = tf.compat.v1.get_collection('eval_sp')
        return tf.compat.v1.train.SessionRunArgs(variables)

    def after_run(self, run_context, run_values):
        masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, max_predictions_per_seq, masked_lm_log_probs.shape[1]))

        for idx in range(len(input_ids)):
            rated = set(input_ids[idx])
            rated.add(0)
            rated.add(masked_lm_ids[idx][0])
            map(lambda x: rated.add(x),
                self.user_history["user_" + str(info[idx][0])][0])
            item_idx = [masked_lm_ids[idx][0]]
            # here we need more consideration
            masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]
            size_of_prob = len(self.ids) + 1  # len(masked_lm_log_probs_elem)
            if use_pop_random:
                if self.vocab is not None:
                    while len(item_idx) < 101:
                        sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
                        sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
                        item_idx.extend(sampled_ids[:])
                    item_idx = item_idx[:101]
            else:
                # print("evaluation random -> ")
                for _ in range(100):
                    t = np.random.randint(1, size_of_prob)
                    while t in rated:
                        t = np.random.randint(1, size_of_prob)
                    item_idx.append(t)

            predictions = -masked_lm_log_probs_elem[item_idx]
            rank = predictions.argsort().argsort()[0]

            self.valid_user += 1

            if self.valid_user % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            if rank < 1:
                self.ndcg_1 += 1
                self.hit_1 += 1
            if rank < 5:
                self.ndcg_5 += 1 / np.log2(rank + 2)
                self.hit_5 += 1
            if rank < 10:
                self.ndcg_10 += 1 / np.log2(rank + 2)
                self.hit_10 += 1

            self.ap += 1.0 / (rank + 1)


def main(_):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    tf.compat.v1.gfile.MakeDirs(checkpoint_dir)

    train_input_files = []
    for input_pattern in train_input_file.split(","):
        train_input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

    test_input_files = []
    if test_input_file is None:
        test_input_files = train_input_files
    else:
        for input_pattern in test_input_file.split(","):
            test_input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

    print("*** train Input Files ***")
    for input_file in train_input_files:
        print("  %s" % input_file)

    print("*** test Input Files ***")
    for input_file in train_input_files:
        print("  %s" % input_file)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_checkpoints_steps=save_checkpoints_steps)

    if vocab_filename is not None:
        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
    item_size = len(vocab.counter)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=use_tpu,
        use_one_hot_embeddings=use_tpu,
        item_size=item_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            "batch_size": batch_size
        })

    if is_training:
        print("***** Running training *****")
        print("  Batch size = %d", batch_size)
        train_input_fn = input_fn_builder(
            input_files=train_input_files,
            max_seq_length=max_seq_length,
            max_predictions_per_seq=max_predictions_per_seq,
            is_training=True)
        estimator.train(
            input_fn=train_input_fn, max_steps=num_train_steps)

    print("***** Running evaluation *****")
    print("  Batch size = %d", batch_size)

    eval_input_fn = input_fn_builder(
        input_files=test_input_files,
        max_seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        is_training=False)

    # tf.logging.info('special eval ops:', special_eval_ops)
    result = estimator.evaluate(
        input_fn=eval_input_fn,
        steps=None,
        hooks=[EvalHooks()])

    output_eval_file = os.path.join(checkpoint_dir,
                                    "eval_results.txt")
    with tf.compat.v1.gfile.GFile(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        print(bert_config.to_json_string())
        writer.write(bert_config.to_json_string() + '\n')
        for key in sorted(result.keys()):
            print("%s = %s" %(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    tf.compat.v1.app.run()
