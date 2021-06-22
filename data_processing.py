import os
import pickle
from collections import defaultdict
from collections import Counter
import random
import time
import multiprocessing
import collections
import tensorflow as tf

dataset_name = "ml-1m"
max_seq_length = 128
masked_lm_prob = 0.2
max_predictions_per_seq = 20

prop_sliding_window = 0.5
mask_prob = 1.0
dupe_factor = 10
pool_size = 10
random_seed = 12345
short_seq_prob = 0

version_id = ''
output_dir = 'data_dir/'


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class FreqVocab(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, user_to_list):
        # layout of the  ulary
        # item_id based on freq
        # special token
        # user_id based on nothing
        self.counter = Counter(
        )  # sorted(self.items(), key=_itemgetter(1), reverse=True)
        self.user_set = set()
        for u, item_list in user_to_list.items():
            self.counter.update(item_list)
            self.user_set.add(str(u))

        self.user_count = len(self.user_set)
        self.item_count = len(self.counter.keys())
        self.special_tokens = {"[pad]", "[MASK]", '[NO_USE]'}
        self.token_to_ids = {}  # index begin from 1
        # first items
        for token, count in self.counter.most_common():
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # then special tokens
        for token in self.special_tokens:
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        # then user
        #         for user in self.user_set:
        #             self.token_to_ids[user] = len(self.token_to_ids) + 1

        self.id_to_tokens = {v: k for k, v in self.token_to_ids.items()}
        self.vocab_words = list(self.token_to_ids.keys())

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.token_to_ids, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.id_to_tokens, ids)

    def get_vocab_words(self):
        return self.vocab_words  # not in order

    def get_item_count(self):
        return self.item_count

    def get_user_count(self):
        return self.user_count

    def get_items(self):
        return list(self.counter.keys())

    def get_users(self):
        return self.user_set

    def get_special_token_count(self):
        return len(self.special_tokens)

    def get_special_token(self):
        return self.special_tokens

    def get_vocab_size(self):
        return self.get_item_count() + self.get_special_token_count() + 1  # self.get_user_count()


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(fname, 'r') as f:
        lines = list(f.readlines())
        for line in lines:
            line = line.rstrip()
            if ' ' not in line:
                continue
            line_split = line.rsplit(' ')
            u = int(line_split[0])
            i = int(line_split[1])
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[PAD]" or token == '[NO_USE]':
            continue
        last_index = i

    assert last_index > 0

    output_tokens = list(tokens)
    output_tokens[last_index] = "[MASK]"

    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]

    return (output_tokens, masked_lm_positions, masked_lm_labels)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([x for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([x for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([x for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    assert len(document) == 1 and len(document[0]) <= max_num_tokens

    tokens = document[0]
    assert len(tokens) >= 1

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    info = [int(user.split("_")[1])]
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    return [instance]


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < mask_prob:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                masked_token = rng.choice(vocab_words)

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_instances_from_document_train(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng, mask_prob):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]

    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq,
            vocab_items, rng, mask_prob)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def create_instances_threading(all_documents, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    cnt = 0
    start_time = time.time()
    instances = []
    for user in all_documents:
        cnt += 1
        if cnt % 1000 == 0:
            print("step: {}, name: {}, step: {}, time: {}".format(step, multiprocessing.current_process().name, cnt,
                                                                  time.time() - start_time))
            start_time = time.time()
        instances.extend(create_instances_from_document_train(
            all_documents, user, max_seq_length, short_seq_prob,
            masked_lm_prob, max_predictions_per_seq, vocab, rng,
            mask_prob))

    return instances


def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user]
    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split("_")[1])]
    vocab_items = vocab.get_items()

    for tokens in document:
        assert len(tokens) >= 1 and len(tokens) <= max_num_tokens

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
        instance = TrainingInstance(
            info=info,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    return instances


def create_training_instances(all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}

    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence

        sliding_step = (int)(
            prop_sliding_window *
            max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            # todo: add slide
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = list(range(len(item_seq) - max_num_tokens, 0, -sliding_step))
                beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]

    instances = []
    if force_last:
        for user in all_documents:
            instances.extend(
                create_instances_from_document_test(
                    all_documents, user, max_seq_length))
        print("num of instance:{}".format(len(instances)))
    else:
        start_time = time.time()
        pool = multiprocessing.Pool(processes=pool_size)
        instances = []
        print("document num: {}".format(len(all_documents)))

        def log_result(result):
            print("callback function result type: {}, size: {} ".format(type(result), len(result)))
            instances.extend(result)

        for step in range(dupe_factor):
            pool.apply_async(
                create_instances_threading, args=(
                    all_documents, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, random.Random(random.randint(1, 10000)),
                    mask_prob, step), callback=log_result)
        pool.close()
        pool.join()

        for user in all_documents:
            instances.extend(
                mask_last(
                    all_documents, user, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab, rng))

        print("num of instance:{}; time:{}".format(len(instances), time.time() - start_time))
    rng.shuffle(instances)
    return instances


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.compat.v1.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except:
            print(instance)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["info"] = create_int_feature(instance.info)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(
            masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            print("*** Example ***")
            print("tokens: %s" % " ".join([x for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                print("%s: %s" % (feature_name," ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    print("Wrote %d total instances", total_written)


def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                prop_sliding_window,
                pool_size,
                force_last=False):
    # create train
    instances = create_training_instances(
        data, max_seq_length, dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, rng, vocab, mask_prob, prop_sliding_window,
        pool_size, force_last)

    write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    [output_filename])


def gen_data():
    os.makedirs(output_dir, exist_ok=True)

    dataset = data_partition(output_dir + dataset_name + '.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    cc = 0.0
    max_len = 0
    min_len = 100000
    for u in user_train:
        cc += len(user_train[u])
        max_len = max(len(user_train[u]), max_len)
        min_len = min(len(user_train[u]), min_len)

    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))

    print('len_train:{}, len_valid:{}, len_test:{}, usernum:{}, itemnum:{}'.
        format(
        len(user_train),
        len(user_valid), len(user_test), usernum, itemnum))

    for idx, u in enumerate(user_train):
        if idx < 10:
            print(user_train[u])
            print(user_valid[u])
            print(user_test[u])

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }
    rng = random.Random(random_seed)

    vocab = FreqVocab(user_test_data)
    user_test_data_output = {
        k: [vocab.convert_tokens_to_ids(v)]
        for k, v in user_test_data.items()
    }

    print('begin to generate train')
    output_filename = output_dir + dataset_name + version_id + '.train.tfrecord'
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        prop_sliding_window,
        pool_size,
        force_last=False)
    print('train:{}'.format(output_filename))

    print('begin to generate test')
    output_filename = output_dir + dataset_name + version_id + '.test.tfrecord'
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        -1.0,
        pool_size,
        force_last=True)
    print('test:{}'.format(output_filename))

    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
          format(vocab.get_vocab_size(),
                 vocab.get_user_count(),
                 vocab.get_item_count(),
                 vocab.get_item_count() + vocab.get_special_token_count()))
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)

    his_file_name = output_dir + dataset_name + version_id + '.his'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    print('done.')


if __name__ == '__main__':
    gen_data()
