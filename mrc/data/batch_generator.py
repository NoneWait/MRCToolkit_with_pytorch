import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, SequentialSampler, RandomSampler
import torch
import logging
import multiprocessing
import six
import numpy as np

class BatchGenerator(object):
    """
    用来封装data的类(使用TensorDataset)，配置sample(SequentialSampler, RandomSampler),
    封装返回一个Data loader

    official: Mapping words and tags to indices, padding length-variable features,
    transforming all of the features into tensors, and then batching them
    """
    def __init__(self, vocab, instances, batch_size=32,
                 use_char=True, training=False, additional_fields=None,
                 feature_vocab=None, num_parallel_calls=0, shuffle=True):
        self.instances = instances
        self.vocab = vocab
        self.batch_size = batch_size
        self.use_char = use_char
        self.training = training
        self.shuffle_ratio = shuffle_ratio
        self.num_parallel_calls = num_parallel_calls if \
            num_parallel_calls > 0 else multiprocessing.cpu_count()/2

        if self.instances is None or len(self.instances) == 0:
            raise ValueError('empty instances!!')

        self.additional_fields = additional_fields if additional_fields is not None else list()
        self.feature_vocab = feature_vocab if feature_vocab is not None else dict()

        # Tensor Dataset
        self.dataset = self.build_input_pipeline()

        # DataLoader
        self.data_loader = DataLoader(dataset=self.dataset,shuffle=shuffle,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_parallel_calls)

    def get_instances_size(self):
        return len(self.instances)

    def get_batch_size(self):
        return self.batch_size

    def get_instances(self):
        return self.instances

    def get_data_loader(self):
        return self.data_loader

    @staticmethod
    def detect_input_type(instance, additional_fields=None):
        instance_keys = instance.keys()
        fields = ['context_tokens', 'question_tokens', 'answer_start', 'answer_end']
        try:
            for f in fields:
                assert f in instance_keys
        except:
            raise ValueError('A instance should contain at least "context_tokens", "question_tokens", \
                                 "answer_start", "answer_end" four fields!')

        if additional_fields is not None and isinstance(additional_fields, list):
            fields.extend(additional_fields)

        def get_type(value):
            if isinstance(value, six.string_types):
                return six.string_types
            elif isinstance(value, bool):
                return bool
            # TODO int32还是int64(long)?
            elif isinstance(value, int):
                return torch.long
            elif isinstance(value, float):
                return torch.float32
            else:
                return None

        input_type = {}

        for field in fields:
            """
            field 分为多种:
            1. None, 比如test集中没有answer字段，则设为None
            2. list, 比如tokens, 是一个列表
            3. 单个, 比如文章长度
            """
            if instance[field] is None:
                # Bug 应该
                if field not in ('answer_start', 'answer_end'):
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)
                continue
            elif isinstance(instance[field], list):
                # 如果是数组
                if len(instance[field]) == 0:
                    logging.warning('Data shape of field "%s" not detected! Skip this field.', field)
                    continue

                field_type = get_type(instance[field][0])
                if field_type is not None:
                    input_type[field] = field_type
                    # input_shape[field] = tf.TensorShape([None])
                else:
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)
            else:
                field_type = get_type(instance[field])
                if field_type is not None:
                    input_type[field] = field_type
                    # input_shape[field] = tf.TensorShape([])
                else:
                    logging.warning('Data type of field "%s" not detected! Skip this field.', field)

        return fields, input_type

    def build_input_pipeline(self):
        """
        # pipeline
        1. trans to 转换成对应的id
        2. padding 进行padding
        3. trans to torch.tensor
        4. return a MRCDataset
        我们选用TensorDataset来作为dataset的具体类
        :return:
        """
        # instances 里有哪些字段，字段的类型，和输入的大小
        input_fields, input_type_dict = \
            BatchGenerator.detect_input_type(self.instances[0], self.additional_fields)

        # 词
        word_vocab = self.vocab.get_word_vocab
        word_table = self.vocab.get_word2idx
        # 字符
        char_vocab = self.vocab.get_char_vocab
        char_table = self.vocab.get_char2idx
        # 特征字典
        if len(self.feature_vocab) > 0:
            pass
        # 2. Character extracting function
        def extract_char(token, default_value="<PAD>"):
            # TODO
            out = token.split(delimiter='')
            # out = tf.sparse.to_dense(out, default_value=default_value)
            return out

        # 处理一个样本，返回这个样本里的tokens_ids和char_ids
        # 类似生成features
        def transform_new_instance(instance):
            context_tokens = instance['context_tokens']
            question_tokens = instance['question_tokens']

            if self.use_char:
                context_char = extract_char(context_tokens)
                question_char = extract_char(question_tokens)
                instance['context_char'] = [char_table[ch] for ch in context_char]
                instance['question_char'] = [char_table[ch] for ch in question_char]
            if self.vocab.do_lowercase:
                lower_context_tokens = [token.lower() for token in context_tokens]
                lower_question_tokens = [token.lower() for token in question_tokens]

                instance['context_word'] = [word_table[token] for token in lower_context_tokens]
                instance['question_word'] = [word_table[token] for token in lower_question_tokens]
            else:
                instance['context_word'] = [word_table[token] for token in context_tokens]
                instance['question_word'] = [word_table[token] for token in question_tokens]

            instance['context_len'] = [len(context_tokens)]
            instance['question_len'] = [len(question_tokens)]


            # TODO 额外的特征怎么弄？
            # if len(self.feature_vocab) > 0:
            #     for field in self.additional_fields:
            #         for feature_name, table in feature_table.items():
            #             if field.endswith(feature_name):
            #                 # TODO
            #                 instance[field] = tf.cast(table.lookup(instance[field]), tf.int32)
            #                 break
            return instance

        new_instance = [transform_new_instance(instance) for instance in self.instances]


        return Dataset()


class MRCDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)
