import json
import os
import pickle

import collections
import numpy
import yaml
from chemdataextractor.nlp import ChemSentenceTokenizer
from sklearn import ensemble
from sklearn.ensemble.forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from synthesis_paragraph_classifier.topics import LightLDAInference


class SynthesisClassifier(object):
    """
    !WARNING!: Careful when editing this class. You might destroy all the pickle'd classifiers.
    """

    def __init__(self, featurizer_list, lda_sentence_model, lda_paragraph_model, dt_classifier=None):

        self.featurizer_list = [x() for x in featurizer_list]
        self.lda_sentence_model = lda_sentence_model
        self.lda_paragraph_model = lda_paragraph_model

        self.vectorizer = DictVectorizer()
        self.label_encoder = LabelEncoder()

        if dt_classifier is None:
            self.dt_classifier = ensemble.RandomForestClassifier(oob_score=True)
        else:
            self.dt_classifier = dt_classifier
        self.model_type = 'RandomForest'
        self.training_data_set = None  # if None, then model is empty

    def check_oob(self, x, y):
        n_samples = y.shape[0]
        in_sample_tensor = numpy.zeros(shape=(
            len(self.dt_classifier.estimators_),
            x.shape[0],
        ))
        out_sample_tensor = numpy.zeros(shape=(
            len(self.dt_classifier.estimators_),
            x.shape[0],
        ))

        for i, estimator in enumerate(self.dt_classifier.estimators_):
            unsampled_indices = _generate_unsampled_indices(
                estimator.random_state, n_samples)
            sampled_indices = _generate_sample_indices(
                estimator.random_state, n_samples)

            assert len(set(unsampled_indices) & set(sampled_indices)) == 0

            unsampled_estimated = estimator.predict(x[unsampled_indices, :])
            unsampled_real = y[unsampled_indices]
            sample_estimated = estimator.predict(x[sampled_indices, :])
            sample_real = y[sampled_indices]

            out_sample_success = numpy.where(unsampled_estimated.astype(int) == unsampled_real)
            out_sample_fail = numpy.where(unsampled_estimated.astype(int) != unsampled_real)
            out_sample_success_indices = unsampled_indices[out_sample_success]
            out_sample_fail_indices = unsampled_indices[out_sample_fail]
            out_sample_tensor[i, out_sample_success_indices] = 1.0
            out_sample_tensor[i, out_sample_fail_indices] = -1.0

            in_sample_success = numpy.where(sample_estimated.astype(int) == sample_real)
            in_sample_fail = numpy.where(sample_estimated.astype(int) != sample_real)
            in_sample_success_indices = sampled_indices[in_sample_success]
            in_sample_fail_indices = sampled_indices[in_sample_fail]
            in_sample_tensor[i, in_sample_success_indices] = 1.0
            in_sample_tensor[i, in_sample_fail_indices] = -1.0

        return in_sample_tensor, out_sample_tensor, y

    def train_model(self, training_data, y_to_learn):
        if self.training_data_set is not None:
            raise RuntimeError('Train a existing model is not permitted!')

        features = [{} for x in training_data]

        for featurizer in self.featurizer_list:
            featurizer.learn_features(training_data, y_to_learn)
            for i, j in zip(features, featurizer.featurize(training_data)):
                i.update(j)

        features_vectorized = self.vectorizer.fit_transform(features)
        x = features_vectorized.toarray()

        labels = [x['y'] for x in training_data]
        y = self.label_encoder.fit_transform(labels)

        self.dt_classifier.fit(x, y)

        self.training_data_set = {
            'feature_names': self.vectorizer.get_feature_names(),
            'class_names': self.label_encoder.classes_.tolist(),
            'training_data': []
        }
        for t, _x, _y in zip(training_data, x, y):
            self.training_data_set['training_data'].append({
                'doi': t['doi'],
                'paragraph_id': t['paragraph_id'],
                'x': _x.tolist(),
                'y': _y.tolist()
            })

        return self.check_oob(x, y)

    def dump_pickle(self):
        return pickle.dumps(self)

    @staticmethod
    def load_pickle(s):
        """

        :param s:
        :return:
        :rtype: SynthesisClassifier
        """
        return pickle.loads(s)

    def featurize(self, topics):
        features = {}
        for featurizer in self.featurizer_list:
            features.update(featurizer.featurize([topics])[0])
        features_vectorized = self.vectorizer.transform([features])
        return features_vectorized[0]

    def predict(self, data_with_topics, return_decision_path=False):
        features = [{} for x in data_with_topics]
        for featurizer in self.featurizer_list:
            for i, j in zip(features, featurizer.featurize(data_with_topics)):
                i.update(j)

        features_vectorized = self.vectorizer.transform(features)
        x = features_vectorized.toarray()

        for i, j in zip(data_with_topics, self.label_encoder.inverse_transform(self.dt_classifier.predict(x))):
            i['y_predicted'] = j

        if not return_decision_path:
            return data_with_topics
        else:
            indicators, n_nodes_ptr = self.dt_classifier.decision_path(x)

            starting_nodes = n_nodes_ptr.tolist()
            all_paths = []

            for path in indicators:
                path = numpy.where(path.toarray().flatten() != 0)[0].tolist()

                paths = []
                current_path = [path.pop(0)]
                current_weight = 0
                while path:
                    if path[0] in starting_nodes:
                        paths.append(current_path)
                        current_weight = path[0]
                        current_path = []
                    current_path.append(path.pop(0) - current_weight)
                if current_path:
                    paths.append(current_path)

                decision_path = []
                for n, p in enumerate(paths):
                    classes = self.dt_classifier.estimators_[n].tree_.value[p[-1]]
                    final_vote = self.label_encoder.classes_[classes.argmax()]

                    decision_path.append({
                        'decision_tree_id': n,
                        'path_seq': p,
                        'voted': final_vote
                    })

                all_paths.append(decision_path)
            return data_with_topics, all_paths


class SynthesisClassification(object):
    def __init__(self, classification_model_path, topic_model_configs):
        self.span_tokenizer = ChemSentenceTokenizer()
        with open(classification_model_path, 'rb') as f:
            self.classification_model = SynthesisClassifier.load_pickle(f.read())

        self.paragraph_model = LightLDAInference(**topic_model_configs[self.classification_model.lda_paragraph_model])
        self.sentence_model = LightLDAInference(**topic_model_configs[self.classification_model.lda_sentence_model])
        self.paragraph_model.__enter__()
        self.sentence_model.__enter__()

    @staticmethod
    def _convert_to_fraction(d):
        result = {}
        total = sum(d.values())
        for _id, val in d.items():
            result[_id] = round(val / total, 3)
        return result

    def _calculate_topics(self, text, repeat):
        trials = [{} for i in range(repeat)]

        # paragraph

        paragraph_topics = self.paragraph_model.infer(text, repeat=repeat)
        for i in range(repeat):
            trials[i]['paragraph_topics'] = self._convert_to_fraction(paragraph_topics[i])

        # sentence

        spans = self.span_tokenizer.span_tokenize(text)
        span_start, span_end, sentence_text, topic_result = [], [], [], []
        for span in spans:
            span_start.append(span[0])
            span_end.append(span[1])
            sentence_text.append(text[span[0]:span[1]])

        for i in range(repeat):
            trials[i]['sentence_topics'] = []

        sentence_topics = self.sentence_model.infer(sentence_text, repeat=repeat)
        for i, (a, b) in enumerate(zip(span_start, span_end)):
            for j in range(repeat):
                trials[j]['sentence_topics'].append(
                    (
                        (a, b),
                        self._convert_to_fraction(sentence_topics[i * repeat + j]))
                )

        return trials

    @property
    def feature_names(self):
        """
        Returns feature names of the vectors from featurize_paragraph().
        :return: list of feature names
        """
        return self.classification_model.vectorizer.get_feature_names().copy()

    def featurize_paragraph(self, paragraph):
        """
        Featurize a text paragraph, return a single sparse vector.

        :param paragraph: Paragraph string to be featurized.
        :type paragraph: str
        :return: Feature of this paragraph
        :rtype: scipy.sparse.csr.csr_matrix
        """
        topics = self._calculate_topics(paragraph, 1)[0]
        return self.classification_model.featurize(topics)

    def classify_paragraph(self, paragraph, restart=1):
        """
        Classify a text paragraph.

        :param paragraph: Paragraph string to be featurized.
        :type paragraph: str
        :param restart: Times to restart classification. DO NOT USE.
        :return: (label_predicted, confidence_score, detailed_result)
        :rtype: tuple(str, float, dict)
        """
        repeated_documents = self._calculate_topics(paragraph, restart)

        predictions, decision_paths = self.classification_model.predict(
            repeated_documents,
            return_decision_path=True
        )

        classification_trials = []
        all_voted = []

        for p, d in zip(predictions, decision_paths):
            trial = {
                'paragraph_topics': json.dumps(p['paragraph_topics']),
                'sentence_topics': json.dumps(p['sentence_topics'])
            }

            decisions = collections.defaultdict(list)
            for i in sorted(d, key=lambda x: x['decision_tree_id']):
                decisions[i['voted']].append('%d:%d' % (i['decision_tree_id'], i['path_seq'][-1]))
            decisions = {x: ';'.join(y) for x, y in decisions.items()}
            trial['_decision_path'] = decisions

            sentence_borders = [x[0] for x in p['sentence_topics']]
            if len(sentence_borders) <= 1:
                for i in d:
                    all_voted.append('something_else')
                trial['_decision_path']['remarks'] = 'Classification stopped by the limit on sentence size.'
            else:
                voted = [x['voted'] for x in d]
                all_voted += voted

            classification_trials.append(trial)

        all_predictions = collections.Counter(all_voted)
        all_predictions = {x: float(y) / len(all_voted) for x, y in all_predictions.items()}
        y_predicted, confidence = max(all_predictions.items(), key=lambda x: x[1])

        result = {
            'predictions': all_predictions,
            'trials': classification_trials
        }

        return y_predicted, confidence, result


def get_default_model():
    package_path = os.path.dirname(os.path.realpath(__file__))

    config_path = os.path.join(
        package_path,
        'data',
        'SynthesisClassification.yml')
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    for model, config in configs['lightlda_models'].items():
        config['input_dir'] = os.path.join(package_path, 'data', config['input_dir'])
        config['output_dir'] = os.path.join(package_path, 'data', config['output_dir'])

    model_path = os.path.join(
        package_path,
        'data',
        configs['model_path'])

    return SynthesisClassification(model_path, configs['lightlda_models'])
