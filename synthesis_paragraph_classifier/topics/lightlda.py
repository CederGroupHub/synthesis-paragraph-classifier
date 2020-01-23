import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

import chemdataextractor.nlp
import collections
import numpy

from synthesis_paragraph_classifier.nlp.preprocessing import TextPreprocessor
from synthesis_paragraph_classifier.nlp.token_storage import LabeledDocumentsReader
from synthesis_paragraph_classifier.nlp.vocabulary import Vocabulary

__author__ = "Haoyan Huo"
__maintainer__ = "Haoyan Huo"
__email__ = "haoyan.huo@lbl.gov"


class TempDirname(object):
    def __init__(self):
        """
        Create a temp dir. The dir is guaranteed to exist in FS.
        """
        self.dirname = tempfile.mkdtemp()
        logging.debug('Created temp dir: %s', self.dirname)

    def __enter__(self):
        return self.dirname

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.dirname)
        logging.debug('Removed temp dir: %s', self.dirname)


class FilterClass(object):
    _stopwords = {
        'a', 'able', 'about', 'above', 'according', 'accordingly', 'across', 'actually', 'after', 'afterwards',
        'again', 'against', 'all', 'allow', 'allows', 'almost', 'alone', 'along', 'already', 'also', 'although',
        'always', 'am', 'among', 'amongst', 'an', 'and', 'another', 'any', 'anybody', 'anyhow', 'anyone', 'anything',
        'anyway', 'anyways', 'anywhere', 'apart', 'appear', 'appreciate', 'appropriate', 'are', 'around', 'as',
        'aside', 'ask', 'asking', 'associated', 'at', 'available', 'away', 'awfully',

        'b', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind',
        'being', 'believe', 'below', 'beside', 'besides', 'best', 'better', 'between', 'beyond', 'both', 'brief',
        'but', 'by',

        'came', 'can', 'cannot', 'cant', 'cause', 'causes', 'certain', 'certainly', 'changes', 'clearly', 'com',
        'come', 'comes', 'concerning', 'consequently', 'consider', 'considering', 'contain', 'containing', 'contains',
        'corresponding', 'could', 'course', 'currently',

        'definitely', 'described', 'despite', 'did', 'different', 'do', 'does', 'doing', 'done', 'down', 'downwards',
        'during',

        'each', 'edu', 'eg', 'either', 'else', 'elsewhere', 'enough', 'entirely', 'especially', 'etc', 'even', 'ever',
        'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'exactly', 'example', 'except',

        'far', 'few', 'fifth', 'first', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth',
        'from', 'further', 'furthermore',

        'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'greetings',

        'had', 'happens', 'hardly', 'has', 'have', 'having', 'he', 'hello', 'help', 'hence', 'her', 'here', 'hereafter',
        'hereby', 'herein', 'hereupon', 'hers', 'herself', 'hi', 'him', 'himself', 'his', 'hither', 'hopefully', 'how',
        'howbeit', 'however',

        'i', 'ie', 'if', 'ignored', 'immediate', 'in', 'inasmuch', 'inc', 'indeed', 'indicate', 'indicated',
        'indicates', 'inner', 'insofar', 'instead', 'into', 'inward', 'is', 'it', 'its', 'itself',

        'just',

        'keep', 'keeps', 'kept', 'know', 'knows', 'known',

        'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'like', 'liked', 'likely',
        'little', 'look', 'looking', 'looks',

        'mainly', 'many', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'merely', 'might', 'more', 'moreover', 'most',
        'mostly', 'much', 'must', 'my', 'myself',

        'name', 'namely', 'nd', 'near', 'nearly', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless',
        'new', 'next', 'no', 'nobody', 'non', 'none', 'noone', 'nor', 'normally', 'not', 'nothing', 'novel', 'now',
        'nowhere',

        'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'only', 'onto', 'or', 'other',
        'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'own',

        'particular', 'particularly', 'per', 'perhaps', 'placed', 'please', 'plus', 'possible', 'presumably',
        'probably', 'provides',

        'que', 'quite', 'qv',

        'rather', 'rd', 're', 'really', 'reasonably', 'regarding', 'regardless', 'regards', 'relatively',
        'respectively', 'right',

        'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed',
        'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'serious', 'seriously', 'several', 'shall',
        'she', 'should', 'since', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
        'somewhat', 'somewhere', 'soon', 'sorry', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup',
        'sure',

        'take', 'taken', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', 'thats', 'the', 'their',
        'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
        'theres', 'thereupon', 'these', 'they', 'think', 'third', 'this', 'thorough', 'thoroughly', 'those', 'though',
        'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries',
        'truly', 'try', 'trying',

        'un', 'under', 'unfortunately', 'unless', 'unlikely', 'until', 'unto', 'up', 'upon', 'us', 'use', 'used',
        'useful', 'uses', 'using', 'usually', 'uucp',

        'value', 'various', 'very', 'via', 'viz', 'vs',

        'want', 'wants', 'was', 'way', 'we', 'welcome', 'well', 'went', 'were', 'what', 'whatever', 'when', 'whence',
        'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which',
        'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'willing', 'wish', 'with',
        'within', 'without', 'wonder', 'would', 'would',

        'yes', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves'
    }
    stopwords = _stopwords | set(x.capitalize() for x in _stopwords)
    cem = chemdataextractor.nlp.cem.CrfCemTagger()

    SAFE_WORDS = ['h', 'h.', 's', 's.', 'CDEMATERIAL']

    def __init__(self, minimum_number_tokens=10):
        self.word_starting_sentence = re.compile(r'^[A-Z][a-z]*$')
        self.word_re = re.compile(r'^[a-zA-Z][a-zA-Z\-.]*$')
        self.number_re = re.compile(r'^[±+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?$')
        self.lang_symbols = re.compile(r'^[.~?><:;,(){}[\]\-–_+=!@#$%^&*|\'"]$')
        self.greek_symbols = re.compile(r'^[αβγδεζηθικλμνξοπρσςτυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ]$')
        self.special_symbols = re.compile(r'^°$')
        self.math_symbols = re.compile(r'^[±+\-×/⇌]+$')

        # TODO: complete this list
        number_re_single = r'[±+\-]?\d+(?:\.\d+)?'
        number_re_single_scientific = number_re_single + r'(?:[eE][+\-]?\d+)?'
        units = r'[GMkmμµnpf]?(?:C|F|g|A|mol|l|L|rpm|wt.?|days?|h|hours?|s|min|minutes?|atm|' \
                r'[cd]?m|K|[gG][pP]a|[Ww]eeks?|[Hh]z|bar|eV|Å|%|Torr|psi|V|mmHg)'
        number_and_range = number_re_single_scientific + r'(?:[\-–_~]+' + number_re_single_scientific + r'?)*'
        num_unit_regex = r'^(?P<number>' + number_and_range + r')?' \
                                                              r'(?P<units>(?:(?:' + units + r')(?:[+\-]?\d+(?:\.\d+)?)?)+)$'
        self.num_unit = re.compile(num_unit_regex)

        self.minimum_number_tokens = minimum_number_tokens

    def __call__(self, orth, lemma, pos):
        if len(orth) == 0:
            return []

        def word_is_material(t, p):
            _, cem_ner = zip(*self.cem.tag([(t, p)]))
            return cem_ner[0] == 'B-CM' or cem_ner[0] == 'I-CM'

        if self.word_starting_sentence.match(orth[0]):
            orth[0] = orth[0].lower()

        new_tokens = []
        for _orth, _lemma, _pos in zip(orth, lemma, pos):
            if self.word_re.match(_orth):
                if _orth not in self.stopwords:
                    new_tokens.append(_orth)
            elif self.number_re.match(_orth):
                new_tokens.append('SYMNUMBER')
            elif self.lang_symbols.match(_orth):
                new_tokens.append('LANGSYM_' + _orth)
            elif self.greek_symbols.match(_orth):
                new_tokens.append('GREEKCHAR_' + _orth)
            elif self.math_symbols.match(_orth):
                new_tokens.append('MATHCHAR_' + _orth)
            elif self.special_symbols.match(_orth):
                new_tokens.append(_orth)
            # elif word_is_material(_orth, _pos):
            #     new_tokens.append('CDEMATERIAL')
            elif self.num_unit.match(_orth):
                m = self.num_unit.match(_orth)
                new_tokens.append('SYMNUMBER')
                new_tokens.append(m.group('units'))
            else:
                pass

        if len(new_tokens) < self.minimum_number_tokens:
            return None

        return new_tokens


class WordLemmaFilterClass(FilterClass):
    def __call__(self, orth, lemma, pos):
        new_tokens = []
        for _lemma in lemma:
            if _lemma.isalpha():
                new_tokens.append(_lemma)

        if len(new_tokens) < 10:
            return None

        return new_tokens


class Filter(FilterClass):
    def __init__(self, *args, **kwargs):
        super(Filter, self).__init__(minimum_number_tokens=0)

        self.bad_symbols = re.compile(r'^SYMNUMBER|^GREEKCHAR|^LANGSYM_|^MATHCHAR_')

    def __call__(self, orth, lemma, pos):
        lst = super(Filter, self).__call__(orth, lemma, pos)
        if lst is None:
            return lst

        new_lst = [i for i in lst if not self.bad_symbols.match(i)]

        if len(new_lst) < self.minimum_number_tokens:
            return None
        return new_lst


class LightLDAOutput(object):
    def __init__(self, output_dir, input_dir=None):
        self.output_dir = output_dir
        self.input_dir = input_dir

        self._check_files()

    def _check_files(self):
        if len(list(self._find_summary_file())) == 0:
            raise ValueError('No topic summary table found.')

        if len(list(self._find_doc_topic_table_sorted())) == 0:
            logging.warning(
                'No doc topic files found at %s, you will not be able to read per-document topics.',
                self.output_dir
            )

        input_file = os.path.join(self.input_dir, 'corpus.libsvm')
        if not os.path.exists(input_file):
            logging.warning(
                'No input file found at %s, you will not be able to read per-document topics.',
                self.input_dir
            )

        if len(list(self._find_word_topic_file())) == 0:
            raise ValueError('No word topic table found.')

        self._find_dict_file()

        if len(list(self._find_log_file())) == 0:
            raise ValueError('No log file found.')

    @staticmethod
    def _parse_line(line):
        if isinstance(line, str):
            separator = r'\s+'
            field_separator = ':'
        elif isinstance(line, bytes):
            separator = br'\s+'
            field_separator = b':'
        else:
            raise TypeError('line must be a sequence.')

        fields = re.split(separator, line.strip())
        if len(fields) < 1:
            raise RuntimeError('No value found for line: ' + repr(line))

        line_label = fields[0]

        values = {}
        for i in fields[1:]:
            column, value = i.split(field_separator)
            try:
                values[int(column)] = int(value)
            except ValueError:
                raise RuntimeError('Cannot parse field into integers: ' + repr(i))

        return line_label, values

    @staticmethod
    def _parse_array_file(filename):
        array = {}
        with open(filename, 'rb') as f:
            for line in f:
                line_label, values = LightLDAOutput._parse_line(line)
                array[int(line_label)] = values

            return array

    def _find_summary_file(self):
        for i in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, i)
            if re.match(r'server_\d+_table_1\.model', i) and os.path.getsize(path) > 2:
                yield path

    def topic_summary_table(self):
        array = {}
        for summary_filename in self._find_summary_file():
            array.update(self._parse_array_file(summary_filename))

        if len(array) != 1 or 0 not in array:
            raise ValueError('Invalid summary array read.')
        return array[0]

    def _find_doc_topic_table_sorted(self):
        doc_topics_filenames = []
        for i in os.listdir(self.output_dir):
            m = re.match(r'doc_topic\.(\d+)\.(\d+)', i)
            if m is not None:
                major, minor = int(m.group(1)), int(m.group(2))
                doc_topics_filenames.append((major, minor, os.path.join(self.output_dir, i)))

        doc_topics_filenames = sorted(doc_topics_filenames)
        return [x[2] for x in doc_topics_filenames]

    def doc_topic_table(self):
        with LabeledDocumentsReader(
                os.path.join(self.input_dir, 'corpus.libsvm')) as doc_ids:
            doc_topic_tables = self._find_doc_topic_table_sorted()

            if len(doc_topic_tables) == 0:
                raise ValueError('No doc topic table found.')

            for filename in doc_topic_tables:
                with open(filename, 'rb') as f:
                    for line in f:
                        doc_label, _ = doc_ids.next()
                        _, values = self._parse_line(line)
                        yield doc_label, values

    def _find_word_topic_file(self):
        for i in os.listdir(self.output_dir):
            if re.match(r'server_\d+_table_0\.model', i):
                yield os.path.join(self.output_dir, i)

    def word_topic_table(self):
        topic_words = {}
        for filename in self._find_word_topic_file():
            topic_words.update(self._parse_array_file(filename))

        return topic_words

    def word_topic_array(self):
        wt_table = self.word_topic_table()
        max_words = max(wt_table.keys())
        max_topics = max(max(x.keys()) for x in wt_table.values()) + 1
        wt_array = numpy.zeros((max_topics, max_words), dtype=int)
        for word, topics in wt_table.items():
            for t, n in topics.items():
                wt_array[t, word-1] = n

        return wt_array

    def word_topic_table_sorted(self):
        topic_words = self.word_topic_table()
        for t, v in topic_words.items():
            s = sum(v.values())
            v = sorted([(y / s, x) for x, y in v.items()], reverse=True)
            topic_words[t] = v
        return topic_words

    def _find_dict_file(self):
        for i in os.listdir(self.input_dir):
            if re.match(r'.*?\.dict$', i):
                return os.path.join(self.input_dir, i)

        raise RuntimeError('No dict found!')

    def dictionary(self):
        with open(self._find_dict_file()) as f:
            dict_ids = {}
            for line in f:
                word_id_s, word_s, count_s = re.split(r'\s+', line.strip())
                word_id, count = int(word_id_s), int(count_s)

                dict_ids[word_id] = (word_s, count)

            return dict_ids

    def _find_log_file(self):
        for i in os.listdir(self.output_dir):
            m = re.match(r'LightLDA\.(\d+)\.\d+\.log', i)
            if m is None:
                continue
            yield int(m.group(1)), os.path.join(self.output_dir, i)

    def model_likelihood(self):
        blocks = {}
        for blockid, logfile in self._find_log_file():
            this_iter = None
            doc_likelihood = []
            word_likelihood = []
            normalized_likelihood = []

            with open(logfile) as f:
                for line in f:
                    m = re.match(r'.*?Iter = (\d+).*', line)
                    if m:
                        this_iter = int(m.group(1))
                        continue

                    m = re.match(r'.*?doc likelihood : (.*)', line)
                    if m:
                        doc_likelihood.append((float(m.group(1)), this_iter))
                        continue

                    m = re.match(r'.*?word likelihood : (.*)', line)
                    if m:
                        word_likelihood.append((float(m.group(1)), this_iter))
                        continue

                    m = re.match(r'.*?Normalized likelihood : (.*)', line)
                    if m:
                        normalized_likelihood.append((float(m.group(1)), this_iter))
                        continue

            y, x = zip(*doc_likelihood)
            doc_likelihood = numpy.array(x), numpy.array(y)

            y, x = zip(*word_likelihood)
            word_likelihood = numpy.array(x), numpy.array(y)

            y, x = zip(*normalized_likelihood)
            normalized_likelihood = numpy.array(x), numpy.array(y)
            assert numpy.equal(doc_likelihood[0], word_likelihood[0]).all()
            assert numpy.equal(doc_likelihood[0], normalized_likelihood[0]).all()

            word_likelihood = word_likelihood[0], word_likelihood[1] + normalized_likelihood[1]

            blocks[blockid] = {
                'doc': doc_likelihood,
                'word': word_likelihood
            }
        return blocks


class LightLDAInference(LightLDAOutput):
    LINE_REGEX = re.compile(r'Topics for \d+: ((?:\d+:\d+\s+)*)\s*$')

    def __init__(self, lightlda_inference, ntopics, nvocabulary, alpha,
                 beta=0.01, niterations=200, mh_steps=10, random_seed=-1,
                 *args, **kwargs):
        super(LightLDAInference, self).__init__(*args, **kwargs)
        self.vocabulary = Vocabulary.load(os.path.join(self.input_dir, 'corpus.dict'))

        self.command_line = [
            lightlda_inference,
            '-rand', '%d' % random_seed,
            '-num_vocabs', '%d' % nvocabulary,
            '-num_topics', '%d' % ntopics,
            '-num_iterations', '%d' % niterations,
            '-mh_steps', '%d' % mh_steps,
            '-alpha', '%f' % alpha,
            '-beta', '%f' % beta,
            '-max_num_document', '1',
            '-input_dir', os.path.realpath(self.output_dir)
        ]
        self.tmp_dir = TempDirname()
        self.tmp_dir_path = None
        self.pipe = None

    def __enter__(self):
        self.tmp_dir_path = self.tmp_dir.__enter__()
        logging.debug('Executing lightlda as %r', self.command_line)
        self.pipe = subprocess.Popen(
            args=self.command_line,
            cwd=os.path.realpath(self.tmp_dir_path),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            encoding='utf8'
        )
        self.pipe.__enter__()

        # This is actually a hack. Make sure the folder is removed.
        self.pipe.stdout.readline()
        self.tmp_dir.__exit__(*sys.exc_info())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pipe.__exit__(exc_type, exc_val, exc_tb)

    def _process_text(self, text):
        processor = TextPreprocessor(text)
        all_lemmas = processor.get_words(lemma=True)
        all_orths = processor.get_words(lemma=False)
        all_pos = processor.get_pos()
        # all_orths, all_pos = [], []

        # for sentence in processor.doc.user_data.sentences:
        #     orths, pos = zip(*sentence.pos_tagged_tokens)
        #     for x in orths: all_orths.append(x)
        #     for x in pos: all_pos.append(x)

        token_filter = Filter(minimum_number_tokens=0)
        tokens = token_filter(all_orths, all_lemmas, all_pos)
        if tokens is None:
            tokens = []

        tokensid = collections.defaultdict(int)
        for i in tokens:
            if i in self.vocabulary.word2id:
                _id = self.vocabulary.word2id[i]
                tokensid[_id] += 1

        if len(tokensid) == 0:
            return None

        input_line = []
        for tokenid, count in tokensid.items():
            input_line.append('%d:%d' % (tokenid, count))
        input_line.append('\n')
        input_line = ' '.join(input_line)

        return input_line

    def _read_one_topic_line(self):
        while True:
            line = self.pipe.stdout.readline()
            logging.debug('lightlda inference returned: %s', line.strip())

            if not line:
                logging.error('lightlda did not give expected output string.')
                raise ValueError('lightlda did not give expected output string.')

            m = self.LINE_REGEX.search(line)
            if m:
                topics_line = m.group(1).strip()
                topics = {}

                if topics_line:
                    for i in re.split(r'\s+', topics_line):
                        t, c = i.split(':')
                        t, c = int(t), int(c)
                        topics[t] = c

                return topics

    def _feed_text_buffered(self, list_of_text, buffer_size=1024):
        """
        Note: list_of_text will be modified.
        """
        while len(list_of_text) > 0:
            text_written = []
            written_bytes = 0

            while written_bytes < buffer_size and len(list_of_text):
                text = list_of_text.pop(0)

                if text is None:
                    text_written.append(False)
                else:
                    self.pipe.stdin.write(text)
                    written_bytes += len(text)
                    text_written.append(True)

            self.pipe.stdin.flush()
            for written in text_written:
                yield written

    def _infer(self, list_of_text, repeat):
        input_lines = []
        if not isinstance(list_of_text, list):
            list_of_text = [list_of_text]
        for text in list_of_text:
            this_line = self._process_text(text)
            for i in range(repeat):
                input_lines.append(this_line)

        for written in self._feed_text_buffered(input_lines):
            if not written:
                yield {}
            else:
                yield self._read_one_topic_line()

    def infer(self, texts, repeat=1):
        status = self.pipe.poll()
        if status is not None:
            raise RuntimeError('lightlda has terminated with error code: %d' % status)

        topic_results = [t for t in self._infer(texts, repeat)]

        return topic_results
