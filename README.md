## Synthesis paragraph classifier

This repo contains codes and data for the paper "Huo, Haoyan, et al. "Semi-supervised machine-learning classification of materials synthesis procedures." npj Computational Materials 5.1 (2019): 1-7.".

## Prerequisites

To run this code, you must compile and install [LightLDA](https://github.com/hhaoyan/LightLDA) (a LDA implementation by Microsoft Research), and put `infer_singlethread` in your `PATH` environment variable.

Pre-compiled binaries can be found [here](https://github.com/hhaoyan/LightLDA/tree/master/binaries).

## Example usage

The basic usage is to classify a paragraph:
```bash
$ python
Python 3.7.4 (default, Aug 13 2019, 15:17:50) 
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from synthesis_paragraph_classifier.classifier import *
>>> model = get_default_model()
WARNING:root:No doc topic files found at /Users/huohaoyan/Projects/PhD-research/Codes/synthesis-paragraph-classifier/synthesis_paragraph_classifier/data/topic_100_p, you will not be able to read per-document topics.
WARNING:root:No input file found at /Users/huohaoyan/Projects/PhD-research/Codes/synthesis-paragraph-classifier/synthesis_paragraph_classifier/data/topic_100_p, you will not be able to read per-document topics.
WARNING:root:No doc topic files found at /Users/huohaoyan/Projects/PhD-research/Codes/synthesis-paragraph-classifier/synthesis_paragraph_classifier/data/topic_200_s, you will not be able to read per-document topics.
WARNING:root:No input file found at /Users/huohaoyan/Projects/PhD-research/Codes/synthesis-paragraph-classifier/synthesis_paragraph_classifier/data/topic_200_s, you will not be able to read per-document topics.
>>> model.classify_paragraph("Al2(WO4)3 is prepared by conventional solid state reaction of Al2O3 and WO3. The compound is found to be phase pure (JCPDS 24-1101) from powder XRD. The powdered product is made into rectangular pellets (∼3×2×0.2mm3) and sintered at 1100 °C to obtain dense pellets for resistivity measurements. High pressure resistivity measurements are carried out using the BA apparatus, of 12.5mm face diameter. The cell assembly consists of two pyrophyllite gaskets of 12.5mm OD, 3mm ID and 0.22mm thickness and two talc disks, which act as pressure transmitting medium. The sample is placed in between the two gaskets with a talc disk below the sample and another on the top. Resistance is measured by standard four-probe technique at various frequencies (12Hz to 100kHz) using GR 1689 Precision RLC Digibridge.")
('solid_state_ceramic_synthesis', 0.8, {'predictions': {'solid_state_ceramic_synthesis': 0.8, 'something_else': 0.1, 'precipitation_ceramic_synthesis': 0.1}, 'trials': [{'paragraph_topics': '{"4": 0.014, "22": 0.014, "26": 0.192, "39": 0.63, "42": 0.014, "68": 0.014, "76": 0.027, "88": 0.041, "93": 0.014, "94": 0.041}', 'sentence_topics': '[[[0, 76], {"169": 0.25, "195": 0.75}], [[77, 148], {"17": 0.167, "47": 0.167, "153": 0.667}], [[149, 294], {"52": 0.25, "144": 0.083, "164": 0.083, "171": 0.417, "179": 0.167}], [[295, 398], {"44": 0.111, "52": 0.111, "57": 0.111, "95": 0.333, "157": 0.111, "161": 0.222}], [[399, 558], {"77": 0.5, "79": 0.111, "95": 0.278, "189": 0.056, "193": 0.056}], [[559, 664], {"28": 0.4, "77": 0.4, "116": 0.2}], [[665, 800], {"52": 0.917, "100": 0.083}]]', '_decision_path': {'solid_state_ceramic_synthesis': '0:66;1:114;2:125;3:112;4:911;5:178;6:731;8:604', 'something_else': '7:77', 'precipitation_ceramic_synthesis': '9:108'}}]})
```

You can also obtain feature vectors for new paragraphs:

```bash
>>> model.feature_names
['high_freq_topic_106', 'high_freq_topic_11', 'high_freq_topic_14', 'high_freq_topic_140', 'high_freq_topic_142', 'high_freq_topic_16', 'high_freq_topic_164', 'high_freq_topic_171', 'high_freq_topic_179', 'high_freq_topic_181', 'high_freq_topic_187', 'high_freq_topic_191', 'high_freq_topic_193', 'high_freq_topic_195', 'high_freq_topic_2', 'high_freq_topic_20', 'high_freq_topic_32', 'high_freq_topic_37', 'high_freq_topic_59', 'high_freq_topic_6', 'high_freq_topic_61', 'high_freq_topic_67', 'high_freq_topic_70', 'high_freq_topic_76', 'high_freq_topic_81', 'high_freq_topic_95', 'high_freq_topic_99', 'high_freq_topic_ngram_(11, 179)', 'high_freq_topic_ngram_(140, 181)', 'high_freq_topic_ngram_(140, 181, 11)', 'high_freq_topic_ngram_(140, 181, 81)', 'high_freq_topic_ngram_(171, 171)', 'high_freq_topic_ngram_(171, 179)', 'high_freq_topic_ngram_(179, 171)', 'high_freq_topic_ngram_(179, 171, 171)', 'high_freq_topic_ngram_(179, 171, 179)', 'high_freq_topic_ngram_(179, 179)', 'high_freq_topic_ngram_(179, 193)', 'high_freq_topic_ngram_(179, 193, 171)', 'high_freq_topic_ngram_(181, 11)', 'high_freq_topic_ngram_(181, 11, 179)', 'high_freq_topic_ngram_(181, 181)', 'high_freq_topic_ngram_(181, 181, 11)', 'high_freq_topic_ngram_(181, 181, 81)', 'high_freq_topic_ngram_(181, 81)', 'high_freq_topic_ngram_(181, 81, 11)', 'high_freq_topic_ngram_(191, 140)', 'high_freq_topic_ngram_(191, 140, 181)', 'high_freq_topic_ngram_(191, 181)', 'high_freq_topic_ngram_(191, 181, 11)', 'high_freq_topic_ngram_(191, 181, 81)', 'high_freq_topic_ngram_(191, 20)', 'high_freq_topic_ngram_(193, 171)', 'high_freq_topic_ngram_(193, 171, 171)', 'high_freq_topic_ngram_(193, 171, 179)', 'high_freq_topic_ngram_(193, 179)', 'high_freq_topic_ngram_(193, 179, 171)', 'high_freq_topic_ngram_(193, 179, 193)', 'high_freq_topic_ngram_(193, 193)', 'high_freq_topic_ngram_(193, 193, 171)', 'high_freq_topic_ngram_(193, 193, 179)', 'high_freq_topic_ngram_(193, 193, 193)', 'high_freq_topic_ngram_(193, 6)', 'high_freq_topic_ngram_(195, 179)', 'high_freq_topic_ngram_(195, 179, 171)', 'high_freq_topic_ngram_(195, 193)', 'high_freq_topic_ngram_(195, 193, 171)', 'high_freq_topic_ngram_(195, 193, 179)', 'high_freq_topic_ngram_(195, 193, 193)', 'high_freq_topic_ngram_(195, 193, 6)', 'high_freq_topic_ngram_(195, 195)', 'high_freq_topic_ngram_(195, 195, 179)', 'high_freq_topic_ngram_(195, 195, 193)', 'high_freq_topic_ngram_(20, 181, 11)', 'high_freq_topic_ngram_(32, 142)', 'high_freq_topic_ngram_(32, 191)', 'high_freq_topic_ngram_(32, 191, 140)', 'high_freq_topic_ngram_(32, 191, 181)', 'high_freq_topic_ngram_(32, 195)', 'high_freq_topic_ngram_(32, 76)', 'high_freq_topic_ngram_(32, 76, 76)', 'high_freq_topic_ngram_(59, 191)', 'high_freq_topic_ngram_(6, 179)', 'high_freq_topic_ngram_(70, 179)', 'high_freq_topic_ngram_(76, 20)', 'high_freq_topic_ngram_(76, 70)', 'high_freq_topic_ngram_(76, 76)', 'high_freq_topic_ngram_(76, 76, 76)', 'high_freq_topic_ngram_(81, 11)']
>>> features = model.featurize_paragraph("Al2(WO4)3 is prepared by conventional solid state reaction of Al2O3 and WO3. The compound is found to be phase pure (JCPDS 24-1101) from powder XRD. The powdered product is made into rectangular pellets (∼3×2×0.2mm3) and sintered at 1100 °C to obtain dense pellets for resistivity measurements. High pressure resistivity measurements are carried out using the BA apparatus, of 12.5mm face diameter. The cell assembly consists of two pyrophyllite gaskets of 12.5mm OD, 3mm ID and 0.22mm thickness and two talc disks, which act as pressure transmitting medium. The sample is placed in between the two gaskets with a talc disk below the sample and another on the top. Resistance is measured by standard four-probe technique at various frequencies (12Hz to 100kHz) using GR 1689 Precision RLC Digibridge.")
>>> type(features)
<class 'scipy.sparse.csr.csr_matrix'>
>>> features.toarray()
array([[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.417, 0.167,
        0.   , 0.   , 0.   , 0.   , 0.75 , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.333, 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ]])
```

To understand the meaning of each topic:
```
>>> wt_array = model.sentence_model.word_topic_array()
>>> vocab = model.sentence_model.dictionary()
>>> topic = 106
>>> # Must index vocab by x+1 because dictionary starts with 1
>>> [vocab[x+1] for x in wt_array[topic].argsort()[::-1][:20]]
[('CDEMATERIAL', 7019040), ('catalysts', 69308), ('prepared', 445882), ('catalyst', 186558), ('Pt', 74807), ('Pd', 35475), ('impregnation', 19050), ('wt', 125222), ('wt.', 94609), ('support', 28711), ('supported', 21339), ('metal', 87752), ('Ni', 43061), ('method', 200712), ('loading', 46317), ('Au', 33957), ('Ru', 10093), ('aqueous', 200616), ('Co', 25039), ('solution', 1024075)]
>>> # Seems topic 106 is about making catalysts.
```
## Citing

```
@article{huo2019semi,
  title={Semi-supervised machine-learning classification of materials synthesis procedures},
  author={Huo, Haoyan and Rong, Ziqin and Kononova, Olga and Sun, Wenhao and Botari, Tiago and He, Tanjin and Tshitoyan, Vahe and Ceder, Gerbrand},
  journal={npj Computational Materials},
  volume={5},
  number={1},
  pages={1--7},
  year={2019},
  publisher={Nature Publishing Group}
}
```
