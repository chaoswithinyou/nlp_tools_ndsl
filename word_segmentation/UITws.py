from word_segmentation.UITws_v1 import UITws_v1

def word_segment(texts, single_text=False):
    model = UITws_v1('word_segmentation/base_sep_sfx.pkl')
    if single_text==False:
        segmented_texts = model.segment(texts=texts, pre_tokenized=True)
    else:
        texts = nltk.sent_tokenize(texts)
        segmented_texts = model.segment(texts=texts, pre_tokenized=True)
    return segmented_texts