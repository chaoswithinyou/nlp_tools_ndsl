from word_segmentation.UITws_v1 import UITws_v1

def word_segment(texts):
    model = UITws_v1('base_sep_sfx.pkl')
    segmented_texts = model.segment(texts=texts, pre_tokenized=True, batch_size=256)
    return segmented_texts
