Model_name="/m2m_12B"
tokenizer_name="/m2m_12B"
data_path='facebook/flores'
cache_dir="/cache/"
subset='eng_Latn-arb_Arab'

CUDA_VISIBLE_DEVICES=1 python ./evalute_hf.py  \
    --model_name_or_path $Model_name \
    --tokenizer_name_or_path $tokenizer_name \
    --dataset $data_path \
    --subset $subset \
    --split 'devtest' \
    --src 'sentence_eng_Latn' \
    --tgt 'sentence_arb_Arab' \
    --evaluate_metric 'bleu' 'bertscore' 'chrf' \
    --b_size 1 \
    --cache_dir $cache_dir \
