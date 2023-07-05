import os
import argparse, logging
import torch
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,M2M100Config,M2M100Tokenizer
from datasets import load_dataset
import json

def get_args():
    Parser = argparse.ArgumentParser(description="Machine Translation Evalution")
    Parser.add_argument(
        '--model_name_or_path',
        type=str,
        help = 'the name or path of the model to use in the test.',
        required=True
    )
    Parser.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        help = 'the name or path of the tokenizer to use in the test',
        required=True
    )
    Parser.add_argument(
        '--dataset',
         type=str,
          help = 'The name of the dataset to use for test.',
           required=True
       )
    Parser.add_argument(
        '--subset',
         type=str,
          help = 'The name of the dataset to use for test.',
           required=True
       )    
    Parser.add_argument(
        '--split',
        type=str,
        default = 'test',
        help ='dataset split to be use in the test'
    )
    Parser.add_argument(
        "--search_method",
        type=str,
        default='beam',
        help='Choose decoding method'
    )
    Parser.add_argument(
        "--cache_dir",
        type=str,
        default = '/media/khalid/data_disk/cache/',
        help = ''
    )
    Parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help='the device in which the test will run on.'
    )
    Parser.add_argument(
        '--src',
        type=str,
        default= 'en',
        help = 'target language code.'
    )
    Parser.add_argument(
        '--tgt',
        type=str,
        default= 'ar',
        help = 'target language code.',
        required=True
    )
    Parser.add_argument(
        '--save_translation',
        type=str,
        default= True,
        help = 'Use this argument if you want to save the translation.'
    )
    Parser.add_argument(
        '--evaluate_metric',
        nargs='+',
        default= ['bleu'],
        help = 'Use this argument to indicate the metric used for evaluation. You can use more than one metric.'
    )
    Parser.add_argument(
        '--b_size',
        type=int,
        default= 32,
        help = 'the number of example in each batch.'
    )
    Parser.add_argument(
        '--save_path',
        type=str,
        help = 'Use this argument if you want for the path where you want to save the result.'
    )

    args = Parser.parse_args()
    return args



def translate(text):
    inputs = tokenizer(text[args.src], return_tensors="pt",truncation=True,padding='longest')
    translated_tokens = model.generate(
    **inputs.to(args.device), forced_bos_token_id=tokenizer.lang_code_to_id['ar'], max_length=30,num_beams=3,
    )
    return {'pred': tokenizer.batch_decode(translated_tokens, skip_special_tokens=True) }


def map_base(text):
    return {'en': text['translation']['en'],
            'ar': text['translation']['ar']}
def map_flores(text):
    return {'eng_Latn': text['eng_Latn'],
            'arb_Arab': text['sentence']}


def spm(text):
    text['pred'] = tokenizer(text['pred'], return_tensors="pt",truncation=True,padding='longest')['input_ids']
    text[args.tgt] = tokenizer(text[args.tgt], return_tensors="pt",truncation=True,padding='longest')['input_ids']
    return text

def sacrebleu_score (pred, reference):
    sacrebleu = evaluate.load("sacrebleu")
    return sacrebleu.compute(predictions=pred, references=reference)

def gleu_score(pred, reference):
    gleu = evaluate.load('gleu','qqb')
    return gleu.compute(predictions=pred, references=reference)

def chrf_score(pred, reference):
    chrf = evaluate.load('chrf')
    return chrf.compute(predictions=pred, references=reference)

def Bert_score(pred, reference):
    bert = evaluate.load('bertscore')
    return bert.compute(predictions=pred, references=reference, lang='ar')

def METEOR_score(pred, reference):
    METEOR = evaluate.load('meteor')
    return METEOR.compute(predictions=pred, references=reference)

def TER_score(pred, reference):
    TER = evaluate.load('ter')
    return TER.compute(predictions=pred, references=reference)

def main(argv):
    results = {'model': args.model_name_or_path,
               'dataset': args.dataset,
                 }
    
    ds = load_dataset(args.dataset,args.subset,split=args.split,cache_dir=args.cache_dir)

    ds = ds.map(translate,batched=True,batch_size=args.b_size)    

    #ds = load_dataset('json',
    #data_files=[f'{args.save_path}/translation.json'],
    #cache_dir=args.cache_dir)

    #if args.save_translation:
    #    ds.to_json(f'{args.save_path}/translation.json',
    #                   force_ascii=False,
    #                   orient='records',
    #                   lines=True,)

    if 'bleu' in args.evaluate_metric:
        result = sacrebleu_score(pred=ds['pred'], reference=ds[args.tgt])
        results['bleu'] = result

    if 'chrf' in args.evaluate_metric:
        result = chrf_score(pred=ds['pred'], reference=ds[args.tgt])
        results['chrf'] = result

    if 'bertscore' in args.evaluate_metric:
        result = Bert_score(pred=ds['pred'], reference=ds[args.tgt])
        f1 = result['f1']
        result['f1'] = sum(f1)/len(f1)

        precision = result['precision']
        result['precision'] = sum(precision)/len(precision)

        recall = result['recall']
        result['recall'] = sum(recall)/len(recall)


        results['bertscore'] = result
        


    with open('translation_scores.json', 'w') as f:
        json.dump(results, f)


#if __name__ == '__main__':
args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,cache_dir=args.cache_dir)
print("Tokenizer Build Successfully")
#model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,cache_dir=args.cache_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,cache_dir=args.cache_dir,torch_dtype=torch.float16) #device_map="auto")

print("Model Build Successfully")
print(model)
model.to(args.device)

main(args)

#print(f"model is loaded on device {model.module.device}")

