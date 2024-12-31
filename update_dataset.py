from datasets import load_from_disk
from transformers import AutoTokenizer



def main():
    ds = load_from_disk('./data/tokenized_dataset_130g_2k_v4')
    print(ds)

    tokenizer = AutoTokenizer.from_pretrained('./tokenizer/tokenizer_v7')

    eos_token_id = tokenizer.eos_token_id
    new_line = tokenizer.encode('\n')
    ds['train']['input_ids'] = ds['train']['input_ids'].map(lambda x: new_line if x == eos_token_id else x)
    ds['test']['input_ids'] = ds['test ']['input_ids'].map(lambda x: new_line if x == eos_token_id else x)

    ds.save_to_disk('./data/tokenized_dataset_130g_2k_v4_R1')


if __name__ == "__main__":
    main()