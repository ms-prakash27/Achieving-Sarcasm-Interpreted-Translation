import pandas as pd

def preprocess(text, prefix=''):
    return prefix + ' '.join(text.strip().split())

def separate_inputs_targets(data):
    data_dict = {}
    for _, row in data.iterrows():
        input_, target_ = row
        if input_ not in data_dict:
            data_dict[input_] = []
        data_dict[input_].append(target_)
    return zip(*data_dict.items())

def drop_duplicates(data, axis='rc'):
    '''
    Drop duplicates from the mentioned axis mentioned
    '''
    if 'c' in axis.lower():
        data.drop_duplicates('interpretation', inplace=True)
    
    if 'r' in axis.lower():
        data = data[data['interpretation'] != data['sarcasm']]
    
    return data

def custom_tokenize(text, tokenizer):
    return ' '.join(tokenizer.tokenize(text))

def save_tests(inputs, preds, model, save_path='/blue/cai6307/n.kolla/data/saved_tests/'):
    test_df = pd.DataFrame({
        'inputs': inputs,
        'preds': preds,
    })
    
    test_df.to_csv(f'{save_path+model}.csv')
    
def load_tests(inputs, targets, input_model, save_path='/blue/cai6307/n.kolla/data/saved_tests/'):
    cur_df = pd.DataFrame({
        'inputs': inputs,
        'targets': targets,
    })
    
    input_df = pd.read_csv(f'{save_path+input_model}.csv')
    
    cur_df = cur_df.merge(input_df, how='inner')
    return cur_df.preds.to_numpy(), cur_df.targets.to_numpy()
