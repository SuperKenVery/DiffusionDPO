from typing import Dict, List, Any

def iter_dict_batch(dict: Dict[Any, List[Any]]) -> List[Dict[Any, Any]]:
    keys = list(dict.keys())
    length = len(dict[keys[0]])
    result = []
    for i in range(length):
        result.append({k: dict[k][i] for k in keys})

    return result

if __name__=="__main__":
    a={
        'xxx': [1,2,3],
        'yyy': [3,4,5]
    }
    b = iter_dict_batch(a)
    assert b==[
        {'xxx': 1, 'yyy': 3},
        {'xxx': 2, 'yyy': 4},
        {'xxx': 3, 'yyy': 5},
    ]
