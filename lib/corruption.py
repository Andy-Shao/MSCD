from dataclasses import dataclass

@dataclass
class CorruptionMeta:
    type:str
    level:str

def corruption_meta(corruption_types:list[str], corruption_levels:list[str]) -> list[CorruptionMeta]:
    ret = []
    for ctype in corruption_types:
        for l in corruption_levels:
            meta = CorruptionMeta(type=ctype, level=l)
            ret.append(meta)
    return ret