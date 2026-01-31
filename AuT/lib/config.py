from lib.utils import ConfigDict

def AuT_base(class_num:int, n_mels:int) -> ConfigDict:
    cfg = ConfigDict()

    cfg.transform = ConfigDict()
    cfg.transform.layer_num = 12
    cfg.transform.head_num = 12
    cfg.transform.atten_drop_rate = .0
    cfg.transform.mlp_mid = 3072
    cfg.transform.mlp_dp_rt = .0

    cfg.classifier = ConfigDict()
    cfg.classifier.class_num = class_num
    cfg.classifier.extend_size = 2048
    cfg.classifier.convergent_size = 256
    cfg.classifier.in_embed_num = 2

    cfg.embedding = ConfigDict()
    cfg.embedding.channel_num = n_mels
    cfg.embedding.marsked_rate = .15
    cfg.embedding.embed_size = 768
    cfg.embedding.in_shape = [80, 104]
    cfg.embedding.num_layers = [6, 8]

    return cfg