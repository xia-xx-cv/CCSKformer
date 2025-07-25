import torch
import torch.nn as nn
from functools import partial
# from models.model_mambaV3_two import MVCINN
from models.CCSKformer import CCSKformer


def my_CCSKFormer(pretrained=False,depth=12,num_heads=9,view=1, **kwargs):
    pre_Path = kwargs['pre_Path']
    kwargs.pop('pre_Path',None)
    if view==1:
        model = CCSKformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=depth,
                    num_heads=num_heads, mlp_ratio=4, qkv_bias=True, **kwargs)
    else:
        model = CCSKformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=depth,
                       num_heads=num_heads, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        print('pred_path:', pre_Path)
        checkpoint = torch.load(pre_Path)
        state_dict = model.state_dict()
        if isinstance(checkpoint,dict):
            if 'model' in checkpoint.keys():
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            
            load_dict = {k:v for k,v in checkpoint_model.items() if k in state_dict.keys()}
            if kwargs['num_classes']!=1000 and pre_Path=='weights/Conformer_base_patch16_rename_3.pth':
                print("load_dict pop head weight and bias")
                load_dict.pop('csk_cls_head.weight')
                load_dict.pop('csk_cls_head.bias')
                load_dict.pop('conv_cls_head.weight')
                load_dict.pop('conv_cls_head.bias')
            # print(load_dict.keys())
            state_dict.update(load_dict)
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(str(e))
            print("model load state dick!")
        else:
            model = checkpoint
    return model
def init_conformer(model,num_classes=5):
    model.csk_cls_head = nn.Linear(model.csk_cls_head.in_features, num_classes) if num_classes > 0 else nn.Identity()
    model.conv_cls_head = nn.Linear(model.conv_cls_head.in_features, num_classes) if num_classes > 0 else nn.Identity()
    return model