import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
from operator import mul
from functools import reduce
import clip
from adapter import adapter
from mlp import mlp
from lavis.models import load_model_and_preprocess
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
class MCDCLIP(nn.Module):
    def __init__(self,classname,prompt_num,prompt_dim,embedding_dim,re_com_dim,re_pri_dim,alpha):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha=alpha
        self.prompt_dim=prompt_dim
        self.prompt_num=prompt_num
        self.embedding_dim=embedding_dim
        self.re_com_dim=re_com_dim
        self.re_pri_dim=re_pri_dim
        self.model, self.preprocess = clip.load('ViT-B/32', device)
        self.prompt_embeddings_front = nn.Parameter(torch.randn(
            1, self.prompt_num, 768).to(device))
        self.num_class=len(classname)
        self.x_class=classname
        self.re_common=nn.Parameter(torch.randn(self.num_class, self.re_com_dim).to(device))
        self.Common_Adapter=adapter(self.embedding_dim)
        self.Private_Adapter_front=adapter(self.embedding_dim)
        self.Private_Adapter_later=adapter(self.embedding_dim)
        self.RepreNet = mlp(self.embedding_dim,self.re_pri_dim)
        self.Prompt_Aligner=mlp(768,768)
        self.mlp=mlp(self.embedding_dim,self.embedding_dim+self.re_com_dim+self.re_pri_dim)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, mask=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tem = self.model.logit_scale.exp().to(device)
        #Prompt Alignment
        view_1=x[:, 0, :, :, :].to(device)
        view_2=x[:, 1, :, :, :].to(device)
        view_1_cls_feature=self.model.encode_image(view_1,1,self.prompt_embeddings_front)
        view_2_cls_feature=self.model.encode_image(view_2,1,self.Prompt_Aligner(self.prompt_embeddings_front))

        #feature disentangling
        view_1_com=self.Common_Adapter(view_1_cls_feature)
        view_2_com=self.Common_Adapter(view_2_cls_feature)
        view_1_com=self.mlp(view_1_com)
        view_2_com=self.mlp(view_2_com)
        view_1_pri=self.Private_Adapter_front(view_1_cls_feature)
        view_2_pri=self.Private_Adapter_later(view_2_cls_feature)

        #Add Additional Textual Representations
        text = [f"a photo of {cls}." for cls in self.x_class]
        text_emb_front=clip.tokenize(text).to(device)
        text_emb_later=clip.tokenize(text).to(device)
        texts_front=self.model.encode_text(text_emb_front)
        texts_later=self.model.encode_text(text_emb_later)
        texts_front=torch.concat((texts_front,self.re_common),dim=1)
        texts_later=torch.concat((texts_later,self.re_common),dim=1)
        view_1_re_private=self.RepreNet(view_1_pri)
        view_2_re_private=self.RepreNet(view_2_pri)
        view_1_re_private=view_1_re_private.unsqueeze(1).expand(-1, self.num_class, -1)
        view_2_re_private=view_2_re_private.unsqueeze(1).expand(-1, self.num_class, -1)
        texts_front=texts_front.unsqueeze(0).repeat((view_1_cls_feature.size(0)), 1, 1)
        texts_later=texts_later.unsqueeze(0).repeat((view_2_cls_feature.size(0)), 1, 1)
        texts_front=torch.concat((texts_front,view_1_re_private),dim=2)
        texts_later=torch.concat((texts_later,view_2_re_private),dim=2)

        #decision(late fusion)
        view_1_cls = view_1_com/view_1_com.norm(dim=-1, keepdim=True)
        view_2_cls = view_2_com/view_2_com.norm(dim=-1, keepdim=True)
        text_front = texts_front/texts_front.norm(dim=-1, keepdim=True)
        text_later = texts_later/texts_later.norm(dim=-1, keepdim=True)
        similarity_front = (tem * torch.bmm(view_1_cls.unsqueeze(1), text_front.transpose(1, 2)).squeeze(1))
        similarity_later = (tem * torch.bmm(view_2_cls.unsqueeze(1), text_later.transpose(1, 2)).squeeze(1))
        kekka=self.alpha*similarity_front+(1-self.alpha)*similarity_later
        return kekka
