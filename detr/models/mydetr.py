from detectron2.modeling import META_ARCH_REGISTRY
from torch import nn

from .detr import (
    MLP as MLP,
    build_matcher,
    SetCriterion
)
from models.backbone import (
    Backbone,
    Joiner
)
from models.position_encoding import (
    PositionEmbeddingSine,
    PositionEmbeddingLearned
)
from models.transformer import Transformer
from util.misc import NestedTensor


@META_ARCH_REGISTRY.register()
class my_DETR(nn.Module):

    # def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
    def __init__(self, args):
        super(my_DETR, self).__init__()

        N_steps = args.hidden_dim // 2
        if args.position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        elif args.position_embedding in ('v3', 'learned'):
            position_embedding = PositionEmbeddingLearned(N_steps)
        else:
            raise ValueError(f"not supported {args.position_embedding}")
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        joiner = Joiner(backbone, position_embedding)
        joiner.num_channels = backbone.num_channels
        transformer = Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
        num_classes = 20 if args.dataset_file != 'coco' else 91
        if args.dataset_file == "coco_panoptic":
            num_classes = 250
        matcher = build_matcher(args)
        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        # if args.masks:
        #     weight_dict["loss_mask"] = args.mask_loss_coef
        #     weight_dict["loss_dice"] = args.dice_loss_coef
        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'boxes', 'cardinality']
        if args.masks:
            losses += ["masks"]
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=args.eos_coef, losses=losses)
        self.num_queries = args.num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(args.num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(joiner.num_channels, hidden_dim, kernel_size=1)
        self.backbone = joiner
        self.aux_loss = args.aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if not isinstance(samples, NestedTensor):
            # print(f'=================={len(samples)}=================')
            # print(samples[0].keys())
            samples = NestedTensor.from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out
