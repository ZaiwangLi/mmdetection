# feature map, anchors representation: list1[list2]
#   2D list element: tensor
#   len(list1) = image num, len(list2) = scale level num

# when to deal with these 2 dim?
#   1. assigning anchors: we dont actually use scale level dimension, omit it
#      how to assign only depends on image index,
#   2. After anchor assign, every bbox even from different images will have 
#      its own gtbox and gtcls. So batch dim will be omitted while scale level
#      is restored.
#   3. calculating the loss => we need scale level dim to balance losses from 
#      different scale level



import torch

from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply


def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels, 
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    
    # since level is not important for anchor assign and sampling, omit
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.
    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    Args:
        target: all the target(boxes or something else) across 
            different images in a batch
        num_level_anchors: list, [lvl1_num, lvl2_num, lvl3_num ...] 
    returns:
        level_targets: split all the targets by levels
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    """
    Args:
        flat_anchors (tensor): shape(-1, 4), only for one image, 
                               scale level info are lost.
        valid_flags (tensor): shape(-1, 1), only for one image,
                              prior knowledge on whether an anchor is useful
        gt_bboxes (tensor): shape(-1, 4), only for one image
                            the bboxes for anchor assign
        gt_bboxes_ignore (tensor): prior knowledge on whether a gtbbox is useful
        img_meta (dict): image info 
        target_means ():
        target_stds ():
        cfg:
        label_chennels: 
        sampling:
        unmap_outputs:
    return:
        ?
    """
    # both index and T are torch/numpy tensors
    #  index = [T F T F T T F F]
    #  T = [1,2,3,4,5,6,5,7,8]
    #  T[index] = [1,3,5,6]
    #
    #  index = [1, 2, 5, 7]
    #  T[index] = [2, 3, 6, 8]
    
    # remove images outside the image
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    
    if not inside_flags.any():
        return (None, ) * 6
    
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
        
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)
    
    
    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        # encoding:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            # binary classification
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    """remove anchors in image padding area by valid_flags
       remove anchors in the border area by allow_border
    Args:
        flat_anchors: shape [-1, 4], for anchors without feature map info
        valid_flags: prior knowledge of validation
        image_shape:
        allowed_border: when >=0 anchors must have limitations 
            regard to the image size
    returns:
        inside_flags: indicate wheter the anchor is in the image 
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags



def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    """
    Args:
        data: subset items
        count: original set item number
        inds: data's original indexes
        fill: fill in default vales for return tensor
    return:
        ret, shape of original data with corresponding pos filled by data
    """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
