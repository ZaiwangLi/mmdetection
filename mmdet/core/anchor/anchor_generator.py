import torch


# key words: pairing, meshgrid, broadcast rule
# broadcast rule: x shape (10, 10), y shape(1, 10). if you do x + y or x * y, 
#                 y will automatically expand its row to be of shape (10, 10),
#                 another exmaple b: shape(1, 10) + shape(10, 1) = shape(10, 10)
#                 choose a collegue to give an example.
#
# meshgrid:       pair all the elements between 2 tensors.
#                 for example: x: (0, 1) y: (2, 3), combinations:(0,2),(1,2)(0,3)(1,3)
#                 it returns xx=(0,1,0,1) yy=(2,2,3,3). xx duplicates itself when each of the original sequence ends
#                 yy duplicates the same elements until the number reaches
# example b is basically a sepcific example of meshgrid.
class AnchorGenerator(object):
    """
    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)
    
    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        
        # base anchors only requires sizes, the position are defined by offsets
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            # if there are n1 w_ratios and n2 scales, ws have length of n1 * n2
            # kind of meshgrid implementation by using broadcast rules
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable
        
        return base_anchors
    
    # meishgrid is implemented in tensorflow but in pytorch you have to do it yourself.
    # all the combinitions of x and y, for example: x: (0, 1) y: (2, 3), combinations:(0,2),(1,2)(0,3)(1,3)
    # xx = (0,1,0,1) yy=(2, 2, 3, 3)^T
    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        
        # not all anchors can fit one perticular feature map, for example:
        # the feature map is downsampled or cropped.
        # so we have to have flags to remove some of them
        #   111100   original anchors spreads over 6*6 
        #   111100   now the anchors are 4*4 by the flag mask
        #   111100
        #   111100
        #   000000
        #   000000
        #   ...
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:,
                      None].expand(valid.size(0),
                                   self.num_base_anchors).contiguous().view(-1)
        return valid
