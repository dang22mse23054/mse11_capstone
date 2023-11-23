'''
	(Tham khảo source code của thầy Cường - https://github.com/ntcuong2103/traffic_sign_detection_pytorch/tree/master)
	Ta có thể dùng Faster-RCNN với nhiều loại backbone khác nhau.
    Nhưng default torchvision dùng ResNet để làm backbone.
    Ta có thể tùy chỉnh bằng ResNet-50, ResNet-101, or ResNet-152,...
    Ngoài ra còn có FPN (Feature Pyramid Network) dùng để phát hiện hình ảnh trong nhiều tỉ lệ kích thước khác nhau
	=> ở đây ta sẽ dùng resnet50fpn làm backbone thông qua utils resnet_fpn_backbone
'''

# Cái này là model architecture
from torchvision.models.detection import FasterRCNN
# cái này để lấy network có sẵn (pretrained model) làm backbone cho Faster-RCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# cái này để gen ra anchor points
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models import ResNeXt50_32X4D_Weights

'''
	num_classes (int): number of output classes of the model (>>> including the Background <<<).
		If box_predictor is specified, num_classes should be None.
		Tức là nếu chỉ nhận biết đâu là Face thì sẽ setting bằng 2 classes (Face, Background)
'''
class FasterRCNNResNet50FPN(FasterRCNN):
	def __init__(self, num_classes, **kwargs):
		# ResNet có 50 lớp và dùng ResNeXt (ResNet mở rộng) với cardinality là 32x4D (tham khảo thêm, chứ chưa hiểu lắm)
		backbone = resnet_fpn_backbone('resnext50_32x4d', weights=ResNeXt50_32X4D_Weights.DEFAULT, trainable_layers=5)
		
		# Setting anchor points size + ratios
		anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
		# kết quả ratios sẽ là ((1.0), (1.0), (1.0), (1.0), (1.0))
		aspect_ratios = ((1.0),) * len(anchor_sizes)
		anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
		
		super().__init__(backbone, num_classes, 
                         rpn_anchor_generator=anchor_generator, 
                         rpn_pre_nms_top_n_train=4000, rpn_pre_nms_top_n_test=2000,
                         rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                         **kwargs)

