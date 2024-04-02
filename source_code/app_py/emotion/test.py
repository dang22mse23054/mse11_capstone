import torch
from dataloader import EmotionDataLoader
from model import EmotionDetectionModel
from pytorch_lightning import Trainer

if __name__ == "__main__":
	ckp_file = 'checkpoint/inception-epoch=9-val_acc=0.7520.ckpt'
	checkpoint = torch.load(ckp_file, map_location=torch.device('cpu'))
	model = EmotionDetectionModel()
	model.load_state_dict(checkpoint['state_dict'])

	data = EmotionDataLoader(batch_size=64, workers=4, img_size = 299)

	trainer = Trainer(
		accelerator="cpu",
		# accelerator="mps",
	)

	trainer.test(model, data)