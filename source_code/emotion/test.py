import torch
from dataloader import EmotionDataLoader
from model import EmotionDetectionModel
from pytorch_lightning import Trainer

if __name__ == "__main__":
	ckp_file = 'checkpoint/epoch=24-val_acc=0.8411-val_age_acc=0.8013-val_gender_acc=0.8857.ckpt'
	checkpoint = torch.load(ckp_file, map_location=torch.device('cpu'))
	model = EmotionDetectionModel()
	model.load_state_dict(checkpoint['state_dict'])

	data = EmotionDataLoader(batch_size=64, workers=4)

	trainer = Trainer(
		# accelerator="cpu",
		accelerator="mps",
	)

	trainer.test(model, data)