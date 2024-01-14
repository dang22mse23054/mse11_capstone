from dataloader import FaceDataLoader
from model import FaceDetectionModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


if __name__ == "__main__":
	model = FaceDetectionModel(
		lr = 1e-4,
		momentum = 0.9,
		weight_decay = 1e-4,        
	)

	data = FaceDataLoader(batch_size=48, workers=4, img_size = 160)

	trainer = Trainer(
		accelerator="cpu",
		# accelerator="mps",
		# checkpoint_callback=True,
		callbacks = [
		    LearningRateMonitor(logging_interval='step'),
		    # ModelCheckpoint(dirpath='', filename='{epoch}-{val_acc:.4f}', save_top_k=5, monitor='val_acc', mode='max'),
			ModelCheckpoint(filename='{epoch}-{val_map:.4f}-{val_age_acc:.4f}-{val_gender_acc:.4f}', save_top_k=5, monitor='val_map', mode='max'),
		], 
		# check_val_every_n_epoch=1,
		# fast_dev_run=True,
		default_root_dir='checkpoint',

		# deterministic=False, 
		max_epochs=10, 
		# log_every_n_steps=2,

		# gpus = [2],
		# amp_backend='apex', 
		# amp_level='O1', 
		# # precision=16,
		# # strategy='ddp',        
	)

	trainer.fit(model, data)
		