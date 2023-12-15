
# How to prepareation

## Watch GPU process (RAM, %)

```sh
watch -n0.1 nvidia-smi
```

## Codes

```sh
data = FaceDataLoader(batch_size=16, workers=2, img_size = 160)

# check whether cuda is enabled
print(torch.cuda.is_available())
accelerator="gpu"
```

## Install Nvidia on Ubuntu (for workstation - Via Command Line)

Ref: https://www.linuxbabe.com/ubuntu/install-nvidia-driver-ubuntu

