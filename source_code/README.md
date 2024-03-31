
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

## WIDER_FACE Dataset

Nothing

## UTKFace Dataset

The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

## S3 Local

```sh
# create s3 bucket
aws --endpoint-url http://localhost:44572 s3api create-bucket --bucket cap-bucket --acl public-read-write --create-bucket-configuration LocationConstraint=ap-northeast-1
```
