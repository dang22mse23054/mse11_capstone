cd /opt/app && pm2 stop all
cd ~/caad-task-tracker/ && git pull
cp -avR ~/caad-task-tracker/app /opt/
cd /root/caad-task-tracker/env_encrypted/ && source /root/.bash_profile && cat .env.$ENV.encrypted | aws-encryption-cli --decrypt -i - --wrapping-keys provider=aws-kms key=arn:aws:kms:ap-northeast-1:101313435800:key/15db81fb-717f-46c5-a6c4-fcace7e9fbe0 profile=task-tracker  --commitment-policy require-encrypt-require-decrypt --output - --decode -S > /opt/.env
cd /opt/app && yarn install
cd /opt/app && yarn run build
cd /opt/app && yarn run db:migrate
cd /opt/app && pm2 start "yarn run start"