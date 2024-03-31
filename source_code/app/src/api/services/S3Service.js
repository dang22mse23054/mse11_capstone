const AWS = require('aws-sdk');
const fs = require('fs');
const path = require('path');
const mime = require('mime-types');
const stream = require('stream');

// if (process.env.NEXT_PUBLIC_NODE_ENV == 'development') {
// AWS.config.credentials = new AWS.SharedIniFileCredentials({ profile: process.env.S3_PROFILE });
// }

class S3Service {
	constructor(profileName) {
		if (profileName) {
			this.credential = new AWS.SharedIniFileCredentials({ profile: profileName });

		} else if (process.env.NEXT_PUBLIC_NODE_ENV == 'development') {
			this.credential = new AWS.SharedIniFileCredentials({ profile: process.env.S3_PROFILE });
		}

		this.profileObj = { credentials: this.credential || undefined }

		if (process.env.NEXT_PUBLIC_NODE_ENV == 'development' && process.env.S3_PROFILE == 'localstack') {
			this.profileObj = {
				credentials: this.credential || undefined,
				endpoint: 'http://s3.localhost.localstack.cloud:44572',
				region: 'ap-northeast-1'
			}
		}
	}

	async createFolder(bucketName, prefix) {
		let s3 = new AWS.S3(this.profileObj);

		return new Promise(function (resolve, reject) {
			s3.putObject({
				Bucket: bucketName,
				Key: `${prefix}/`,
			}).promise().then(resolve, reject);
		});
	}

	async removeFolder(bucketName, prefix) {
		let s3 = new AWS.S3(this.profileObj);

		// get all items by prefix
		const objects = await s3.listObjects({
			Bucket: bucketName,
			Prefix: prefix
		}).promise();

		return Promise.all(
			objects.Contents.map(t =>
				s3.deleteObject({
					Bucket: bucketName,
					Key: t.Key
				}).promise()
			)
		);
	}

	saveFile(filePath, bucketName, keyPrefix, fileName = path.basename(filePath)) {
		let s3 = new AWS.S3(this.profileObj);

		// ex: /path/to/my-picture.png becomes my-picture.png
		var fileStream = fs.createReadStream(filePath);

		// If you want to save to "my-bucket/{prefix}/{filename}"
		//                    ex: "my-bucket/my-pictures-folder/my-picture.png"
		var keyName = path.join(keyPrefix, fileName);

		// We wrap this in a promise so that we can handle a fileStream error
		// since it can happen *before* s3 actually reads the first 'data' event
		return new Promise(function (resolve, reject) {
			fileStream.once('error', reject);
			s3.upload({
				Bucket: bucketName,
				Key: keyName,
				Body: fileStream,
				ContentType: mime.lookup(fileName)
			}).promise().then(resolve, reject);
		});
	}

	async copyFile(srcUri, tarUri) {
		let parts = tarUri.split('/');
		if (parts.length > 1) {
			let s3 = new AWS.S3(this.profileObj);
			let tarBucket = parts[0];
			let tarFile = tarUri.slice(tarUri.indexOf(parts[1]));

			return new Promise(function (resolve, reject) {
				s3.copyObject({
					CopySource: srcUri,
					Bucket: tarBucket,
					Key: tarFile,
				}).promise().then(resolve, reject);
			});
		}
	}

	async copyFiles(list = [/* {srcUri: 'srcBucket/(prefix/)srcFile', tarUri: 'tarBucket/(prefix/)tarFile'} */]) {
		return Promise.all(
			list.map(item => this.copyFile(item.srcUri, item.tarUri))
		);
	}

	async moveFile(srcUri, tarUri) {
		return this.copyFile(srcUri, tarUri)
			.then(() => this.removeFile(srcUri))
			.catch((err) => console.error(err));
	}

	async moveFiles(list) {
		return Promise.all(
			list.map(item => this.copyFile(item.srcUri, item.tarUri))
		).then(() => {
			this.removeFiles(list);
		});
	}

	async removeFile(delUri) {
		if (delUri) {
			let parts = delUri.split('/');
			if (parts.length > 1) {
				let s3 = new AWS.S3(this.profileObj);
				let bucket = parts[0];
				let delFile = delUri.slice(bucket.length + 1);

				return new Promise(function (resolve, reject) {
					s3.deleteObject({
						Bucket: bucket,
						Key: delFile,
					}).promise().then(resolve, reject);
				});
			}
		}
	}

	async removeFiles(list = [/* 'srcBucket/(prefix/)srcFile' */]) {
		return Promise.all(
			list.map(uri => this.removeFile(uri))
		);
	}

	async getPolicy(bucketName) {
		let s3 = new AWS.S3(this.profileObj);
		return new Promise(function (resolve, reject) {
			s3.getBucketPolicy({
				Bucket: bucketName
			}).promise().then(resolve, reject);
		});
	}

	async changePolicy(bucketName, policy) {
		let s3 = new AWS.S3(this.profileObj);
		return new Promise(function (resolve, reject) {
			s3.putBucketPolicy({
				Bucket: bucketName,
				Policy: policy
			}).promise().then(resolve, reject);
		});
	}

	getMetaData(params /* { bucketName, objectKey } */) {
		let s3 = new AWS.S3(this.profileObj);

		return new Promise((resolve, reject) => {
			// get file info
			s3.headObject(params, (err, metadata) => {
				if (err && ['NotFound', 'Forbidden'].indexOf(err.code) > -1) {
					return reject(err);
				} else if (err) {
					const e = Object.assign({}, 'Unknown Error', { err });
					return reject(e);
				}
				return resolve(metadata);
			});
		});
	}

	async downStreamFile(bucketName, objectKey, res) {
		let s3 = new AWS.S3(this.profileObj);
		const params = {
			Bucket: bucketName,
			Key: objectKey
		};
		return new Promise((resolve, reject) => {
			this.getMetaData(params)
				.then((metaData) => {
					const stream = s3.getObject(params).createReadStream();
					resolve({ stream, metaData });
				})
				.catch((err) => {
					reject(err);
				});
		});
	}

	upStreamFile(bucketName, objectKey) {
		let s3 = new AWS.S3(this.profileObj);
		// create Duplex stream to read & write
		var upStream = new stream.PassThrough();

		const params = {
			Bucket: bucketName,
			Key: objectKey,
			Body: upStream
		};

		return {
			writeStream: upStream,
			promise: s3.upload(params).promise()
		};
	}
}

module.exports = S3Service;