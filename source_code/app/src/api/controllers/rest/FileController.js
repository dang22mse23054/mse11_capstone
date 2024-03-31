const { s3Service, videoService } = require('apiDir/services');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const BaseResponse = require('apiDir/dto/BaseResponse');
const formidable = require('formidable');
const fs = require('fs');
const moment = require('moment-timezone');
const { RoutePaths, Common, ErrorCodes, VideoStatus } = require('commonDir/constants');
const { _7Z, ZIP, MS_OFFICE_FILE, XLS, XLSX, XLSM, DOC, DOCX, JPEG, PNG, CSV, PPT, PPTX, MP4 } = Common.FileTypes;
const ACCETED_TYPES = [_7Z, ZIP, MS_OFFICE_FILE, XLS, XLSX, XLSM, DOC, DOCX, PPT, PPTX, JPEG, PNG, CSV, MP4];

module.exports = class FileController {


	upload = async (req, res) => {
		let respObj = new BaseResponse();
		let respStatus = 200;

		let formData = new formidable.IncomingForm();

		let file = null;
		new Promise((resolve) => {
			formData.parse(req, (err, fields, files) => resolve({ err, fields, files }));

		}).then(async ({ err, fields, files }) => {
			if (files) {
				file = files.uploadedFile;
			}

			if (err) {
				// Check for and handle any errors here.
				throw err;
			}

			if (file) {
				// log.debug(file);

				let now = moment().tz('UTC').format('YYYYMMDD_HHmm');
				let folderName = process.env.S3_TMP_FOLDER;
				let fileName = file.name.replaceAll(':', '_');

				// check file type
				const contentType = file.type;
				let contentTypeInfo = null;
				for (let i = 0; i < ACCETED_TYPES.length; i++) {
					if (ACCETED_TYPES[i].val.includes(contentType)) {
						contentTypeInfo = { ...ACCETED_TYPES[i] };
						break;
					}
				}
				if (!contentTypeInfo) {
					throw (ErrorCodes.FILE.UNSUPPORTED);
				}
				// set file extension
				let extension = contentTypeInfo.ext;

				let newName = `${now}_${Math.random().toString(36).substring(3)}.${extension}`;
				const filePath = `${folderName}/${newName}`;

				// save file
				await s3Service.saveFile(file.path, process.env.S3_BUCKET, folderName, newName)
					.then(async (result) => {
						// log.debug(JSON.stringify(result));

						respObj.setData({
							fileName,
							filePath,
							url: `https://${process.env.NEXT_PUBLIC_SERVER_DOMAIN}/${RoutePaths.PREFIX.S3}/${filePath}`,
							updatedAt: result.updatedAt
						});

						return res.status(respStatus).json(respObj);
					})
					.catch((err) => {
						log.error(err);
						throw err;
					});
			}
		}).catch((err) => {
			log.error(err);
			respStatus = 500;
			respObj.setError(err.code, err.message);
			return res.status(respStatus).json(respObj);

		}).finally(() => {
			// remove uploaded tmp files on server
			if (file) {
				fs.unlink(file.path, (err => {
					if (err) {
						log.error(err);
						throw err;

					}
				}));
			}
		});
	}

	remove = async (req, res) => {
		let respObj = new BaseResponse();
		let respStatus = 200;
		const { target, videoId } = req.body;

		let result = null;
		try {
			switch (target) {
				case Common.FileTargets.Video:
					result = await videoService.removeFile(videoId);
					break;
			}

			if (result) {
				respObj.setData(true);
			} else {
				respObj.setError(404, 'Not found');
			}

			return res.status(respStatus).json(respObj);
		} catch (err) {
			log.error(err);
			respStatus = 500;
			respObj.setError(err.code, err.message);
			return res.status(respStatus).json(respObj);
		}
	}

	streamFile = async (req, res) => {
		let objKey = req.params.objKey;
		const parts = objKey.split('/');
		/*
			Temporary: 	[S3_TMP_FOLDER]/filename
			Video: 	[S3_VIDEO_FOLDER]/<s_id>/filename
		*/
		const folderName = parts[0];
		let fileName = null;
		switch (folderName) {
			case process.env.S3_TMP_FOLDER:
				fileName = parts[1];
				break;

			case process.env.S3_VIDEO_FOLDER:
				const vid = parts[1];
				const video = await videoService.getVideo(vid);

				if (video) {
					fileName = video.refFileName;
				}
				break;

			default:
				return res.redirect('/static/img/no-image.png');
		}

		s3Service.downStreamFile(process.env.S3_BUCKET, objKey)
			.then(({ stream, metaData }) => {
				res.set('Content-Disposition', `attachment; filename="${encodeURI(fileName)}"`);
				res.set('Content-Type', metaData.ContentType);
				res.set('Content-Length', metaData.ContentLength);
				res.set('Last-Modified', metaData.LastModified);
				res.set('ETag', metaData.ETag);

				//Pipe the s3 object to the response
				stream.pipe(res);
			}).catch((err) => {
				log.error(err);
				res.redirect('/static/img/no-image.png');
			});
	}

	autoclear = async (req, res) => {
		res.statusCode = 200;
		let isErr = false;
		let prefix = moment().tz('UTC').startOf('hour').subtract(3, 'hours').format('YYYYMMDD_HH');

		s3Service.removeFolder(process.env.S3_BUCKET, `${process.env.S3_TMP_FOLDER}/${prefix}`)
			.then((result) => {
				log.info('Autoclear success');
				res.statusCode = 200;
			})
			.catch((err) => {
				res.statusCode = 500;
				isErr = true;
				log.error('Autoclear error');
				log.error(err);
			}).finally(() => {
				res.type('text');
				res.end(`Auto clear [${process.env.S3_TMP_FOLDER}] folder: ${isErr ? 'Fail' : 'Success'}`);
			});
	}

};