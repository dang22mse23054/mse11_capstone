const { videoService } = require('apiDir/services');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const BaseResponse = require('apiDir/dto/BaseResponse');
const ErrorObject = require('apiDir/dto/ErrorObject');
const formidable = require('formidable');
const fs = require('fs');
const FormData = require('form-data')
const moment = require('moment-timezone');
const { RoutePaths, Common, ErrorCodes, VideoStatus } = require('commonDir/constants');
const { JPEG, PNG, MP4 } = Common.FileTypes;
const ACCETED_TYPES = [JPEG, PNG, MP4];
const axios = require('axios');

module.exports = class AdsAdviceController {
	getAll = async (req, res) => {

		let respObj = new BaseResponse();
		let respStatus = 200;

		try {
			// get all enabled videos
			const videos = await videoService.getVideo();
			
			respObj.setData(videos?.map(i => i.refFileName));
			return res.status(respStatus).json(respObj);
		} catch (err) {
			log.error(err);
			respStatus = 500;
			respObj.setError(err.code, err.message);
			return res.status(respStatus).json(respObj);
		}
	}
	
	advice = async (req, res) => {

		let respObj = new BaseResponse();
		let respStatus = 200;

		let formData = new formidable.IncomingForm();

		let file = null;
		return new Promise((resolve) => {
			formData.parse(req, (err, fields, files) => resolve({ err, fields, files }));

		}).then(async ({ err, fields, files }) => {
			console.log('========== Parsing Request ==========');
			console.log(files);

			file = files.uploadedFile;

			if (!file) {
				throw new ErrorObject(ErrorCodes.FILE.NO_FILE);
			}
			
			if (err) {
				// Check for and handle any errors here.
				throw err;
			}

			// check file type
			const contentType = file.type;
			console.log('Content type: ', contentType)
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

			// Prepare form data
			const formData = new FormData();
			console.log(file.name);

			// khác với client, trên server phải tự tạo đọc stream từ file rồi append vào form-data
			formData.append('img_file', fs.createReadStream(file.path), {
				filename: file.name
			});

			return formData

		}).then((formData) => {
			console.log('========== Request to ML Server ==========');
			return axios.post('http://localhost:5001/predict', formData, {
				headers: {
					// phải tự định nghĩa boundary cho form-data thì data mới gửi đúng request dạng upload files lên server
					'Content-Type': `multipart/form-data; boundary=${formData._boundary}`
				}
			})
		}).then((axiosResp) => {
			return axiosResp.data;

		}).then(async (finalData) => {
			console.log('========== Biến tấu ==========');
			const data = await videoService.getVideo({
				age: finalData.age_groups,
				gender: finalData.genders
			});
			
			respObj.setData({
				...finalData,
				videos: data?.map(i => i.refFileName)
			});

			return res.status(respStatus).json(respObj);

		}).catch((err) => {
			if (err instanceof ErrorObject) {
				err = err.message
			}

			log.error(err);
			respStatus = 500;
			respObj.setError(err.code, err.message);
			return res.status(respStatus).json(respObj);

		}).finally(() => {
			// remove uploaded tmp files on server
			console.log('Finally block');
			if (file) {
				console.log('Removing tmp file');
				fs.unlink(file.path, (err => {
					if (err) {
						log.error(err);
						console.log('Error while removing tmp file');
						throw err;

					}
				}));
			}
		});
	}
}
