const { s3Service, taskService, scheduleService, fileService, csvService } = require('apiDir/services');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const BaseResponse = require('apiDir/dto/BaseResponse');
const formidable = require('formidable');
const fs = require('fs');
const moment = require('moment-timezone');
const { RoutePaths, Common, ErrorCodes, VideoStatus } = require('commonDir/constants');
const { _7Z, ZIP, MS_OFFICE_FILE, XLS, XLSX, XLSM, DOC, DOCX, JPEG, PNG, CSV, PPT, PPTX } = Common.FileTypes;
const ACCETED_TYPES = [_7Z, ZIP, MS_OFFICE_FILE, XLS, XLSX, XLSM, DOC, DOCX, PPT, PPTX, JPEG, PNG, CSV];

module.exports = class FileController {


	upload = async (req, res) => {
		let respObj = new BaseResponse();
		let respStatus = 200;

		let formData = new formidable.IncomingForm();

		let file = null;
		new Promise((resolve) => {
			formData.parse(req, (err, fields, files) => resolve({ err, fields, files }));

		}).then(async ({ err, fields, files }) => {
			const { target = Common.FileTargets.Temporary, taskId, taskProcId, updatedAt } = fields;
			// log.debug(fields);
			// log.debug(files);

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

				// If this file is uploaded for Task
				if (taskId && target == Common.FileTargets.Task && [_7Z.ext, ZIP.ext, XLS.ext, XLSX.ext].includes(extension)) {
					// => check whether the file has been set password if ZIP or Excel file
					if (!await fileService.hasProtectedByPassword(file.path, extension)) {
						throw (ErrorCodes.FILE.NO_PASSWORD);
					}
				}

				// Check Obsolete updatedAt and file name 
				switch (target) {
					case Common.FileTargets.Task:
						// Check Obsolete updatedAt
						const taskObj = await taskService.getTask(taskId);
						if (!(taskObj && taskObj.updatedAt.getTime() == updatedAt)) {
							throw (ErrorCodes.OBSOLETE_DATA);
						}

						// check whether task's destination is CLIENT => check file name has Client name
						if (taskObj.destType == Common.DestTypes.CLIENT.id) {
							const isValidFilename = await fileService.hasClientCompNameInFileName(file.name, taskObj, { minClient: 1 });
							if (!isValidFilename) {
								throw (ErrorCodes.FILE.NAME_WITH_CLIENT_NAME);
							}
						}

						folderName = `${process.env.S3_TASK_FOLDER}/${taskId}`;
						break;

				}

				let newName = `${now}_${Math.random().toString(36).substring(3)}.${extension}`;
				const filePath = `${folderName}/${newName}`;

				// save file
				await s3Service.saveFile(file.path, process.env.S3_BUCKET, folderName, newName)
					.then(async (result) => {
						// log.debug(JSON.stringify(result));

						// Save uploaded file to DB if Task or TaskProcess
						switch (target) {
							case Common.FileTargets.Task:
								// save to Task table
								result = await taskService.saveFile({ taskId, fileName, filePath });
								break;

							case Common.FileTargets.TaskProcess:
								// save to Task table
								result = await taskService.saveFile({ taskId, taskProcId, fileName, filePath });
								break;
						}

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
		const { target, taskId, taskProcId, scheduleId, updatedAt } = req.body;

		let result = null;
		try {
			// Save uploaded file to DB if Task or TaskProcess
			switch (target) {
				case Common.FileTargets.Task:
					// Check Obsolete updatedAt
					const taskObj = await taskService.getTask(taskId);
					if (!(taskObj && taskObj.updatedAt.getTime() == updatedAt)) {
						throw (ErrorCodes.OBSOLETE_DATA);
					}

					result = await taskService.removeFile({ taskId });
					break;

				case Common.FileTargets.TaskProcess:
					const taskProc = await taskService.getProcesses(taskId, taskProcId);
					if (!(taskProc && taskProc.updatedAt.getTime() == updatedAt)) {
						throw (ErrorCodes.OBSOLETE_DATA);
					}

					result = await taskService.removeFile({ taskId, taskProcId });
					break;

				case Common.FileTargets.Schedule:
					result = await scheduleService.removeFile(scheduleId);
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
			Schedule: 	[S3_SCHEDULE_FOLDER]/<s_id>/filename
			Task: 		[S3_TASK_FOLDER]/<t_id>/filename
		*/
		const folderName = parts[0];
		let fileName = null;
		switch (folderName) {
			case process.env.S3_TMP_FOLDER:
				fileName = parts[1];
				break;

			case process.env.S3_TASK_FOLDER:
				const taskId = parts[1];

				if (parts.length > 3) {
					// Download TaskProc file
					const procId = parts[3];
					const taskProc = await taskService.getProcesses(taskId, procId);

					if (taskProc) {
						fileName = taskProc.fileName;
					}
				} else {
					// Download Task file 
					// get task info 
					const task = await taskService.getTask(taskId);

					if (task) {
						fileName = task.fileName;
						//Update donwload counter
						await taskService.updateDownCount({
							taskId: task.id,
							downCount: task.status == VideoStatus.STOPPED ? task.downCount + 1 : undefined
						});
					}
				}

				break;

			case process.env.S3_SCHEDULE_FOLDER:
				const sid = parts[1];
				const schedule = await scheduleService.getSchedule(sid);

				if (schedule) {
					fileName = schedule.refFileName;
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



	// copyImage(req, res) {
	// 	s3Service.copyFiles([{
	// 		srcUri: 'daka-test/tmp/demo.png',
	// 		tarUri: 'daka-test/img/demo.png',
	// 	}]).then((result) => {
	// 		console.log(result);
	// 	}).catch((err, data) => {
	// 		console.error(err);
	// 		console.log(data);
	// 	});
	// 	res.statusCode = 200;
	// 	res.type('text');
	// 	res.end('copy OK');
	// }

	// deleteImage(req, res) {
	// 	s3Service.removeFiles([
	// 		'daka-test/tmp/demo.png',
	// 		'daka-test/img/demo.png',
	// 	])
	// 		.then((result) => {
	// 			console.log(result);
	// 		})
	// 		.catch((err, data) => {
	// 			console.error(err);
	// 			console.log(data);
	// 		});
	// 	res.statusCode = 200;
	// 	res.type('text');
	// 	res.end('remove OK');
	// }

	// deleteFolder(req, res) {
	// 	s3Service.removeFolder(process.env.S3_BUCKET, process.env.S3_TMP_FOLDER)
	// 		.then((result) => {
	// 			console.log(result);

	// 			res.statusCode = 200;
	// 			res.type('text');
	// 			res.end('remove OK');
	// 		})
	// 		.catch((err, data) => {
	// 			console.error(err);
	// 			console.log(data);
	// 		});
	// }

	// createFolder(req, res) {
	// 	s3Service.createFolder(process.env.S3_BUCKET, 'te/le/ne')
	// 		.then((result) => {
	// 			console.log(result);
	// 		})
	// 		.catch((err, data) => {
	// 			console.error(err);
	// 			console.log(data);
	// 		});
	// 	res.statusCode = 200;
	// 	res.type('text');
	// 	res.end('crete OK');
	// }

};