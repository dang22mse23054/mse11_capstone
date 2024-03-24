const { ActionStatus } = require('commonDir/constants');
const { getUniqueList } = require('commonDir/utils/Utils');
const { Common, VideoStatus } = require('commonDir/constants');
const Video = require('modelDir/Video');
const VideoCategory = require('modelDir/VideoCategory');
const VideoBO = require('../db/business/VideoBO');
const VideoCategoryBO = require('../db/business/VideoCategoryBO');
const LogService = require('commonDir/logger');
const log = LogService.getInstance();
const Database = require('dbDir');
// const { ValidationError } = require('objection');
const moment = require('moment-timezone');
const ErrorObject = require('apiDir/dto/ErrorObject');

module.exports = class TaskService {

	constructor(outsideTransaction) {
		this.outTrx = outsideTransaction;
	}

	getVideo = (id) => {
		const videoBO = new VideoBO();
		try {
			return videoBO.getById(id);
		} catch (err) {
			log.error(err);
		}
		return null;
	}

	/**
	 * 
	 * @param {*} options 
	 * @param {*} cursor 
	 * @param {int} limit Maximum number of tasks returned, used for pagination.
	 * @returns 
	 */
	searchVideos = async (options, cursor, limit) => {
		const videoBO = new VideoBO();
		console.log('hehehehehe')

		try {
			// options = {
			// 	...options,
			// 	eagerLoad: '[processes]'
			// };
			let result;


			// There are 2 cases:
			// 1. useCache is false
			// 2. is navigator
			if (!result) {
				//const startTime = Date.now();
				result = await videoBO.search(options, cursor, limit);
			}

			return result;
		} catch (err) {
			log.error(err.stack);
		}
		return null;
	}


	insertOrUpdate = async (
		video,
		userInfo,
		alsoCreateTasks = true
	) => {

		// New transaction
		const trx = await Database.transaction();
		const videoBO = new VideoBO(trx);
		const videoCategoryBO = new VideoCategoryBO(trx);
		const rmFilesAfterFinished = []

		try {


			let videoId = Number(video.id) || 0;

			const currVideo = videoId > 0 ? await videoBO.getById(videoId) : null;

			if (video.id && !currVideo) {
				throw new CustomError('Invalid video id');
			}

			//#region Update video
			if (currVideo) {

				// check refFile
				const { rmFilePath, newFilePath } = this.processRefFile(
					videoId,
					currVideo.path,
					video.path
				);
				// if has old file
				if (rmFilePath) {
					// TODO -> change path
					rmFilesAfterFinished.push(`${process.env.S3_BUCKET}/${rmFilePath}`);
				}

				// update sched info
				await videoBO.update({
					...video,
					id: videoId,
					// update refFilePath again
					path: newFilePath,
				});

				await videoCategoryBO.delete({ videoId: videoId });
			}
			//#endregion

			//#region Create new video
			else {

				video.isEnabled = true;
				const newVideo = await videoBO.insert(
					Video.filterPropsData(video)
				);
				videoId = newVideo.id;

				const { rmFilePath, newFilePath } = this.processRefFile(
					videoId,
					null,
					video.path
				);

				if (rmFilePath) {
					rmFilesAfterFinished.push(`${process.env.S3_BUCKET}/${rmFilePath}`);
				}

				await videoBO.update(
					Video.filterPropsData({
						id: videoId,
						path: newFilePath,
					})
				);
			}
			//#endregion

			//#region === Add related data of Video ===

			// add schedule's media
			await videoCategoryBO.upsertMulti(
				video.categories.map((categoryId) =>
					VideoCategory.filterPropsData(
						{
							videoId,
							categoryId,
						},
						true
					)
				)
			);
			//#endregion

			// remove unused files
			await this.s3Service.removeFiles(rmFilesAfterFinished);

			trx.commit();
			sids.push(videoId);

		} catch (error) {
			const ind = schedule.inputIndex || index;
			let errorObj = {
				name: `Error when handling index ${ind}`,
				errMsg: 'Input value causes error'
			};
			if (error instanceof CustomError) {
				// Only return error msg of known error 
				// and hide the detail info since the security risk.
				errorObj.errMsg = `${error.message}`;
			}
			console.error(error);
			failScheds.push(errorObj);
			trx.rollback(Error(error.message));
		}
	};

	processRefFile = (scheduleId, currFilePath, newFilePath) => {
		// (NOTE: value of NULL can be updated, not accept `undefined`)
		if (newFilePath !== undefined && scheduleId) {
			//  ALREADY existed => set removed file later
			let rmFilePath = newFilePath != currFilePath ? currFilePath : null;

			// format: tmp/fileName
			// 	<=>  parts[0]/parts[1]
			const tmpPathRegex = new RegExp(`^${process.env.S3_TMP_FOLDER}/([\\w.]+)`);

			// only process `tmp/fileName`
			if (tmpPathRegex.test(newFilePath)) {
				const parts = tmpPathRegex.exec(newFilePath);

				if (newFilePath != currFilePath) {
					// === process add/update refFile === //

					// If newFilePath not NULL
					if (newFilePath) {
						// move from `tmp/` to `schedule/` and update newFilePath again
						const srcUri = `${process.env.S3_BUCKET}/${newFilePath}`;
						newFilePath = `${process.env.S3_SCHEDULE_FOLDER}/${scheduleId}/${parts[1]}`;

						// TODO -> move to video folder
						this.s3Service.copyFile(srcUri, `${process.env.S3_BUCKET}/${newFilePath}`);
					}
				}
			}

			return {
				rmFilePath,
				newFilePath,
			};
		}
		return {};
	};

	getCategoryIds = async (videoId) => {
		const videoCategoryBO = new VideoCategoryBO();
		try {
			const data = await videoCategoryBO.getByVideo(videoId, [
				'categoryId',
			]);

			if (data) {
				return data.map((item) => item.categoryId);
			}
		} catch (err) {
			log.error(err);
		}
		return null;
	};
};