const { VideoStatus } = require('commonDir/constants');
const { videoService } = require('apiDir/services');

const Type = {

	categoryIds: (obj, args, context, info) => {
		const videoId = obj.id;
		if (obj.categoryIds == null) {
			return videoService.getCategoryIds(videoId);
		}
		return obj.categoryIds.split(',');
	},

	categories: async (obj, args, context, info) => {
		const videoId = obj.id;
		const categoryIds = obj.categoryIds === undefined ? await videoService.getCategoryIds(videoId) : (obj.categoryIds && obj.categoryIds.split(','));
		if (categoryIds) {
			return context.categoryLoader.many(categoryIds);
		}
		return null;
	},

	status: (obj, args, context, info) => {
		const isEnabled = obj.isEnabled;
		const isDeleted = obj.deletedAt ? true : false;
		if (isDeleted) {
			return VideoStatus.STOPPED
		} else {
			return isEnabled ? VideoStatus.PLAYING : VideoStatus.PAUSED;
		}
	},

	statistic: (obj, args, context, info) => {
		const videoId = obj.id;
		const options = info.variableValues?.options;

		return videoService.getStatistic(videoId, options?.startDate, options?.endDate);
	},
};

module.exports = { Type };