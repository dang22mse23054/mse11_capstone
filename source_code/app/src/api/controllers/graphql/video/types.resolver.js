const { VideoStatus } = require('commonDir/constants');
// const { taskService } = require('apiDir/services');
const { videoService } = require('apiDir/services');

const Type = {

	categoryIds: (obj, args, context, info) => {
		if (obj.categoryIds === undefined) {
			return videoService.getCategoryIds(obj.videoId);
		}
		return obj.categoryIds.split(',');
	},

	categories: async (obj, args, context, info) => {
		const categoryIds = obj.categoryIds === undefined ? await videoService.getCategoryIds(obj.videoId) : (obj.categoryIds && obj.categoryIds.split(','));
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
};

module.exports = { Type };